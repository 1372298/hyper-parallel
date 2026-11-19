# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DeepSeek sparse-attention implementations and shared projection mathematics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_kl_loss import dsa_kl_loss
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention
from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import (
    IndexShareContext, IndexShareState, resolve_index_share_plan,
)


@dataclass(frozen=True)
class SparseAttentionSequenceMetadata:
    """Kernel endpoints and the consumed KV prefix for one invocation."""

    kv_prefix_length: int
    query_ends: torch.Tensor
    key_ends: torch.Tensor

    @property
    def sequence_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cumulative int32 Q/K endpoints for the existing CANN kernels."""
        return self.query_ends, self.key_ends

    @classmethod
    def build(
        cls, *, batch_size: int, local_seq_len: int, device: torch.device,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> SparseAttentionSequenceMetadata:
        """Resolve local cumulative endpoints through the shared packed parser.

        Args:
            batch_size: Number of batch-major rows.
            local_seq_len: Token count per row.
            device: Kernel metadata device.
            actual_seq_len: Cumulative packed endpoints without leading zero.
            kwargs: Supported packed aliases, including packed_seq_params.
        """
        total = batch_size * local_seq_len
        packed = dict(kwargs or {})
        if actual_seq_len is not None:
            packed["actual_seq_len"] = actual_seq_len
        queries, keys = resolve_packed_sequence_lengths(packed, total, total)
        if queries is None:
            queries = keys = [(row + 1) * local_seq_len for row in range(batch_size)]
        if len(queries) != len(keys):
            raise ValueError("Sparse attention requires the same number of packed Q and KV documents")
        return cls(
            local_seq_len,
            torch.tensor(queries, dtype=torch.int32, device=device),
            torch.tensor(keys, dtype=torch.int32, device=device),
        )


def _apply_attention_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the configured RoPE convention to an attention query/key pair."""
    if position_embeddings is None:
        return query, key
    # The rotary backend imports torch_npu; defer it until positional math is requested.
    from hyper_parallel.components.functional import (  # pylint: disable=import-outside-toplevel
        apply_rotary_pos_emb, apply_rotary_pos_emb_interleave,
    )

    cos, sin = position_embeddings
    rotary_op = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb
    return rotary_op(query, key, cos, sin, unsqueeze_dim=2)


def _apply_index_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    rope_head_dim: int,
    index_head_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the rotary portion of an indexer query/key pair."""
    if position_embeddings is None:
        return query, key
    # The rotary backend is optional for construction and CPU metadata/projection tests.
    from hyper_parallel.components.functional import (  # pylint: disable=import-outside-toplevel
        apply_rotary_pos_emb, apply_rotary_pos_emb_interleave,
    )

    cos, sin = position_embeddings
    split_sizes = (rope_head_dim, index_head_dim - rope_head_dim)
    query_rot, query_pass = torch.split(query, split_sizes, dim=-1)
    key_rot, key_pass = torch.split(key, split_sizes, dim=-1)
    rotary_op = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb
    query_rot, key_rot = rotary_op(query_rot, key_rot, cos, sin, unsqueeze_dim=2)
    return torch.cat((query_rot, query_pass), dim=-1), torch.cat((key_rot, key_pass), dim=-1)


def _absorb_query_projection(
    query: torch.Tensor,
    key_up_weight: torch.Tensor,
    *,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    input_dim: int,
    output_dim: int,
) -> torch.Tensor:
    """Absorb the key up-projection weight into query states."""
    query = query.permute(2, 0, 1, 3).reshape(num_heads, batch_size * seq_length, input_dim)
    query = torch.bmm(query, key_up_weight).view(num_heads, batch_size, seq_length, output_dim)
    return query.permute(1, 2, 0, 3)


def _restore_attention_projection(
    output: torch.Tensor,
    value_up_weight: torch.Tensor,
    *,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    kv_lora_rank: int,
    value_head_dim: int,
) -> torch.Tensor:
    """Apply the absorbed value up-projection and restore the hidden layout."""
    output = output[..., :kv_lora_rank]
    output = output.permute(2, 0, 1, 3).reshape(num_heads, batch_size * seq_length, kv_lora_rank)
    output = torch.bmm(output, value_up_weight)
    return output.view(num_heads, batch_size, seq_length, value_head_dim).permute(1, 2, 0, 3).reshape(
        batch_size, seq_length, -1
    )


def prepare_indexer_teacher(
    query: torch.Tensor, key: torch.Tensor,
    query_rope: torch.Tensor, key_rope: torch.Tensor,
    maximum: torch.Tensor, total: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Detach the supplied teacher tensors without changing their local layout.

    Args:
        query: Main-attention TND query.
        key: Dense per-head key or sparse shared latent key in TND.
        query_rope: TND query positional band.
        key_rope: TND key positional band.
        maximum: Dense N,T,1 or sparse 1,T,N softmax maximum.
        total: Corresponding softmax exponential sum.

    Returns:
        Detached Q, K, Q-rope, K-rope, maximum and sum for the existing KL kernel.
    """
    return tuple(tensor.detach() for tensor in (query, key, query_rope, key_rope, maximum, total))


def apply_mome(
    hidden_states: torch.Tensor,
    mome_mask: torch.Tensor | None,
    convolution: nn.Conv1d,
    *,
    fused: bool,
) -> torch.Tensor:
    """Apply masked causal depthwise convolution and residual."""
    if mome_mask is None:
        raise ValueError("mome_mask is required when MOME is enabled")
    if mome_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"mome_mask must have shape {hidden_states.shape[:2]}, "
            f"but got {tuple(mome_mask.shape)}"
        )

    mome_mask = mome_mask.to(device=hidden_states.device, dtype=torch.bool)
    padding = convolution.kernel_size[0] - 1
    padded_states = F.pad(hidden_states, (0, 0, padding, 0))
    if fused:
        # The MOME backend requires optional Omni operators, unlike V3.2 imports.
        from hyper_parallel.components.functional.aggregate_hidden import (  # pylint: disable=import-outside-toplevel
            aggregate_hidden,
        )

        padded_mask = F.pad(mome_mask, (padding, 0), value=False)
        weight = convolution.weight.squeeze(1).transpose(0, 1)
        mixed_states = aggregate_hidden(  # pylint: disable=not-callable
            padded_states.transpose(0, 1).contiguous(),
            weight,
            padded_mask,
        )
        mixed_states = mixed_states[padding:].transpose(0, 1).contiguous()
    else:
        mixed_states = convolution(padded_states.transpose(1, 2)).transpose(1, 2)
        mixed_states = mixed_states * mome_mask.unsqueeze(-1).to(mixed_states.dtype)
    return hidden_states + mixed_states


@dataclass(frozen=True)
class SparseAttentionInputs:
    """Invocation-local operands and the resolved dense/sparse training stage."""

    index_states: tuple[torch.Tensor, ...]
    attention_states: tuple[torch.Tensor, ...]
    metadata: SparseAttentionSequenceMetadata
    use_dense: bool


def _dense_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: tuple[torch.Tensor, torch.Tensor],
    *,
    need_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run one TND FA and return dense-indexer statistics."""
    query_ends, key_ends = (lengths.tolist() for lengths in sequence_lengths)
    q_flat, k_flat, v_flat = (tensor.flatten(0, 1) for tensor in (query, key, value))
    num_heads = q_flat.shape[1]
    causal_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device).triu(diagonal=1)
    import torch_npu  # pylint: disable=import-outside-toplevel

    result = torch_npu.npu_fusion_attention(
        q_flat.contiguous(), k_flat.contiguous(), v_flat.contiguous(), num_heads, "TND",
        pse=None, padding_mask=None, atten_mask=causal_mask, scale=module.scaling,
        pre_tockens=getattr(module, "pre_tockens", 1048576),
        next_tockens=getattr(module, "next_tockens", 0), keep_prob=1.0, inner_precise=0,
        actual_seq_qlen=query_ends, actual_seq_kvlen=key_ends, sparse_mode=3,
    )
    output = result[0].reshape(*query.shape[:3], value.shape[-1])
    if not need_stats:
        return output, None, None

    # TND FA stores each packed segment as [head, token, 8], while exposing a
    # physical [token, head, 8] shape. Dense KL consumes [head, token, 1].
    statistics = []
    starts = [0, *query_ends[:-1]]
    for statistic in result[1:3]:
        rows = statistic.reshape(-1, statistic.shape[-1])
        segments = [
            rows[start * num_heads : end * num_heads].reshape(
                num_heads, end - start, statistic.shape[-1]
            )[..., :1]
            for start, end in zip(starts, query_ends)
        ]
        statistics.append(torch.cat(segments, dim=1).detach())
    return output, statistics[0], statistics[1]


def _deepseek_v32_dense_attention(
    module: nn.Module,
    index_states: tuple[torch.Tensor, ...],
    attention_states: tuple[torch.Tensor, ...],
    sequence_lengths: tuple[torch.Tensor, torch.Tensor],
    *, loss_coefficient: float | None = None,
) -> torch.Tensor:
    """Run unabsorbed MLA FA and the fused dense indexer KL on local states."""
    query, compressed_kv, query_rope, key_rope = attention_states
    projected_kv = module.kv_b_proj(compressed_kv.squeeze(2)).view(
        *compressed_kv.shape[:2], query.shape[2], module.qk_nope_head_dim + module.v_head_dim,
    )
    key, value = projected_kv.split((module.qk_nope_head_dim, module.v_head_dim), dim=-1)
    key_rope = key_rope.expand(-1, -1, query.shape[2], -1)
    coefficient = module.dsa_loss_coeff if loss_coefficient is None else loss_coefficient
    need_stats = bool(module.training and not module.freeze_dsa and coefficient)
    output, maximum, total = _dense_flash_attention(
        module, torch.cat((query, query_rope), dim=-1), torch.cat((key, key_rope), dim=-1),
        value, sequence_lengths, need_stats=need_stats,
    )
    if need_stats:
        # Dense CANN operators are optional for sparse and Omni execution.
        # pylint: disable=import-outside-toplevel
        from hyper_parallel.components.functional.dsa_dense import (
            dsa_dense_indexer_kl_loss,
            dsa_dense_indexer_lse,
        )
        # pylint: enable=import-outside-toplevel

        query_lengths, key_lengths = sequence_lengths
        index_query, index_key, weights = (tensor.flatten(0, 1) for tensor in index_states)
        indexer_stats = dsa_dense_indexer_lse(
            index_query, index_key, weights, query_lengths=query_lengths, key_lengths=key_lengths,
        )
        teacher_query, teacher_key, teacher_qrope, teacher_krope, maximum, total = prepare_indexer_teacher(
            query.flatten(0, 1), key.flatten(0, 1),
            query_rope.flatten(0, 1), key_rope.flatten(0, 1), maximum, total,
        )
        loss = dsa_dense_indexer_kl_loss(
            index_query, index_key, weights, teacher_query, teacher_key,
            maximum, total, indexer_stats, scale=module.scaling,
            query_rope=teacher_qrope, key_rope=teacher_krope,
            query_lengths=query_lengths, key_lengths=key_lengths, loss_coefficient=coefficient,
        )
        output = aux_loss_auto_scale(output, loss)
    return output


class DeepseekV32SparseAttentionModule(nn.Module):
    """DeepSeek-V3.2 sparse attention with native configuration and source-module construction.

    It supports causal training and packed sequences. Packed boundaries must
    be supplied through ``actual_seq_len``. The optional ``attention_mask`` is
    assumed to be the standard causal mask created by Transformers; padding,
    custom attention masks, dropout, and KV cache are not supported.
    """

    def _initialize_dimensions(self, module: nn.Module | None) -> None:
        """Read main-attention and indexer dimensions from a source or config."""
        config = self.config
        indexer = getattr(module, "indexer", None)
        self.layer_idx = getattr(
            module, "layer_idx", getattr(module, "layer_number", getattr(config, "layer_idx", None)),
        )
        self.num_heads = getattr(module, "num_heads", config.num_attention_heads)
        self.q_lora_rank = getattr(module, "q_lora_rank", config.q_lora_rank)
        self.kv_lora_rank = getattr(module, "kv_lora_rank", config.kv_lora_rank)
        self.qk_rope_head_dim = getattr(module, "qk_rope_head_dim", config.qk_rope_head_dim)
        self.qk_nope_head_dim = getattr(module, "qk_nope_head_dim", config.qk_nope_head_dim)
        self.v_head_dim = getattr(module, "v_head_dim", config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            raise ValueError("DeepseekV32SparseAttention requires the Q-LoRA projection path")
        self.index_head_dim = getattr(indexer, "head_dim", config.index_head_dim)
        self.num_index_heads = getattr(
            indexer,
            "n_heads",
            getattr(config, "index_n_heads", getattr(config, "index_num_attention_heads", None)),
        )
        if self.num_index_heads is None:
            raise ValueError("DeepseekV32SparseAttention requires the number of indexer heads")
        self.index_topk = getattr(indexer, "index_topk", config.index_topk)

    def _initialize_options(self, module: nn.Module | None) -> None:
        """Read sparse attention options while preserving source scaling and RoPE conventions."""
        config = self.config
        self.scaling = getattr(module, "scaling", None)
        if self.scaling is None:
            self.scaling = self._config_scaling(config)
        attention_dropout = getattr(module, "attention_dropout", getattr(config, "attention_dropout", 0.0))
        if isinstance(attention_dropout, nn.Dropout):
            attention_dropout = attention_dropout.p
        if attention_dropout != 0.0:
            raise ValueError(
                "DeepseekV32SparseAttention requires attention_dropout=0 because "
                "the sparse-attention operator does not apply dropout"
            )
        source_rotary_interleaved = getattr(module, "rotary_interleaved", getattr(config, "rotary_interleaved", None))
        if source_rotary_interleaved is None:
            source_rotary_interleaved = getattr(config, "rope_interleave", None)
        if source_rotary_interleaved is None:
            source_rotary_interleaved = getattr(config, "rope_interleaved", None)
        self.rotary_interleaved = bool(True if source_rotary_interleaved is None else source_rotary_interleaved)
        self.index_rotary_interleaved = bool(
            getattr(
                getattr(module, "indexer", None), "rotary_interleaved",
                getattr(config, "index_rope_interleaved", False),
            )
        )
        self.dsa_loss_coeff = getattr(config, "dsa_loss_coeff", 0.0)
        self.freeze_dsa = bool(getattr(config, "freeze_DSA", getattr(config, "freeze_dsa", False)))
        self.dsa_dense_warm_up = bool(
            getattr(module, "dsa_dense_warm_up", getattr(config, "dsa_dense_warm_up", False))
        )
        if getattr(module, "param_sink_number", getattr(config, "param_sink_number", 0)) > 0:
            raise ValueError("DeepseekV32SparseAttention does not support the parameter-sink path")
        if getattr(module, "use_mome", getattr(config, "use_mome", False)):
            raise ValueError("DeepseekV32SparseAttention does not support the MOME path")

    def _config_scaling(self, config: Any) -> float:
        """Apply the source-compatible YaRN factor for native construction."""
        scaling = self.qk_head_dim**-0.5
        rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default" and rope.get("mscale_all_dim", 0):
            factor = rope["factor"]
            mscale = 1.0 if factor <= 1 else 1.0 + 0.1 * rope["mscale_all_dim"] * math.log(factor)
            scaling *= mscale * mscale
        return getattr(config, "scaling", scaling)

    def _initialize_config_layers(self, config: Any) -> None:
        """Allocate the native parameter tree without constructing a Transformers model."""
        dtype = getattr(config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."))
        factory_kwargs = {"device": getattr(config, "device", None), "dtype": dtype}
        bias = getattr(config, "attention_bias", False)
        self._q_latent_output_size = self.q_lora_rank
        self._kv_latent_output_size = self.kv_lora_rank + self.qk_rope_head_dim
        self.linear_qkv = nn.Linear(
            config.hidden_size, self._q_latent_output_size + self._kv_latent_output_size,
            bias=bias, **factory_kwargs,
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=eps, **factory_kwargs)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=eps, **factory_kwargs)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, **factory_kwargs,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=bias, **factory_kwargs,
        )
        self.indexer = None
        if not self.is_index_leader:
            return
        self.indexer = nn.Module()
        self.indexer.n_heads, self.indexer.head_dim = self.num_index_heads, self.index_head_dim
        self.indexer.index_topk = self.index_topk
        self.indexer.wq_b = nn.Linear(
            self.q_lora_rank, self.num_index_heads * self.index_head_dim, bias=False, **factory_kwargs,
        )
        self.indexer.wk = nn.Linear(config.hidden_size, self.index_head_dim, bias=False, **factory_kwargs)
        self.indexer.k_norm = nn.LayerNorm(self.index_head_dim, eps=1e-6, **factory_kwargs)
        self.indexer.weights_proj = nn.Linear(config.hidden_size, self.num_index_heads, bias=False, **factory_kwargs)

    def _initialize_source_layers(self, module: nn.Module) -> None:
        """Fuse loaded down-projection values and retain the remaining Parameters."""
        q_a, kv_a = module.q_a_proj, module.kv_a_proj_with_mqa
        self._q_latent_output_size, self._kv_latent_output_size = q_a.out_features, kv_a.out_features
        can_fuse = (
            q_a.in_features == kv_a.in_features
            and q_a.weight.requires_grad == kv_a.weight.requires_grad
            and q_a.weight.device == kv_a.weight.device
            and q_a.weight.dtype == kv_a.weight.dtype
            and (q_a.bias is None) == (kv_a.bias is None)
            and (q_a.bias is None or q_a.bias.requires_grad == kv_a.bias.requires_grad)
        )
        if not can_fuse:
            raise ValueError("Q and KV latent projections cannot be represented by one fused projection")
        self.linear_qkv = nn.Linear(
            q_a.in_features, q_a.out_features + kv_a.out_features, bias=q_a.bias is not None,
            device=q_a.weight.device, dtype=q_a.weight.dtype,
        )
        self.linear_qkv.weight.requires_grad_(q_a.weight.requires_grad)
        if self.linear_qkv.bias is not None:
            self.linear_qkv.bias.requires_grad_(q_a.bias.requires_grad)
        if not q_a.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((q_a.weight, kv_a.weight), dim=0))
                if self.linear_qkv.bias is not None:
                    self.linear_qkv.bias.copy_(torch.cat((q_a.bias, kv_a.bias), dim=0))
        for name in ("q_a_layernorm", "kv_a_layernorm", "q_b_proj"):
            setattr(self, name, getattr(module, name))
        if module.kv_b_proj.bias is not None:
            raise ValueError("Absorbed KV projection requires a bias-free Linear")
        self.kv_b_proj = module.kv_b_proj
        self.o_proj = module.o_proj
        self.indexer = module.indexer if self.is_index_leader else None

    def _reuse_local_member(self, module: DeepseekV32SparseAttentionModule) -> None:
        """Promote a loaded unbound member without re-fusing or reinitializing values."""
        if getattr(module, "_hp_parallel_binding", None) is not None:
            raise ValueError("DeepSeek-V3.2 promotion must precede parallel binding and FSDP")
        fields = (
            "config", "layer_idx", "num_heads", "q_lora_rank", "kv_lora_rank",
            "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "qk_head_dim", "index_head_dim",
            "num_index_heads", "index_topk", "scaling", "rotary_interleaved", "index_rotary_interleaved",
            "dsa_loss_coeff", "freeze_dsa", "dsa_dense_warm_up",
            "index_share_plan", "is_index_leader",
            "_q_latent_output_size", "_kv_latent_output_size",
        )
        for name in fields:
            setattr(self, name, getattr(module, name))
        for name in ("linear_qkv", "q_a_layernorm", "kv_a_layernorm", "q_b_proj", "kv_b_proj", "o_proj", "indexer"):
            setattr(self, name, getattr(module, name))

    def __init__(
        self, *, module: nn.Module | None = None, config: Any = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        cp_strategy: str = "kv_allgather", overlap: bool = True, dense_warmup: bool | None = None,
    ) -> None:
        """Construct native sparse attention parameters, convert a source, or promote a local member.

        Args:
            module: Source attention or an unbound concrete family member.
            config: Native dimensions/options, exclusive with module; optional dtype/device
                select parameter placement. Otherwise the active Torch defaults apply.
            module_fqn: Source path supplied by replacement.
            context: Logical construction context, without execution resources.
            cp_strategy: Reserved CP request retained for family promotion.
            overlap: Communication scheduling option retained by the family request.
            dense_warmup: None follows the mutable training stage; bool pins it.
        """
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("DeepSeek-V3.2 sparse attention requires exactly one of config or module")
        if cp_strategy != "kv_allgather":
            raise ValueError("DeepSeek-V3.2 supports only cp_strategy='kv_allgather'")
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        self.cp_strategy, self.overlap = cp_strategy, overlap
        self._dense_warmup_override = dense_warmup
        if isinstance(module, DeepseekV32SparseAttentionModule):
            self._reuse_local_member(module)
        else:
            self.config = config if module is None else module.config
            if hasattr(self.config, "indexer_types"):
                raise TypeError(
                    "DeepSeek-V3.2 sparse attention does not support shared-indexer or three-value output contracts"
                )
            self._initialize_dimensions(module)
            self._initialize_options(module)
            self.index_share_plan = resolve_index_share_plan(self.config)
            if self.index_share_plan.enabled and (
                type(self.layer_idx) is not int or not 0 <= self.layer_idx < len(self.index_share_plan.leaders)
            ):
                raise ValueError("Indexer sharing requires a zero-based global layer_idx within num_hidden_layers")
            self.is_index_leader = (
                not self.index_share_plan.enabled or self.index_share_plan.leaders[self.layer_idx] == self.layer_idx
            )
            if module is None:
                self._initialize_config_layers(config)
            else:
                self._initialize_source_layers(module)
        if module is not None:
            self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize native/meta parameters without resetting converted source weights."""
        if not self._config_initialization:
            return
        for child in (
            self.linear_qkv, self.q_a_layernorm, self.kv_a_layernorm, self.q_b_proj,
            self.kv_b_proj, self.o_proj,
        ):
            child.reset_parameters()
        if self.indexer is not None:
            for child in self.indexer.children():
                child.reset_parameters()

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        if self._config_initialization:
            return []
        transforms: list[WeightConverter] = [
            WeightConverter(
                source_patterns=[
                    "q_a_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                ],
                target_patterns="linear_qkv.weight",
                operations=[
                    ConcatenateWithSections(
                        sections=(
                            self._q_latent_output_size,
                            self._kv_latent_output_size,
                        ),
                        dim=0,
                    )
                ],
            ),
        ]
        if self.linear_qkv.bias is not None:
            transforms.insert(
                1,
                WeightConverter(
                    source_patterns=["q_a_proj.bias", "kv_a_proj_with_mqa.bias"],
                    target_patterns="linear_qkv.bias",
                    operations=[
                        ConcatenateWithSections(
                            sections=(
                                self._q_latent_output_size,
                                self._kv_latent_output_size,
                            ),
                            dim=0,
                        )
                    ],
                ),
            )
        return transforms

    @staticmethod
    def _invalid_position_ids(
        position_ids: torch.Tensor | None,
        batch_size: int,
        seq_length: int,
    ) -> bool:
        """Return whether position IDs violate the supported dense layout."""
        if position_ids is None:
            return False
        if position_ids.ndim != 2:
            return True
        return position_ids.shape[0] not in (1, batch_size) or position_ids.shape[1] != seq_length

    @staticmethod
    def _validate_forward_inputs(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None,
        position_ids: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> None:
        """Validate inputs supported by the DeepSeek sparse-attention path."""
        if hidden_states.ndim != 3:
            raise ValueError("DeepSeek-V3.2 hidden_states must have shape [batch, sequence, hidden]")
        batch_size, seq_length = hidden_states.shape[:-1]
        if past_key_values is not None:
            raise NotImplementedError("DeepseekV32SparseAttention does not support KV cache")
        if DeepseekV32SparseAttentionModule._invalid_position_ids(position_ids, batch_size, seq_length):
            raise ValueError("position_ids must have shape [1 or batch_size, sequence_length]")
        if attention_mask is not None:
            expected_shape = (batch_size, 1, seq_length, seq_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    "DeepseekV32SparseAttention supports only a Transformers 4D causal mask; "
                    "padding and custom masks are not supported"
                )
        if kwargs.pop("output_attentions", False):
            raise NotImplementedError("DeepseekV32SparseAttention does not expose attention weights")

    def _project_attention_states(
        self, hidden_states: torch.Tensor, *, use_dense: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Project local states inside the attention/block FSDP boundary."""
        batch_size, seq_length = hidden_states.shape[:-1]
        q_latent, latent_kv, key_rope = torch.split(
            self.linear_qkv(hidden_states),
            (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim), dim=-1,
        )
        q_resid = self.q_a_layernorm(q_latent)
        projected = self.q_b_proj(q_resid)
        heads = projected.shape[-1] // self.qk_head_dim
        query, query_rope = projected.view(
            batch_size, seq_length, heads, self.qk_head_dim,
        ).split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        if not use_dense:
            weight = self.kv_b_proj.weight.view(
                heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank,
            )
            query = _absorb_query_projection(
                query, weight[:, :self.qk_nope_head_dim], num_heads=heads,
                batch_size=batch_size, seq_length=seq_length,
                input_dim=self.qk_nope_head_dim, output_dim=self.kv_lora_rank,
            )
        latent_kv = self.kv_a_layernorm(latent_kv).view(batch_size, seq_length, 1, self.kv_lora_rank)
        key_rope = key_rope.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        return q_resid, (query, latent_kv, query_rope, key_rope)

    def _project_index_states(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build indexer query, key, and merge-weight states."""
        batch_size, seq_length = hidden_states.shape[:-1]
        index_query = self.indexer.wq_b(q_resid.detach()).view(
            batch_size, seq_length, self.num_index_heads, self.index_head_dim
        )
        index_key = self.indexer.k_norm(self.indexer.wk(hidden_states.detach())).unsqueeze(2)
        merge_weight = self.indexer.weights_proj(hidden_states.detach())
        # Avoid intermediate low-precision rounding at the TopK boundary.
        weight_scale = (self.num_index_heads * self.index_head_dim)**-0.5
        merge_weight = (merge_weight.float() * weight_scale).to(merge_weight.dtype)
        index_query, index_key = _apply_index_rope(
            index_query,
            index_key,
            position_embeddings,
            rope_head_dim=self.qk_rope_head_dim,
            index_head_dim=self.index_head_dim,
            interleaved=self.index_rotary_interleaved,
        )
        return index_query, index_key, merge_weight

    def _apply_auxiliary_loss(
        self,
        attn_output: torch.Tensor,
        index_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        softmax_stats: tuple[torch.Tensor, torch.Tensor],
        actual_q_len: torch.Tensor,
        actual_kv_len: torch.Tensor,
        *, loss_coefficient: float | None = None,
    ) -> torch.Tensor:
        """Attach the sparse attention KL auxiliary loss when training enables it."""
        coefficient = self.dsa_loss_coeff if loss_coefficient is None else loss_coefficient
        if not (self.training and not self.freeze_dsa and coefficient):
            return attn_output
        query_tnd, key_tnd, q_rot_tnd, k_rot_tnd = (
            tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
            for tensor in attention_states
        )
        query_tnd, key_tnd, q_rot_tnd, k_rot_tnd, maximum, total = prepare_indexer_teacher(
            query_tnd, key_tnd, q_rot_tnd, k_rot_tnd, *softmax_stats,
        )
        aux_loss = dsa_kl_loss(
            *index_states, query_tnd, key_tnd,
            topk_indices, maximum, total, q_rot_tnd, k_rot_tnd,
            actual_q_len, actual_kv_len, self.scaling, coefficient,
            backend="cann",
        )
        return aux_loss_auto_scale(attn_output, aux_loss)

    def use_dense_attention(self) -> bool:
        """Read the current training stage unless an explicit override pins it."""
        return self.dsa_dense_warm_up if self._dense_warmup_override is None else self._dense_warmup_override

    def preflight_attention(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
        past_key_values: Any | None, position_ids: torch.Tensor | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: dict[str, Any], *, use_dense: bool,
    ) -> SparseAttentionSequenceMetadata:
        """Validate local inputs and construct packed geometry before projection."""
        del use_dense
        self._validate_forward_inputs(hidden_states, attention_mask, past_key_values, position_ids, kwargs)
        return SparseAttentionSequenceMetadata.build(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, actual_seq_len=actual_seq_len, kwargs=kwargs,
        )

    def prepare_attention(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        *, use_dense: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Run projections and both independent RoPE conventions on local tokens."""
        q_resid, states = self._project_attention_states(hidden_states, use_dense=use_dense)
        query, latent_kv, query_rope, key_rope = states
        query_rope, key_rope = _apply_attention_rope(
            query_rope, key_rope, position_embeddings, interleaved=self.rotary_interleaved,
        )
        index_states = (
            self._project_index_states(hidden_states, q_resid, position_embeddings) if self.is_index_leader else ()
        )
        return index_states, (query, latent_kv, query_rope, key_rope)

    def prepare_attention_inputs(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        metadata: SparseAttentionSequenceMetadata, *, use_dense: bool,
    ) -> SparseAttentionInputs:
        """Prepare local operands while retaining one invocation's resolved mode."""
        index_states, attention_states = self.prepare_attention(
            hidden_states, position_embeddings, use_dense=use_dense,
        )
        return SparseAttentionInputs(index_states, attention_states, metadata, use_dense)

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Return ready operands; local attention needs no sequence exchange."""
        return inputs

    def run_attention(
        self, index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
        sequence_lengths: tuple[torch.Tensor, torch.Tensor], *, use_dense: bool,
        index_share_context: IndexShareContext | None = None,
    ) -> torch.Tensor:
        """Run selection once per group and supervise with each served layer's teacher."""
        coefficient = self.dsa_loss_coeff
        plan = self.index_share_plan
        if plan.enabled:
            if not isinstance(index_share_context, IndexShareContext):
                raise ValueError("Indexer sharing requires a fresh IndexShareContext for each model forward")
            leader = plan.leaders[self.layer_idx]
            if plan.loss_mode == "served":
                coefficient /= plan.group_size(self.layer_idx)
            elif not self.is_index_leader:
                coefficient = 0.0
            if not self.is_index_leader:
                state = index_share_context.groups.get(leader)
                if state is None:
                    raise ValueError(f"Shared layer {self.layer_idx} requires leader {leader} to run first")
                if (state.use_dense != use_dense or state.index_topk != self.index_topk
                        or state.index_states[0].shape[:2] != attention_states[0].shape[:2]
                        or state.index_states[0].device != attention_states[0].device):
                    raise ValueError("Shared Indexer requires matching stage, Top-K, batch/sequence and device")
                index_states, topk = state.index_states, state.topk
        if self.is_index_leader:
            if use_dense:
                topk = None
            else:
                topk, *_ = dsa_indexer(
                    *index_states, *sequence_lengths, self.index_topk, backend="cann",
                )
            if plan.enabled:
                index_share_context.groups[self.layer_idx] = IndexShareState(
                    index_states, None if topk is None else topk.detach(), use_dense, self.index_topk,
                )
        if use_dense:
            return _deepseek_v32_dense_attention(
                self, index_states, attention_states, sequence_lengths, loss_coefficient=coefficient,
            ).flatten(2)
        flattened = tuple(tensor.flatten(0, 1) for tensor in index_states)
        output, maximum, total = dsa_sparse_attention(
            *attention_states, topk, self.scaling, *sequence_lengths, backend="cann",
        )
        output = self._apply_auxiliary_loss(
            output, tuple(flattened), attention_states, topk, (maximum, total), *sequence_lengths,
            loss_coefficient=coefficient,
        )
        heads = output.shape[2]
        weight = self.kv_b_proj.weight.view(
            heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank,
        )
        return _restore_attention_projection(
            output, weight[:, self.qk_nope_head_dim:].transpose(1, 2), num_heads=heads,
            batch_size=output.shape[0], seq_length=output.shape[1],
            kv_lora_rank=self.kv_lora_rank, value_head_dim=self.v_head_dim,
        )

    def project_output(self, output: torch.Tensor) -> torch.Tensor:
        """Run the output Linear before the shared TP boundary."""
        return self.o_proj(output)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None, past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        index_share_context: IndexShareContext | None = None, **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run causal or packed sparse attention while retaining the Transformers output ABI."""
        use_dense = self.use_dense_attention()
        if self.index_share_plan.enabled and not isinstance(index_share_context, IndexShareContext):
            raise ValueError("Indexer sharing requires a fresh IndexShareContext for each model forward")
        if (self.index_share_plan.enabled and self.training and self.dsa_loss_coeff and not self.freeze_dsa
                and not torch.is_grad_enabled()):
            raise ValueError("Indexer sharing training requires grad-enabled forward and non-reentrant checkpointing")
        metadata = self.preflight_attention(
            hidden_states, attention_mask, past_key_values, position_ids, actual_seq_len, kwargs,
            use_dense=use_dense,
        )
        inputs = self.prepare_attention_inputs(
            hidden_states, position_embeddings, metadata, use_dense=use_dense,
        )
        inputs = self.exchange_attention(inputs)
        output = self.run_attention(
            inputs.index_states, inputs.attention_states, inputs.metadata.sequence_lengths, use_dense=inputs.use_dense,
            index_share_context=index_share_context,
        )
        del inputs
        return self.project_output(output), None


def _projection_output(result: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
    """Read the activation from native Linear or source tuple-returning projections."""
    return result[0] if isinstance(result, tuple) else result


class DeepseekSparseAttentionModule(nn.Module):
    """Sparse attention with flat source-compatible parameters and native construction.

    Packed boundaries are supplied through ``actual_seq_len``. The local member
    supports MOME and parameter sinks; parallel family members require both off.
    Attention masks, KV reuse, KV cache, and dense warm-up are not supported.
    """

    _LAYER_NAMES = (
        "linear_qkv", "q_layernorm", "k_layernorm", "linear_qb", "linear_kvb", "linear_proj",
        "index_linear_qb", "index_linear_k", "index_k_layernorm", "linear_merge_weight",
    )

    def __init__(
        self, *, module: nn.Module | None = None, config: Any = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        cp_strategy: str = "kv_allgather", overlap: bool = True, dense_warmup: bool | None = None,
    ) -> None:
        """Construct native parameters or reuse a compatible source parameter tree.

        Args:
            module: Source attention or an unbound concrete family member.
            config: Native dimensions/options, exclusive with module. Optional
                dtype/device select placement; otherwise Torch defaults apply.
            module_fqn: Source path supplied by replacement.
            context: Logical construction context, without execution resources.
            cp_strategy: CP request retained for family promotion.
            overlap: Communication scheduling option retained for promotion.
            dense_warmup: Only None and False are supported by this implementation.
        """
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("DeepseekSparseAttention requires exactly one of config or module")
        if cp_strategy != "kv_allgather":
            raise ValueError("DeepseekSparseAttention supports only cp_strategy='kv_allgather'")
        if (
            isinstance(module, DeepseekSparseAttentionModule)
            and getattr(module, "_hp_parallel_binding", None) is not None
        ):
            raise ValueError("Sparse-attention promotion must precede parallel binding and FSDP")
        self.config = config if module is None else module.config
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        self.cp_strategy, self.overlap = cp_strategy, overlap
        self._dense_warmup_override = dense_warmup
        self._initialize_dimensions(module)
        self._initialize_options(module)
        if dense_warmup or self.dsa_dense_warm_up:
            raise NotImplementedError("DeepseekSparseAttention does not support dense warm-up")
        if module is None:
            self._initialize_config_layers()
        else:
            for name in self._LAYER_NAMES:
                setattr(self, name, getattr(module, name))
            if self.use_mome:
                for name in ("qa_conv", "compresskv_conv", "o_conv"):
                    setattr(self, name, getattr(module, name))
            if self.param_sink_number > 0:
                self.param_sink_k_pe = module.param_sink_k_pe
                self.param_sink_compressed_kv = module.param_sink_compressed_kv
            self.train(module.training)

    def _initialize_dimensions(self, module: nn.Module | None) -> None:
        """Read dimensions without requiring native-config aliases on source modules."""
        config = self.config
        self.layer_number = getattr(module, "layer_number", getattr(config, "layer_number", None))
        self.num_heads = getattr(module, "num_heads", getattr(config, "num_attention_heads", None))
        for name in (
            "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "index_head_dim",
        ):
            setattr(self, name, getattr(module, name, getattr(config, name, None)))
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            raise ValueError("DeepseekSparseAttention requires the Q-LoRA projection path")
        self.num_index_heads = getattr(
            module, "num_index_heads",
            getattr(config, "index_n_heads", getattr(
                config, "index_num_attention_heads", getattr(config, "num_index_heads", None),
            )),
        )
        if self.num_index_heads is None:
            raise ValueError("DeepseekSparseAttention requires the number of indexer heads")
        self.index_topk = getattr(module, "index_topk", getattr(config, "index_topk", None))

    def _initialize_options(self, module: nn.Module | None) -> None:
        """Preserve source options and supply matching defaults for native construction."""
        def option(name: str, default: Any) -> Any:
            return getattr(module, name, getattr(self.config, name, default))

        if option("mla_mm_split", False):
            raise NotImplementedError("DeepseekSparseAttention does not support split latent projections")
        self.rotary_interleaved = bool(option("rotary_interleaved", False))
        self.dsa_loss_coeff = option("dsa_loss_coeff", 0.0)
        self.freeze_dsa = bool(option("freeze_dsa", getattr(self.config, "freeze_DSA", False)))
        self.dsa_dense_warm_up = bool(option("dsa_dense_warm_up", False))
        self.use_flash_attn = bool(option("use_flash_attn", True))
        self.use_mome = bool(option("use_mome", False))
        self.use_fused_mome = bool(option("use_fused_mome", False))
        self.param_sink_number = int(option("param_sink_number", 0))
        self.param_sink_scalar = option("param_sink_scalar", None)
        self.apply_fa_rescale = bool(option("apply_fa_rescale", False))
        dropout = option("attention_dropout", 0.0)
        self.attention_dropout = dropout if isinstance(dropout, nn.Dropout) else nn.Dropout(dropout)
        if self.param_sink_number > 0 and (self.param_sink_scalar or not self.apply_fa_rescale):
            raise NotImplementedError(
                "DeepseekSparseAttention parameter sinks require apply_fa_rescale=True and no scalar sink"
            )

    def _initialize_config_layers(self) -> None:
        """Allocate native Linear, normalization, and optional local MOME parameters."""
        config = self.config
        dtype = getattr(config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."))
        factory_kwargs = {"device": getattr(config, "device", None), "dtype": dtype}
        bias = getattr(config, "attention_bias", False)
        self.linear_qkv = nn.Linear(
            config.hidden_size, self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias, **factory_kwargs,
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_layernorm = nn.RMSNorm(self.q_lora_rank, eps=eps, **factory_kwargs)
        self.k_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=eps, **factory_kwargs)
        self.linear_qb = nn.Linear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, **factory_kwargs,
        )
        self.linear_kvb = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, **factory_kwargs,
        )
        self.linear_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=bias, **factory_kwargs,
        )
        self.index_linear_qb = nn.Linear(
            self.q_lora_rank, self.num_index_heads * self.index_head_dim, bias=False, **factory_kwargs,
        )
        self.index_linear_k = nn.Linear(config.hidden_size, self.index_head_dim, bias=False, **factory_kwargs)
        self.index_k_layernorm = nn.LayerNorm(self.index_head_dim, eps=1e-6, **factory_kwargs)
        self.linear_merge_weight = nn.Linear(config.hidden_size, self.num_index_heads, bias=False, **factory_kwargs)
        if self.use_mome:
            kernel_size = getattr(config, "mome_kernel_size", None)
            if kernel_size is None:
                raise ValueError("Native MOME construction requires config.mome_kernel_size")
            for name, channels in (
                ("qa_conv", self.q_lora_rank), ("compresskv_conv", self.kv_lora_rank),
                ("o_conv", self.num_heads * self.v_head_dim),
            ):
                setattr(self, name, nn.Conv1d(
                    channels, channels, kernel_size, groups=channels,
                    bias=getattr(config, "mome_bias", False), **factory_kwargs,
                ))
        if self.param_sink_number > 0:
            self.param_sink_k_pe = nn.Parameter(torch.zeros(
                self.param_sink_number, self.qk_rope_head_dim, **factory_kwargs,
            ))
            self.param_sink_compressed_kv = nn.Parameter(torch.zeros(
                self.param_sink_number, self.kv_lora_rank, **factory_kwargs,
            ))

    def reset_parameters(self) -> None:
        """Initialize native/meta parameters without resetting converted source weights."""
        if not self._config_initialization:
            return
        names = self._LAYER_NAMES + (("qa_conv", "compresskv_conv", "o_conv") if self.use_mome else ())
        for name in names:
            getattr(self, name).reset_parameters()
        if self.param_sink_number > 0:
            nn.init.zeros_(self.param_sink_k_pe)
            nn.init.zeros_(self.param_sink_compressed_kv)

    def make_transforms(self) -> list[WeightConverter]:
        """Keep source checkpoint names because flat Parameters are reused unchanged."""
        return []

    def _prepare_param_sink(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the local sink key and value in BSND layout."""
        sink_rotary_key = self.param_sink_k_pe.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)
        sink_latent = self.param_sink_compressed_kv.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)
        sink_latent = self.k_layernorm(sink_latent)
        sink_value = (
            F.pad(sink_latent, (0, self.qk_rope_head_dim))
            if self.use_flash_attn and self.qk_rope_head_dim > 0 else sink_latent
        )
        return torch.cat((sink_latent, sink_rotary_key), dim=-1), sink_value

    def _validate_forward_inputs(
        self, attention_mask: torch.Tensor | None, kv_reuse_states: Any | None,
        past_key_values: Any | None, cache_position: torch.Tensor | None,
        output_attentions: bool,
    ) -> None:
        """Reject unsupported contracts before projection or communication."""
        if attention_mask is not None:
            raise ValueError(
                "DeepseekSparseAttention does not consume attention_mask; "
                "pass packed sequence boundaries through actual_seq_len"
            )
        if kv_reuse_states is not None:
            raise NotImplementedError("DeepseekSparseAttention does not support KV reuse")
        if past_key_values is not None or cache_position is not None:
            raise NotImplementedError("DeepseekSparseAttention does not support KV cache")
        if output_attentions:
            raise NotImplementedError("DeepseekSparseAttention does not expose attention weights")

    def _project_attention_states(
        self, hidden_states: torch.Tensor, mome_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Project local head shards inside the attention/block FSDP boundary."""
        batch_size, seq_length = hidden_states.shape[:-1]
        latent_states = _projection_output(self.linear_qkv(hidden_states))
        q_latent, kv_nope, key_rope = latent_states.split(
            (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim), dim=-1,
        )
        if self.use_mome:
            q_latent = apply_mome(q_latent, mome_mask, self.qa_conv, fused=False)
            kv_nope = apply_mome(kv_nope, mome_mask, self.compresskv_conv, fused=False)
        q_resid = self.q_layernorm(q_latent)
        projected = _projection_output(self.linear_qb(q_resid.contiguous()))
        heads = projected.shape[-1] // self.qk_head_dim
        query, query_rope = projected.view(batch_size, seq_length, heads, self.qk_head_dim).split(
            (self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1,
        )
        weight = self.linear_kvb.weight.view(heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        query = _absorb_query_projection(
            query, weight[:, :self.qk_nope_head_dim], num_heads=heads,
            batch_size=batch_size, seq_length=seq_length,
            input_dim=self.qk_nope_head_dim, output_dim=self.kv_lora_rank,
        )
        kv_nope = self.k_layernorm(kv_nope).view(batch_size, seq_length, 1, self.kv_lora_rank)
        key_rope = key_rope.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        return q_resid, (query, kv_nope, query_rope, key_rope)

    def _project_index_states(
        self, hidden_states: torch.Tensor, q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build replicated indexer states and apply their rotary embedding."""
        batch_size, seq_length = hidden_states.shape[:-1]
        index_query = _projection_output(self.index_linear_qb(q_resid.detach()))
        index_query = index_query.view(batch_size, seq_length, self.num_index_heads, self.index_head_dim)
        index_key = _projection_output(self.index_linear_k(hidden_states.detach()))
        index_key = self.index_k_layernorm(index_key.unsqueeze(2))
        merge_weight = _projection_output(self.linear_merge_weight(hidden_states.detach()))
        merge_weight = merge_weight * self.num_index_heads**-0.5 * self.index_head_dim**-0.5
        index_query, index_key = _apply_index_rope(
            index_query, index_key, position_embeddings, rope_head_dim=self.qk_rope_head_dim,
            index_head_dim=self.index_head_dim, interleaved=self.rotary_interleaved,
        )
        return index_query, index_key, merge_weight

    def _run_sparse_attention(
        self, attention_states: tuple[torch.Tensor, ...], topk_indices: torch.Tensor,
        actual_q_len: torch.Tensor, actual_kv_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sparse attention with optional local parameter-sink rescaling."""
        query, latent_kv, query_rope, key_rope = attention_states
        sparse_scale = self.qk_head_dim**-0.5
        if self.param_sink_number <= 0:
            return dsa_sparse_attention(  # pylint: disable=not-callable
                query, latent_kv, query_rope, key_rope, topk_indices,
                sparse_scale, actual_q_len, actual_kv_len,
            )
        batch_size, seq_length, heads = query.shape[:3]
        sink_key, sink_value = self._prepare_param_sink(batch_size)
        # Sink rescaling owns optional Omni dependencies; ordinary sparse attention does not.
        from hyper_parallel.components.functional.dsa_sparse_attention_rescale import (  # pylint: disable=C0415
            dsa_sparse_attention_rescale,
        )

        return dsa_sparse_attention_rescale(  # pylint: disable=not-callable
            query, latent_kv, query_rope, key_rope, sink_key, sink_value,
            topk_indices, batch_size, seq_length, heads, sparse_scale,
            1 - self.attention_dropout.p, actual_q_len, actual_kv_len,
        )

    def _apply_auxiliary_loss(
        self, output: torch.Tensor, index_states: tuple[torch.Tensor, ...],
        attention_states: tuple[torch.Tensor, ...], topk_indices: torch.Tensor,
        maximum: torch.Tensor, total: torch.Tensor, sequence_lengths: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Attach the local-teacher Indexer loss through the shared loss scaler."""
        if not (self.training and not self.freeze_dsa and self.dsa_loss_coeff):
            return output
        query, key, query_rope, key_rope = (
            tensor.flatten(0, 1) for tensor in attention_states
        )
        query, key, query_rope, key_rope, maximum, total = prepare_indexer_teacher(
            query, key, query_rope, key_rope, maximum, total,
        )
        loss = dsa_kl_loss(  # pylint: disable=not-callable
            *index_states, query, key, topk_indices, maximum, total, query_rope, key_rope,
            *sequence_lengths, self.qk_head_dim**-0.5, self.dsa_loss_coeff,
        )
        return aux_loss_auto_scale(output, loss)

    def preflight_attention(
        self, hidden_states: torch.Tensor, actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: Mapping[str, Any],
    ) -> SparseAttentionSequenceMetadata:
        """Resolve local sequence boundaries before context-parallel communication."""
        return SparseAttentionSequenceMetadata.build(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, actual_seq_len=actual_seq_len, kwargs=kwargs,
        )

    def prepare_attention_inputs(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        metadata: SparseAttentionSequenceMetadata, mome_mask: torch.Tensor | None,
    ) -> SparseAttentionInputs:
        """Prepare only activations, keeping no parameter views across CP exchange."""
        q_resid, attention_states = self._project_attention_states(hidden_states, mome_mask)
        query, key, query_rope, key_rope = attention_states
        query_rope, key_rope = _apply_attention_rope(
            query_rope, key_rope, position_embeddings, interleaved=self.rotary_interleaved,
        )
        index_states = self._project_index_states(hidden_states, q_resid, position_embeddings)
        return SparseAttentionInputs(index_states, (query, key, query_rope, key_rope), metadata, False)

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Keep local inputs unchanged; CP family members exchange key banks."""
        return inputs

    def run_attention(
        self, index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
        sequence_lengths: tuple[torch.Tensor, torch.Tensor], *, use_dense: bool = False,
    ) -> torch.Tensor:
        """Run sparse selection/attention and restore this rank's value-head outputs."""
        if use_dense:
            raise NotImplementedError("DeepseekSparseAttention does not support dense warm-up")
        indices, *flattened_index_states = dsa_indexer(  # pylint: disable=not-callable
            *index_states, *sequence_lengths, self.index_topk,
        )
        output, maximum, total = self._run_sparse_attention(attention_states, indices, *sequence_lengths)
        output = self._apply_auxiliary_loss(
            output, tuple(flattened_index_states), attention_states, indices, maximum, total, sequence_lengths,
        )
        batch_size, seq_length, heads = attention_states[0].shape[:3]
        weight = self.linear_kvb.weight.view(heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        return _restore_attention_projection(
            output, weight[:, self.qk_nope_head_dim:].transpose(1, 2), num_heads=heads,
            batch_size=batch_size, seq_length=seq_length,
            kv_lora_rank=self.kv_lora_rank, value_head_dim=self.v_head_dim,
        )

    def project_output(
        self, output: torch.Tensor, mome_mask: torch.Tensor | None = None, return_bias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply local MOME and output projection while preserving the source tuple ABI."""
        if self.use_mome:
            output = apply_mome(output, mome_mask, self.o_conv, fused=self.use_fused_mome)
        projected = self.linear_proj(output)
        output, bias = projected if isinstance(projected, tuple) else (projected, None)
        return (output, bias) if return_bias else (output, None)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None, past_key_values: Any | None = None,
        cache_position: torch.Tensor | None = None, actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        kv_reuse_states: Any | None = None, output_attentions: bool = False, return_bias: bool = False,
        mome_mask: torch.Tensor | None = None, **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the shared attention stages and preserve local MOME/sink behavior."""
        self._validate_forward_inputs(
            attention_mask, kv_reuse_states, past_key_values, cache_position, output_attentions,
        )
        metadata = self.preflight_attention(hidden_states, actual_seq_len, kwargs)
        inputs = self.prepare_attention_inputs(hidden_states, position_embeddings, metadata, mome_mask)
        inputs = self.exchange_attention(inputs)
        output = self.run_attention(
            inputs.index_states, inputs.attention_states, inputs.metadata.sequence_lengths, use_dense=inputs.use_dense,
        )
        del inputs
        return self.project_output(output, mome_mask, return_bias)
