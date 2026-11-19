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
"""Explicit local-query KV-AllGather scheduling for DeepSeek sparse attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    DeepseekSparseAttentionModule, DeepseekV32SparseAttentionModule,
    SparseAttentionInputs, SparseAttentionSequenceMetadata,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention_family import (
    sparse_attention_parallel_spec, v32_sparse_attention_parallel_spec, validate_sparse_attention_layout,
)
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


def _project_ends(
    boundaries: Sequence[int], *, global_seq_len: int, local_seq_len: int, rank: int, query: bool,
) -> tuple[int, ...]:
    """Intersect every document with the Q shard or causal KV prefix in each row."""
    previous = total = 0
    result = []
    for end in boundaries:
        length = 0
        for row in range(previous // global_seq_len, (end - 1) // global_seq_len + 1):
            row_start = row * global_seq_len
            begin = row_start + rank * local_seq_len if query else row_start
            stop = row_start + (rank + 1) * local_seq_len
            length += max(0, min(end, stop) - max(previous, begin))
        total += length
        result.append(total)
        previous = end
    return tuple(result)


def build_cp_sequence_metadata(
    *, batch_size: int, local_seq_len: int, device: torch.device, cp_size: int, cp_rank: int,
    actual_seq_len: torch.Tensor | Sequence[int] | None = None, kwargs: Mapping[str, Any] | None = None,
) -> SparseAttentionSequenceMetadata:
    """Project packed documents after checking every rank before communication.

    Args:
        batch_size: Number of batch-major rows.
        local_seq_len: Equal contiguous token count per row and CP rank.
        device: Kernel metadata device.
        cp_size: Context-parallel degree.
        cp_rank: Rank within the borrowed CP group.
        actual_seq_len: Global cumulative packed endpoints without leading zero.
        kwargs: Supported packed aliases, including packed_seq_params.

    Returns:
        This rank's cumulative Q/K endpoints and causal KV prefix length.
    """
    global_length = local_seq_len * cp_size
    total = batch_size * global_length
    packed = dict(kwargs or {})
    if actual_seq_len is not None:
        packed["actual_seq_len"] = actual_seq_len
    queries, keys = resolve_packed_sequence_lengths(packed, total, total)
    if queries is None:
        queries = keys = [(row + 1) * global_length for row in range(batch_size)]
    if len(queries) != len(keys):
        raise ValueError("Sparse attention requires the same number of packed Q and KV documents")
    projected = []
    for rank in range(cp_size):
        query = _project_ends(
            queries, global_seq_len=global_length, local_seq_len=local_seq_len, rank=rank, query=True,
        )
        key = _project_ends(
            keys, global_seq_len=global_length, local_seq_len=local_seq_len, rank=rank, query=False,
        )
        if any(end == previous for previous, end in zip((0, *query[:-1]), query)):
            raise ValueError(f"Sparse attention does not support zero-query packed documents on CP rank {rank}")
        if any(end == previous for previous, end in zip((0, *key[:-1]), key)):
            raise ValueError(f"Sparse attention has an empty packed KV document on CP rank {rank}")
        projected.append((query, key))
    local_query, local_key = projected[cp_rank]
    return SparseAttentionSequenceMetadata(
        (cp_rank + 1) * local_seq_len,
        torch.tensor(local_query, dtype=torch.int32, device=device),
        torch.tensor(local_key, dtype=torch.int32, device=device),
    )


def _gather_key_banks(
    context: CPContext, overlap: bool,
    index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
    metadata: SparseAttentionSequenceMetadata,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Gather the three key banks in a fixed order, then select the causal prefix."""
    index_query, index_key, weights = index_states
    query, latent_kv, query_rope, key_rope = attention_states
    tensors = (index_key, latent_kv, key_rope)
    if overlap:
        pending = tuple(context.launch_gather_sequence(tensor, dim=1) for tensor in tensors)
        gathered = tuple(receipt.wait()[0] for receipt in pending)
    else:
        gathered = tuple(context.gather_sequence(tensor, dim=1)[0] for tensor in tensors)
    index_key, latent_kv, key_rope = (
        tensor[:, :metadata.kv_prefix_length].contiguous() for tensor in gathered
    )
    return (index_query, index_key, weights), (query, latent_kv, query_rope, key_rope)


class CPDeepseekV32SparseAttentionModule(DeepseekV32SparseAttentionModule):
    """Borrow CP transport while reusing the local projection and kernel stages."""

    def __init__(
        self, *, module: nn.Module | None = None, config: Any = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        cp_strategy: str = "kv_allgather", overlap: bool = True, dense_warmup: bool | None = None,
    ) -> None:
        """Retain the source parameters; defer process-group ownership to binding."""
        super().__init__(
            module=module, config=config, module_fqn=module_fqn, context=context,
            cp_strategy=cp_strategy, overlap=overlap, dense_warmup=dense_warmup,
        )
        self._cp_context: CPContext | None = None

    @staticmethod
    def parallel_module_spec() -> ParallelModuleSpec:
        """Declare operand exchange while the local algorithm owns mathematical dispatch."""
        local_spec = v32_sparse_attention_parallel_spec(module=None, member=None)
        return ParallelModuleSpec(
            "attention.deepseek_v32_sparse_attention.cp", binding_kind="cp",
            stages=local_spec.stages,
            validate=validate_sparse_attention_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow an existing CPContext before FSDP installs projection hooks."""
        context = binding.cp_context
        if context is None or context.size <= 1:
            raise ValueError("DeepSeek-V3.2 CP member requires an active CPContext")
        if self._cp_context is not None and self._cp_context.mesh is not context.mesh:
            raise ValueError("DeepSeek-V3.2 CP member already belongs to another mesh")
        self._cp_context = context

    def prepare_key_bank(
        self, index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
        metadata: SparseAttentionSequenceMetadata,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Gather three banks in fixed order and consume their causal prefixes.

        All readiness receipts belong to this invocation. The existing gather
        adjoint sums consumers into each source shard, including the zero
        gradients from unused suffix tokens.
        """
        return _gather_key_banks(self._cp_context, self.overlap, index_states, attention_states, metadata)

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Return ready key banks without taking ownership of the attention algorithm."""
        index_states, attention_states = self.prepare_key_bank(
            inputs.index_states, inputs.attention_states, inputs.metadata,
        )
        return SparseAttentionInputs(index_states, attention_states, inputs.metadata, inputs.use_dense)

    def _validate_attention_mask(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        """Accept existing causal-mask carriers without reading their device contents."""
        if attention_mask is None:
            return
        batch_size, local_length = hidden_states.shape[:2]
        global_length = local_length * self._cp_context.size
        prefix_length = local_length * (self._cp_context.rank + 1)
        shapes = {
            (local_length, local_length), (local_length, global_length),
            (global_length, global_length), (local_length, prefix_length),
        }
        # TextParallelBatch constructs a global square mask after slicing token fields.
        # Packed endpoints, rather than mask values, remain the CANN causality contract.
        if (attention_mask.ndim != 4 or attention_mask.shape[0] not in (1, batch_size)
                or attention_mask.shape[1] != 1 or tuple(attention_mask.shape[-2:]) not in shapes):
            raise ValueError(
                "DeepSeek-V3.2 CP requires a standard 4D causal mask with local/global sequence dimensions; "
                "padding and custom masks are not supported"
            )

    def preflight_attention(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
        past_key_values: Any | None, position_ids: torch.Tensor | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: dict[str, Any], *, use_dense: bool,
    ) -> SparseAttentionSequenceMetadata:
        """Validate every rank's packed geometry before any operand collective."""
        del use_dense
        if self._cp_context is None:
            raise ValueError("DeepSeek-V3.2 CP member must be bound before execution")
        self._validate_forward_inputs(hidden_states, None, past_key_values, position_ids, kwargs)
        self._validate_attention_mask(hidden_states, attention_mask)
        return build_cp_sequence_metadata(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, cp_size=self._cp_context.size, cp_rank=self._cp_context.rank,
            actual_seq_len=actual_seq_len, kwargs=kwargs,
        )


class CPDeepseekSparseAttentionModule(DeepseekSparseAttentionModule):
    """Exchange generic sparse-attention key banks through an existing CP context."""

    def __init__(
        self, *, module: nn.Module | None = None, config: Any = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        cp_strategy: str = "kv_allgather", overlap: bool = True, dense_warmup: bool | None = None,
    ) -> None:
        """Retain the generic parameter tree and borrow transport at binding time."""
        super().__init__(
            module=module, config=config, module_fqn=module_fqn, context=context,
            cp_strategy=cp_strategy, overlap=overlap, dense_warmup=dense_warmup,
        )
        if self.use_mome or self.param_sink_number > 0:
            raise ValueError("MOME and parameter-sink sparse attention require CP1")
        self._cp_context: CPContext | None = None

    @staticmethod
    def parallel_module_spec() -> ParallelModuleSpec:
        """Keep the generic mathematical stages while declaring CP resource binding."""
        local_spec = sparse_attention_parallel_spec(module=None, member=None)
        return ParallelModuleSpec(
            "attention.deepseek_sparse_attention.cp", binding_kind="cp", stages=local_spec.stages,
            validate=validate_sparse_attention_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the already-created CP context before parameter materialization."""
        context = binding.cp_context
        if context is None or context.size <= 1:
            raise ValueError("Sparse-attention CP member requires an active CPContext")
        self._cp_context = context

    def preflight_attention(
        self, hidden_states: torch.Tensor, actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: Mapping[str, Any],
    ) -> SparseAttentionSequenceMetadata:
        """Resolve global packed boundaries before the first key-bank collective."""
        if self._cp_context is None:
            raise ValueError("Sparse-attention CP member must be bound before execution")
        return build_cp_sequence_metadata(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, cp_size=self._cp_context.size, cp_rank=self._cp_context.rank,
            actual_seq_len=actual_seq_len, kwargs=kwargs,
        )

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Gather ready key banks while preserving the local query and packed metadata."""
        index_states, attention_states = _gather_key_banks(
            self._cp_context, self.overlap, inputs.index_states, inputs.attention_states, inputs.metadata,
        )
        return SparseAttentionInputs(index_states, attention_states, inputs.metadata, inputs.use_dense)
