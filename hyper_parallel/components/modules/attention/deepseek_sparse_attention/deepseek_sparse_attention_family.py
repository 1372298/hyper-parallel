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
"""DeepSeek sparse-attention families, parameter layouts and parallel initialization."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import resolve_index_share_plan
from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def validate_sparse_attention_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Check logical topology before allocating native or replacement parameters."""
    config = config if module is None else module.config
    if context.get("sequence_parallel", False):
        raise ValueError("DeepSeek sparse attention does not support sequence parallelism")
    if member.axis_sizes["pp"] > 1:
        raise ValueError("DeepSeek sparse attention does not support pipeline parallelism")
    heads = getattr(module, "num_heads", getattr(config, "num_attention_heads", None))
    if heads % member.axis_sizes["tp"]:
        raise ValueError("Sparse-attention heads must be divisible by TP degree")


def validate_generic_sparse_attention_family(
    *, config: Any, module: nn.Module | None, context: Any, member: Any,
) -> None:
    """Keep MOME and parameter sinks local until their own distributed algorithms exist."""
    validate_sparse_attention_family(config=config, module=module, context=context, member=member)
    config = config if module is None else module.config
    if resolve_index_share_plan(config).enabled:
        raise ValueError("Indexer sharing is supported by DeepseekV32SparseAttention, not the generic flat family")
    use_mome = getattr(module, "use_mome", getattr(config, "use_mome", False))
    sink_count = getattr(module, "param_sink_number", getattr(config, "param_sink_number", 0))
    if (member.axis_sizes["tp"] > 1 or member.axis_sizes["cp"] > 1) and (
        use_mome or sink_count > 0
    ):
        raise ValueError("MOME and parameter-sink sparse attention require TP1 and CP1")
    projection = getattr(module, "linear_proj", None)
    if member.axis_sizes["tp"] > 1 and getattr(projection, "bias", None) is not None and (
        not isinstance(projection, nn.Linear) or getattr(projection, "skip_bias_add", False)
    ):
        raise ValueError("TP sparse attention requires standard nn.Linear output-bias semantics")


def sparse_attention_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare every owned parameter and external token port without name fallbacks."""
    del module_fqn, member
    if context.sequence_parallel:
        raise ValueError("DeepSeek sparse attention does not support sequence parallelism")
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    generic = hasattr(module, "linear_kvb")
    column_names = ("linear_qb", "linear_kvb") if generic else ("q_b_proj", "kv_b_proj")
    replica_names = (
        "linear_qkv", "q_layernorm", "k_layernorm", "index_linear_qb", "index_linear_k",
        "index_k_layernorm", "linear_merge_weight", "qa_conv", "compresskv_conv", "o_conv",
    ) if generic else (
        "linear_qkv", "q_a_layernorm", "kv_a_layernorm",
        "indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.weights_proj",
    )
    output_name = "linear_proj" if generic else "o_proj"
    columns = {f"{name}.{suffix}" for name in column_names for suffix in ("weight", "bias")}
    replicas = {
        f"{name}.{suffix}"
        for name in replica_names
        for suffix in ("weight", "bias")
    }
    roles = {
        **dict.fromkeys(columns, Shard(0)), **dict.fromkeys(replicas, Replicate()),
        f"{output_name}.weight": Shard(1), f"{output_name}.bias": Replicate(),
        "param_sink_k_pe": Replicate(), "param_sink_compressed_kv": Replicate(),
    }
    params = {}
    for name, _ in module.named_parameters(remove_duplicate=False):
        if name not in roles:
            raise ValueError(f"Sparse-attention parameter {name!r} has no declared sharding role")
        params[name] = dict(placements)
        if context.axis_sizes["tp"] > 1:
            params[name]["tp"] = roles[name]
    tokens = dict(placements)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    output = dict(tokens)
    if context.axis_sizes["tp"] > 1:
        output["tp"] = Partial()
    return ModuleShardingSpec(
        params=params, in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": dict(tokens)},
        out_src={"output": output}, out_dst={"output": dict(tokens)},
        out_names=("output", "bias" if generic else "attention_weights"), is_boundary=True,
    )


def validate_sparse_attention_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Validate the final overridden contract, including the unsupported SP boundary."""
    expected = sparse_attention_sharding(module=module, module_fqn="", member=None, context=context)
    validate_tp_boundary(sharding, expected, name="DeepSeek sparse attention")
    for field, port in (("in_src", "hidden_states"), ("out_dst", "output")):
        if getattr(sharding, field)[port].get("tp", Replicate()) != Replicate():
            raise ValueError("DeepSeek sparse attention does not support SP token-port overrides")
    if tuple(sharding.out_names or ()) != tuple(expected.out_names):
        raise ValueError("Sparse-attention output names must preserve the declared return contract")


def v32_sparse_attention_parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Declare shared algorithm stages and validate layouts outside local computation."""
    del module, member
    return ParallelModuleSpec(
        "attention.deepseek_v32_sparse_attention.local",
        stages={
            "preflight_attention": (
                "hidden_states", "attention_mask", "past_key_values", "position_ids",
                "actual_seq_len", "kwargs", "use_dense",
            ),
            "prepare_attention_inputs": ("hidden_states", "position_embeddings", "metadata", "use_dense"),
            "exchange_attention": ("inputs",),
            "run_attention": ("index_states", "attention_states", "sequence_lengths", "use_dense", "index_share_context"),
            "project_output": ("output",),
        },
        validate=validate_sparse_attention_layout,
    )


def sparse_attention_parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Declare generic sparse-attention stages and the replicated Indexer contract."""
    del module, member
    return ParallelModuleSpec(
        "attention.deepseek_sparse_attention.local",
        stages={
            "preflight_attention": ("hidden_states", "actual_seq_len", "kwargs"),
            "prepare_attention_inputs": ("hidden_states", "position_embeddings", "metadata", "mome_mask"),
            "exchange_attention": ("inputs",),
            "run_attention": ("index_states", "attention_states", "sequence_lengths", "use_dense"),
            "project_output": ("output", "mome_mask", "return_bias"),
        },
        validate=validate_sparse_attention_layout,
    )


@module_replacement
def DeepseekV32SparseAttention(  # pylint: disable=invalid-name
    *, module: nn.Module | None = None, config: Any = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, cp_strategy: str = "kv_allgather",
    overlap: bool = True, dense_warmup: bool | None = None,
) -> nn.Module:
    """Build a local or CP member while retaining V3.2 source and checkpoint contracts.

    Args:
        module: HF attention source or an already loaded unbound local member.
        config: Native attention dimensions and options; mutually exclusive with module.
        module_fqn: Source path supplied by the replacement executor.
        context: Logical job topology used for family selection.
        mesh: Optional existing MeshContext for shared layout planning and binding.
        cp_strategy: Supported CP strategy, kv_allgather.
        overlap: Submit all key-bank gathers before waiting when true.
        dense_warmup: None follows the mutable model stage; bool pins the stage.

    Returns:
        Concrete member retaining the requested family and parameter layout.
    """
    return build_family(
        DeepseekV32SparseAttention, module=module, config=config, module_fqn=module_fqn, context=context,
        mesh=mesh, cp_strategy=cp_strategy, overlap=overlap, dense_warmup=dense_warmup,
    )


register_module_family(ModuleFamily(
    name="deepseek_v32_sparse_attention", baseline=DeepseekV32SparseAttention,
    variants={
        "local": (
            "hyper_parallel.components.modules.attention.deepseek_sparse_attention."
            "deepseek_sparse_attention.DeepseekV32SparseAttentionModule"
        ),
        "cp": (
            "hyper_parallel.components.modules.attention.deepseek_sparse_attention."
            "cp_deepseek_sparse_attention.CPDeepseekV32SparseAttentionModule"
        ),
    },
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=sparse_attention_sharding, validate=validate_sparse_attention_family,
    parallel_spec=v32_sparse_attention_parallel_spec,
))


@module_replacement
def DeepseekSparseAttention(  # pylint: disable=invalid-name
    *, module: nn.Module | None = None, config: Any = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, cp_strategy: str = "kv_allgather",
    overlap: bool = True, dense_warmup: bool | None = None,
) -> nn.Module:
    """Build generic sparse attention from native configuration or its original parameter tree.

    Args:
        module: Source attention using flat projection and Indexer names.
        config: Native dimensions/options, mutually exclusive with module.
        module_fqn: Source path supplied by the replacement executor.
        context: Logical topology for local/CP selection.
        mesh: Existing MeshContext for parameter layout and communication binding.
        cp_strategy: Supported CP strategy, kv_allgather.
        overlap: Submit key-bank gathers together before waiting.
        dense_warmup: None or False; dense warm-up belongs to the V3.2 implementation.

    Returns:
        The local or CP member, retaining generic projection/checkpoint names.
    """
    return build_family(
        DeepseekSparseAttention, module=module, config=config, module_fqn=module_fqn, context=context,
        mesh=mesh, cp_strategy=cp_strategy, overlap=overlap, dense_warmup=dense_warmup,
    )


register_module_family(ModuleFamily(
    name="deepseek_sparse_attention", baseline=DeepseekSparseAttention,
    variants={
        "local": (
            "hyper_parallel.components.modules.attention.deepseek_sparse_attention."
            "deepseek_sparse_attention.DeepseekSparseAttentionModule"
        ),
        "cp": (
            "hyper_parallel.components.modules.attention.deepseek_sparse_attention."
            "cp_deepseek_sparse_attention.CPDeepseekSparseAttentionModule"
        ),
    },
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=sparse_attention_sharding, validate=validate_generic_sparse_attention_family,
    parallel_spec=sparse_attention_parallel_spec,
))
