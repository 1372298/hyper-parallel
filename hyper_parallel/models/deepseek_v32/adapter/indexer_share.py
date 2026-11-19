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
"""Pass invocation-local sharing state through the native HF decoder kwargs."""

from typing import Any

from torch import nn

from hyper_parallel.models.adapter_spec import RecomputePolicy

from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import (
    IndexShareContext, resolve_index_share_plan,
)


def _start_index_share_forward(
    module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Allocate a fresh carrier before entering the decoder/checkpoint loop."""
    del module
    if "index_share_context" in kwargs:
        raise ValueError("The model owns index_share_context; pass it only when calling individual attention layers")
    return args, {**kwargs, "index_share_context": IndexShareContext()}


def install_indexer_share(model: nn.Module) -> None:
    """Install the per-forward carrier on a V3.2 decoder model, once.

    Args:
        model: HF causal LM or decoder model. Call before compilation/checkpoint
            wrapping. All decoder attentions must use DeepseekV32SparseAttention.
            The normal adapter materialization callback installs this automatically.
    """
    decoder = getattr(model, "model", model)
    plan = resolve_index_share_plan(decoder.config)
    if not plan.enabled:
        return
    layers = getattr(decoder, "layers", ())
    if len(layers) != len(plan.leaders):
        raise ValueError("Indexer sharing requires the complete decoder layer stack")
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", layer)
        if getattr(attention, "index_share_plan", None) != plan or getattr(attention, "layer_idx", None) != index:
            raise ValueError("Replace all decoder attentions with the same Indexer Share plan before installation")
    if getattr(decoder, "_hp_index_share_hook", None) is None:
        decoder._hp_index_share_hook = decoder.register_forward_pre_hook(  # pylint: disable=protected-access
            _start_index_share_forward, with_kwargs=True,
        )


def build_recompute_policy() -> RecomputePolicy:
    """Keep decoder regions non-reentrant and install the invocation carrier.

    Full recomputation uses the model-owned region path so shared differentiable
    Indexer projections remain connected to the leader's autograd graph.
    """
    return RecomputePolicy(
        region_patterns=("model.layers.*", "layers.*"), on_applied=install_indexer_share,
    )
