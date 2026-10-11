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
"""Bounded terminal projection for complete Qwen3.5 text next-token probabilities."""

from functools import wraps
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.utils.checkpoint import checkpoint  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.fully_shard.hsdp_utils import get_hsdp_state
from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=protected-access
    _ForwardRewriteRequest,
    _commit_forward_rewrite,
)


def _probability_inputs(args: tuple, kwargs: dict, position_mask: Any) -> torch.Tensor:
    """Validate complete text inputs before invoking the decoder."""
    if args:
        raise TypeError("Qwen3.5 token probability inputs must be passed by keyword")
    tokens = kwargs.get("input_ids")
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
            or tokens.dtype != torch.long or tokens.shape[0] == 0 or tokens.shape[1] < 2):
        raise ValueError("Qwen3.5 token probabilities require nonempty int64 [batch, length>=2] input_ids")
    if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
        raise ValueError("Qwen3.5 token probabilities require complete cache-free sequences")
    if kwargs.get("labels") is not None or kwargs.get("inputs_embeds") is not None:
        raise ValueError("Qwen3.5 token probabilities require input_ids without labels or inputs_embeds")
    if position_mask is not None and (
            not isinstance(position_mask, torch.Tensor) or position_mask.dtype != torch.bool
            or position_mask.device != tokens.device
            or tuple(position_mask.shape) != (tokens.shape[0], tokens.shape[1] - 1)):
        raise ValueError("token_log_probs_mask must be boolean [batch, length - 1] on the input device")
    return tokens


def bind_token_log_probs(model: nn.Module, *, chunk_size: int = 512) -> None:
    """Bind an explicit probability-output mode without changing ordinary HF forward.

    ``model(..., return_token_log_probs=True)`` returns FP32 ``[batch, length - 1]``
    probabilities for every next token, including prompt and padding positions.
    Optional boolean ``token_log_probs_mask`` selects positions to project; unselected
    entries are zero and must remain excluded by the caller's loss mask. The decoder
    still consumes the complete context. Cache-free text inputs and full-vocabulary
    projection are required; vocabulary-sharded training is outside this adapter's scope.

    Args:
        model: Finalized Qwen3.5 text causal LM, including its existing distributed hooks.
        chunk_size: Maximum total token rows per head call, bounded by 512.

    Raises:
        ValueError: The chunk bound or model identity is unsupported.
        TypeError: The model does not expose the expected decoder and output head.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 0 < chunk_size <= 512:
        raise ValueError("Qwen3.5 token probability chunk_size must be an integer between 1 and 512")
    if getattr(model, "supports_token_log_probs", False):
        return
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_5_text":
        raise ValueError("Qwen3.5 token probabilities require config.model_type='qwen3_5_text'")
    if not isinstance(getattr(model, "model", None), nn.Module):
        raise TypeError("Qwen3.5 token probabilities require a decoder at model.model")
    if not isinstance(getattr(model, "lm_head", None), nn.Linear):
        raise TypeError("Qwen3.5 token probabilities require a Linear lm_head")
    # Unequal sequence lengths imply unequal chunk counts across DP ranks.
    # Only the outer root may introduce HSDP collectives around these head calls.
    if get_hsdp_state(model.lm_head) is not None:
        raise ValueError("Qwen3.5 token probabilities do not support an independently sharded lm_head")
    if get_hsdp_state(model) is not None and model.hsdp_scheduler.reshard_after_forward:
        raise ValueError("Qwen3.5 token probabilities require root reshard_after_forward=False")

    original_forward = model.forward

    def project(hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Use the currently installed head parameters on forward and recomputation."""
        logits = model.lm_head(hidden)
        return logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    @wraps(original_forward)
    def token_log_probs_forward(*args: Any, **kwargs: Any) -> Any:
        """Keep normal HF calls unchanged and bound only the requested terminal output."""
        requested = kwargs.pop("return_token_log_probs", False)
        position_mask = kwargs.pop("token_log_probs_mask", None)
        if not isinstance(requested, bool):
            raise ValueError("return_token_log_probs must be a boolean")
        if not requested:
            if position_mask is not None:
                raise ValueError("token_log_probs_mask requires return_token_log_probs=True")
            return original_forward(*args, **kwargs)
        tokens = _probability_inputs(args, kwargs, position_mask)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int) or logits_to_keep != 0:
            raise ValueError("Qwen3.5 token probabilities cannot select a subset of sequence positions")
        kwargs.pop("labels", None)
        kwargs.pop("return_dict", None)
        kwargs["use_cache"] = False
        hidden = model.model(**kwargs, return_dict=True).last_hidden_state[:, :-1, :]
        targets = tokens[:, 1:]
        if hidden.shape[:2] != targets.shape:
            raise ValueError("Qwen3.5 decoder hidden positions do not align with next-token targets")
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        target_flat = targets.reshape(-1)
        indices = None
        if position_mask is not None:
            indices = position_mask.reshape(-1).nonzero(as_tuple=True)[0]
            # A zero-action DP row must still participate in decoder/head backward.
            # One dummy projection keeps every parameter's zero gradient connected.
            if indices.numel() == 0:
                indices = torch.zeros(1, dtype=torch.long, device=tokens.device)
            hidden_flat = hidden_flat.index_select(0, indices)
            target_flat = target_flat.index_select(0, indices)
        chunks = []
        for start in range(0, target_flat.numel(), chunk_size):
            hidden_chunk = hidden_flat[start:start + chunk_size]
            target_chunk = target_flat[start:start + chunk_size]
            # Plain chunks retain every vocabulary-sized softmax for backward.
            # Resolve the live head module during recomputation, never an old weight view.
            if torch.is_grad_enabled():
                chosen = checkpoint(project, hidden_chunk, target_chunk, use_reentrant=False, preserve_rng_state=True)
            else:
                chosen = project(hidden_chunk, target_chunk)
            chunks.append(chosen)
        values = torch.cat(chunks)
        if indices is None:
            return values.reshape(targets.shape)
        values = values * position_mask.reshape(-1).index_select(0, indices)
        return values.new_zeros(targets.numel()).scatter(0, indices, values).reshape(targets.shape)

    _commit_forward_rewrite(_ForwardRewriteRequest(
        model, token_log_probs_forward,
        companion_attrs={"supports_token_log_probs": True, "supports_token_log_probs_mask": True},
    ))
