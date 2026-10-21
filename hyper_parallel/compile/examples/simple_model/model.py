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
"""Toy causal LM for the config-driven graph-mode ``TextTrainer`` example.

The model is a pure-PyTorch embedding + projection head following the Trainer
model contract:

- it is a ``torch.nn.Module`` with a ``config`` exposing ``to_dict()`` (the
  Trainer logs the model config);
- ``forward(**model_inputs, use_cache=False)`` returns an object exposing a
  ``loss`` tensor (the default ``ModelOutputLoss`` reads ``model_output.loss``).

``build_model`` is the YAML ``model._target_`` factory. FSDP sharding is left
to the graph-mode ``FSDPPass`` (``compile.enabled`` + ``use_joint_graph``), so
the factory only materializes the model on the current device.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from hyper_parallel.trainer.runtime.device import get_device_type

__all__ = [
    "SimpleCausalLMConfig",
    "SimpleCausalLMOutput",
    "SimpleCausalLM",
    "build_model",
]


class SimpleCausalLMConfig:
    """Minimal Transformers-style config exposing ``to_dict()``."""

    def __init__(self, vocab_size: int, dim: int, max_seq_len: int) -> None:
        """Record the toy model dimensions."""
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.model_type = "simple_graph_lm"
        self.architectures = ["SimpleCausalLM"]

    def to_dict(self) -> dict:
        """Return the JSON-serializable config mapping used by the Trainer."""
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
        }


class SimpleCausalLMOutput:
    """Model output carrying the (optional) causal-LM ``loss`` and ``logits``."""

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        """Store the loss and logits."""
        self.loss = loss
        self.logits = logits


class SimpleCausalLM(nn.Module):
    """Embedding + projection causal LM computing next-token CE loss."""

    def __init__(self, config: SimpleCausalLMConfig) -> None:
        """Build the embedding and projection head."""
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.linear = nn.Linear(config.dim, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> SimpleCausalLMOutput:
        """Project embeddings to logits and build the shifted CE loss.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            labels: Optional next-token targets of shape ``(batch, seq_len)``.
            use_cache: Accepted for the Trainer forward contract; unused.
            **kwargs: Ignored extra model inputs.

        Returns:
            Output exposing ``loss`` (``None`` without labels) and ``logits``.
        """
        del use_cache, kwargs
        logits = self.linear(self.embed(input_ids))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleCausalLMOutput(loss=loss, logits=logits)


def build_model(
    *,
    vocab_size: int,
    dim: int,
    max_seq_len: int,
    distributed_setup: Any = None,
    **kwargs: Any,
) -> SimpleCausalLM:
    """YAML ``model`` factory: build the toy model on the current device.

    Args:
        vocab_size: Vocabulary size.
        dim: Hidden dimension.
        max_seq_len: Maximum sequence length recorded in the config.
        distributed_setup: Trainer distributed setup (unused; FSDP sharding is
            done by the graph-mode ``FSDPPass``).
        **kwargs: Ignored extra Trainer build kwargs.

    Returns:
        The materialized ``SimpleCausalLM`` on the current device.
    """
    del distributed_setup, kwargs
    config = SimpleCausalLMConfig(
        vocab_size=vocab_size, dim=dim, max_seq_len=max_seq_len
    )
    model = SimpleCausalLM(config)
    return model.to(torch.device(get_device_type()))
