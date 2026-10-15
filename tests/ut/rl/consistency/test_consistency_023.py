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
"""Pinned 0.23 recipe selection and default-sum dtype/autograd compatibility."""
# pylint: disable=protected-access

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.consistency.qwen3_dense as profile_module


@pytest.mark.parametrize("ascend", ["0.23.0.post1", "0.23.0.post2"])
def test_recipe_selection_requires_exact_runtime_pair(monkeypatch, ascend) -> None:
    """An unverified patch release cannot reuse the validated 0.23 recipe."""
    versions = {**profile_module._EXPECTED_PACKAGE_VERSIONS_023, "vllm-ascend": ascend}
    monkeypatch.setattr(profile_module, "package_version", lambda name: versions[name] + "+local")
    recipe = profile_module.consistency_profile({"consistency": {"enabled": True}})
    if ascend == "0.23.0.post1":
        assert recipe == profile_module.QWEN3_ASCEND_CONSISTENCY_V023
        profile_module._require_package_versions(recipe)
        versions["flash-attn-npu"] = "0.2.0b2"
        with pytest.raises(ValueError, match="flash-attn-npu==0.2.0b1"):
            profile_module._require_package_versions(recipe)
    else:
        assert recipe == profile_module.QWEN3_ASCEND_CONSISTENCY_V1
        with pytest.raises(ValueError, match="vllm==0.22.1"):
            profile_module._require_package_versions(recipe)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                    torch.float64, torch.int64, torch.bool])
def test_default_sum_registration_preserves_dtype_and_gradients(monkeypatch, dtype) -> None:
    """Default sum supports token masks and differentiable losses without an axis argument."""
    calls = []

    def native_sum(tensor: torch.Tensor, dim: int | list[int], keepdim: bool) -> torch.Tensor:
        """Record the native reduction signature and preserve autograd."""
        calls.append((dim, keepdim))
        return tensor.sum(dim=dim, keepdim=keepdim)

    namespace = SimpleNamespace(npu_reduce_sum_batch_invariant=native_sum)
    monkeypatch.setattr(profile_module.torch.ops, "batch_invariant_ops", namespace)
    registered = {}

    def enable() -> None:
        """Capture the shim installed during native registration."""
        registered["sum"] = namespace.npu_reduce_sum_batch_invariant

    profile_module._enable_batch_invariant_023(enable)
    assert namespace.npu_reduce_sum_batch_invariant is native_sum
    tensor = torch.arange(6).reshape(2, 3).to(dtype=dtype)
    if tensor.is_floating_point():
        tensor.requires_grad_(True)
    actual = registered["sum"](tensor)
    torch.testing.assert_close(actual, tensor.sum())
    assert actual.dtype == tensor.sum().dtype
    if tensor.is_floating_point():
        actual.backward()
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
    assert bool(calls) == (dtype in (torch.float16, torch.bfloat16, torch.float32))
    torch.testing.assert_close(registered["sum"](tensor, dtype=torch.float64), tensor.sum(dtype=torch.float64))


def test_default_sum_registration_restores_namespace_after_failure(monkeypatch) -> None:
    """A partial installation must not leave a Python shim replacing the native op packet."""
    native_sum = object()
    namespace = SimpleNamespace(npu_reduce_sum_batch_invariant=native_sum)
    monkeypatch.setattr(profile_module.torch.ops, "batch_invariant_ops", namespace)

    def fail() -> None:
        """Simulate an interrupted native registration."""
        raise RuntimeError("registration failed")

    with pytest.raises(RuntimeError, match="registration failed"):
        profile_module._enable_batch_invariant_023(fail)
    assert namespace.npu_reduce_sum_batch_invariant is native_sum


def test_packed_bf16_logits_match_sampler_fp32_scores_and_gradients(monkeypatch) -> None:
    """Packing preserves the sampler's FP32 probability contract and excludes padding gradients."""
    monkeypatch.setattr(profile_module._runtime, "installed_profile", profile_module.QWEN3_ASCEND_CONSISTENCY_V023)
    logits = torch.nn.Parameter(torch.arange(56).reshape(1, 7, 8).to(torch.bfloat16) / 8)
    def model(**_kwargs: Any) -> SimpleNamespace:
        """Return packed logits without changing their gradient graph."""
        return SimpleNamespace(logits=logits)
    sequences = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 0]])
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    scores = profile_module.trainer_sequence_log_probs(model, sequences, mask)
    # This is the pinned sampler's raw probability operation, applied independently.
    reference = logits[0].log_softmax(dim=-1, dtype=torch.float32)
    first = reference[:3].gather(-1, sequences[0, 1:, None]).flatten()
    second = reference[4:6].gather(-1, sequences[1, 1:3, None]).flatten()
    expected = torch.stack((first, torch.cat((second, torch.zeros(1)))))
    assert scores.dtype is torch.float32
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(scores.sum(), logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
    assert torch.count_nonzero(actual_grad[0, [3, 6]]) == 0
