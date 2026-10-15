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
"""CPU contracts for the DeepSeek V4.1 MegaGate replacement."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_gate import function as gate_function
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_gate import (
    DeepseekV41MegaGate,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _initialize_v41_owned_module,
)
from tests.common.mark_utils import arg_mark


def _config(vision: bool, scoring: str = "sqrtsoftplus") -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=128, num_local_experts=16, num_experts_per_tok=6,
        scoring_func=scoring, routed_scaling_factor=1.5, v41_vision_enabled=vision,
        initializer_range=0.02,
    )


def _replace(router: DeepseekV41TopKRouter) -> DeepseekV41MegaGate:
    return DeepseekV41MegaGate(
        module=router, module_fqn="model.layers.0.mlp.gate", context={},
    )


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
class TestMegaGateReplacement(unittest.TestCase):
    """Preserve model state and construction semantics across replacement."""

    def test_replacement_preserves_router_state(self):
        """Reuse source parameters without loading native code or changing RNG state."""
        for vision in (False, True):
            with self.subTest(vision=vision):
                router = DeepseekV41TopKRouter(_config(vision)).eval()
                router.weight.requires_grad_(False)
                rng = torch.get_rng_state().clone()
                with patch.object(gate_function, "_native_ops", side_effect=AssertionError("native load")):
                    gate = _replace(router)
                self.assertIs(gate.weight, router.weight)
                self.assertIs(gate.bias, router.bias)
                self.assertIs(gate.bias_vl, router.bias_vl)
                self.assertFalse(gate.training)
                self.assertFalse(gate.weight.requires_grad)
                self.assertEqual(set(gate.state_dict()), set(router.state_dict()))
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_replacement_supports_meta_initialization(self):
        """Materialize and initialize all parameters owned by a visual router."""
        with torch.device("meta"):
            gate = _replace(DeepseekV41TopKRouter(_config(vision=True)))
        self.assertTrue(all(parameter.is_meta for parameter in gate.parameters()))

        gate.to_empty(device="cpu")
        with torch.no_grad():
            _initialize_v41_owned_module(gate, 0.035)

        self.assertTrue(torch.isfinite(gate.weight).all())
        self.assertGreater(gate.weight.std().item(), 0.02)
        self.assertEqual(torch.count_nonzero(gate.bias).item(), 0)
        self.assertEqual(torch.count_nonzero(gate.bias_vl).item(), 0)

    def test_cpu_fallback_matches_router(self):
        """Match the real router for every fallback score function and mask contract."""
        mask = torch.arange(14).reshape(2, 7).remainder(3) == 0
        invalid_mask = torch.zeros(14, dtype=torch.bool)
        for scoring in ("sqrtsoftplus", "softmax", "sigmoid"):
            with self.subTest(scoring=scoring):
                golden = DeepseekV41TopKRouter(_config(vision=True, scoring=scoring))
                with torch.no_grad():
                    _initialize_v41_owned_module(golden, 0.02)
                    golden.bias.uniform_(-0.5, 0.5)
                    golden.bias_vl.uniform_(-0.5, 0.5)
                gate = _replace(deepcopy(golden))
                hidden = torch.randn(2, 7, 128, requires_grad=True)
                reference = hidden.detach().clone().requires_grad_(True)
                expected = golden(reference, image_mask=mask)
                actual = gate(hidden, image_mask=mask)
                for result, baseline in zip(actual, expected):
                    torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
                expected[1].square().sum().backward()
                actual[1].square().sum().backward()
                torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(gate.weight.grad, golden.weight.grad, rtol=1e-5, atol=1e-6)

        for router in (golden, gate):
            with self.assertRaisesRegex(ValueError, "image_mask must have shape \\[batch, sequence\\]"):
                router(torch.randn(2, 7, 128), image_mask=invalid_mask)

        for router in (golden, gate):
            router.scoring_func = "unsupported"
            with self.assertRaisesRegex(ValueError, "Unsupported V4.1 router scoring function"):
                router(torch.randn(2, 7, 128))
