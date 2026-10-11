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
"""Regression coverage for the direct vLLM plugin entry point."""

from importlib.metadata import EntryPoint
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rl.roles.rollout.vllm_plugin as plugin_module


def test_external_plugin_entry_point_registers_qwen3(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public entry point must import and register the supported dense model."""
    entry_point = EntryPoint(
        name="hyper_parallel",
        value="rl.roles.rollout.vllm_plugin:register_hyper_models",
        group="vllm.general_plugins",
    )
    register_models = entry_point.load()
    assert register_models is plugin_module.register_hyper_models
    registry = SimpleNamespace(get_supported_archs=lambda: (), register_model=Mock())
    versions = {"vllm": "0.22.1", "vllm-ascend": "0.22.1rc1"}
    monkeypatch.setattr(plugin_module, "package_version", versions.__getitem__)
    monkeypatch.setattr(plugin_module, "install_vllm_weight_sync_hooks", Mock())
    evidence_hook = Mock()
    monkeypatch.setattr(plugin_module, "install_tool_evidence", evidence_hook)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    monkeypatch.delenv("HYPER_RL_CONSISTENCY_PROFILE", raising=False)
    monkeypatch.delenv("HYPER_RL_TEST_QWEN3_RMS_NORM", raising=False)

    register_models()
    evidence_hook.assert_called_once_with()

    registry.register_model.assert_called_once_with(
        "HyperQwen3ForCausalLM", "rl.roles.rollout.consistency_models.qwen3.model:HyperQwen3ForCausalLM",
    )


@pytest.mark.parametrize("pair,supported", [
    (("0.23.0+empty", "0.23.0.post1"), True),
    (("0.23.0", "0.22.1rc1"), False),
    (("0.24.0", "0.23.0.post1"), False),
])
def test_plugin_installs_all_private_hooks_only_for_exact_runtime_pairs(
    monkeypatch: pytest.MonkeyPatch, pair: tuple[str, str], supported: bool,
) -> None:
    """Known new versions enable the full adapter; mixed/unknown pairs never do."""
    versions = dict(zip(("vllm", "vllm-ascend"), pair))
    hooks = Mock()
    evidence = Mock()
    registry = SimpleNamespace(get_supported_archs=lambda: (), register_model=Mock())
    monkeypatch.setattr(plugin_module, "package_version", versions.__getitem__)
    monkeypatch.setattr(plugin_module, "install_vllm_weight_sync_hooks", hooks)
    monkeypatch.setattr(plugin_module, "install_tool_evidence", evidence)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    monkeypatch.delenv("HYPER_RL_CONSISTENCY_PROFILE", raising=False)
    monkeypatch.delenv("HYPER_RL_TEST_QWEN3_RMS_NORM", raising=False)
    plugin_module.register_hyper_models()
    assert hooks.call_args_list[0].kwargs == {"private_lifecycle": False}
    if supported:
        assert hooks.call_args_list[1].kwargs == {"private_lifecycle": True}
        evidence.assert_called_once_with()
        registry.register_model.assert_called_once()
    else:
        assert hooks.call_count == 1
        evidence.assert_not_called()
        registry.register_model.assert_not_called()
