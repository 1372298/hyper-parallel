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
"""CANN cross-layer sharing CP numerical self-consistency, with full-gradient evidence."""

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RotaryEmbedding

from hyper_parallel.components.modules import DeepseekV32SparseAttention
from hyper_parallel.components.modules.attention.deepseek_sparse_attention import deepseek_sparse_attention as impl
from hyper_parallel.components.modules.attention.deepseek_sparse_attention import IndexShareContext
from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.distributed.mesh import MeshContext


class Stack(nn.Module):
    """The same residual attention stack, with only CP selection differing."""

    def __init__(
        self, sources: list[nn.Module], mesh: MeshContext | None, dense: bool, overlap: bool, recompute: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DeepseekV32SparseAttention(module=deepcopy(source), mesh=mesh,
                                    dense_warmup=dense, overlap=overlap) for source in sources])
        self.recompute = recompute

    def forward(
        self, hidden: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], positions: torch.Tensor,
    ) -> torch.Tensor:
        """Pass one invocation carrier through the optionally checkpointed stack."""
        context = IndexShareContext()
        for layer in self.layers:
            kwargs = dict(position_embeddings=rope, position_ids=positions, index_share_context=context)
            result = (
                checkpoint(layer, hidden, use_reentrant=False, **kwargs)
                if self.recompute else layer(hidden, **kwargs)
            )
            hidden = hidden + 0.125 * result[0]
        return hidden


def digest(values: list[torch.Tensor]) -> str:
    """Fingerprint full values, shapes and dtypes before either execution."""
    result = hashlib.sha256()
    for value in values:
        cpu = value.detach().cpu().contiguous()
        result.update(str((cpu.shape, cpu.dtype)).encode())
        result.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict[str, Any]:
    """Apply both fixed numerical bounds to complete tensor values."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    difference = actual - expected
    absolute = difference.abs().max().item()
    relative = (difference.norm() / expected.norm().clamp_min(1e-12)).item()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return dict(name=name, max_abs=absolute, relative_l2=relative, finite=finite,
                passed=finite and absolute <= 0.008 and relative <= 0.05)


def membership(indices: torch.Tensor, length: int, offset: int) -> torch.Tensor:
    """Compare valid causal membership independently of padded selection slots."""
    indices = indices.cpu().reshape(-1, indices.shape[-1]).long()
    result = torch.zeros(indices.shape[0], length, dtype=torch.bool)
    for query, values in enumerate(indices):
        valid = values[(values >= 0) & (values <= offset + query)]
        result[query, valid] = True
    return result


def run_case(
    context: MeshContext, dense: bool, mode: str, overlap: bool, recompute: bool, roles: list[str],
) -> dict[str, Any]:
    """Compare CP against identical unsharded weights, inputs and real RoPE."""
    rank, degree = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    length = 64 if dense else 1056
    local_length = length // degree
    start, end = rank * local_length, (rank + 1) * local_length
    torch.manual_seed(15981)
    config = DeepseekV32Config(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=len(roles),
        num_attention_heads=32, num_key_value_heads=32, q_lora_rank=128, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        index_n_heads=32, index_head_dim=128, index_topk=1024,
        mlp_layer_types=["dense"] * len(roles), max_position_embeddings=2048,
        attention_dropout=0.0, use_cache=False, tie_word_embeddings=False,
    )
    config.dsa_indexer_types = roles
    config.dsa_index_share_loss = mode
    config.dsa_loss_coeff = 0.1
    model = DeepseekV32ForCausalLM(config)
    sources = [layer.self_attn.bfloat16() for layer in model.model.layers]
    reference = Stack(sources, None, dense, overlap, False).to(device)
    parallel = Stack(sources, context, dense, overlap, recompute).to(device)
    state_hash = digest(list(reference.parameters()))
    assert state_hash == digest(list(parallel.parameters()))
    hashes = [None] * degree
    dist.all_gather_object(hashes, state_hash)
    assert len(set(hashes)) == 1
    torch.manual_seed(15982)
    value = (torch.randn(1, length, 128) * 0.2).bfloat16()
    positions = torch.arange(length).unsqueeze(0)
    rope = DeepseekV32RotaryEmbedding(config)(value, positions)
    data_hash = digest([value, positions, *rope])
    dist.all_gather_object(hashes, data_hash)
    assert len(set(hashes)) == 1
    full = value.to(device).requires_grad_()
    local = value[:, start:end].contiguous().to(device).requires_grad_()
    selections = {"reference": [], "parallel": []}
    native = impl.dsa_indexer

    def capture(label):
        def call(*args, **kwargs):
            result = native(*args, **kwargs)
            selections[label].append(result[0].detach().cpu())
            return result
        return call

    with aux_loss_scale_context(), patch.object(impl, "dsa_indexer", side_effect=capture("reference")):
        expected = reference(full, tuple(t.to(device) for t in rope), positions.to(device))
        expected_loss = expected.float().square().mean()
        bind_aux_loss_scale(expected_loss)
        expected_loss.backward()
    with aux_loss_scale_context(), patch.object(impl, "dsa_indexer", side_effect=capture("parallel")):
        actual = parallel(local, tuple(t[:, start:end].contiguous().to(device) for t in rope),
                          positions[:, start:end].contiguous().to(device))
        forward_calls = len(selections["parallel"])
        local_loss = actual.float().square().mean()
        bind_aux_loss_scale(local_loss)
        (local_loss / degree).backward()
    global_loss = local_loss.detach().clone() / degree
    dist.all_reduce(global_loss)
    checks = [compare(actual, expected[:, start:end], "output"),
              compare(local.grad, full.grad[:, start:end], "input_gradient"),
              compare(global_loss, expected_loss, "loss")]
    for name, parameter in parallel.named_parameters():
        assert parameter.grad is not None, name
        gradient = parameter.grad.clone()
        dist.all_reduce(gradient)
        checks.append(compare(gradient, reference.get_parameter(name).grad, "gradient " + name))
    expected_calls = 0 if dense else roles.count("full")
    assert forward_calls == expected_calls
    different_queries = []
    if not dense:
        for idx in range(expected_calls):
            expected_mask = membership(selections["reference"][idx], length, 0)[start:end]
            actual_mask = membership(selections["parallel"][idx], length, start)
            different_queries.append(int((expected_mask != actual_mask).any(-1).sum()))
    result = dict(rank=rank, cp=degree, dense=dense, mode=mode, overlap=overlap, recompute=recompute,
                  roles=roles, initial_state_sha256=state_hash, input_sha256=data_hash,
                  forward_topk_calls=forward_calls, different_topk_queries=different_queries,
                  comparisons=checks, passed=all(row["passed"] for row in checks) and not any(different_queries))
    return result


def main() -> None:
    """Run nine cases per rank; fail on any output, gradient or selection mismatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    if dist.get_world_size() not in (2, 4):
        raise ValueError("This validation matrix requires torchrun with 2 or 4 ranks")
    context = MeshContext(cp_size=dist.get_world_size(), dp_shard_size=dist.get_world_size())
    context.build_meshs("npu", dist.get_world_size())
    cases = []
    for dense in (False, True):
        for mode in ("served", "leader"):
            for overlap, recompute in ((False, False), (True, True)):
                result = run_case(context, dense, mode, overlap, recompute, ["full", "shared", "shared"])
                cases.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}), flush=True)
    cases.append(run_case(context, False, "served", True, True, ["full", "shared", "full", "shared"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{dist.get_rank()}.json").write_text(json.dumps(cases, indent=2) + "\n")
    failed = torch.tensor(int(not all(case["passed"] for case in cases)), device="npu")
    dist.all_reduce(failed)
    dist.destroy_process_group()
    if failed.item():
        raise SystemExit("CP module precision failed; inspect rank JSON without changing thresholds")


if __name__ == "__main__":
    main()
