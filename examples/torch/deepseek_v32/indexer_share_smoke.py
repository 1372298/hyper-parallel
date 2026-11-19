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
"""Single-NPU HF Indexer Share training, replay parity and checkpoint/resume smoke."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

from hyper_parallel.components.modules import DeepseekV32SparseAttention
from hyper_parallel.components.modules.attention.deepseek_sparse_attention import deepseek_sparse_attention as impl
from hyper_parallel.models.deepseek_v32.adapter.indexer_share import install_indexer_share


def build(mode: str, warmup: bool, recompute: bool) -> torch.nn.Module:
    """Build three real HF decoder layers with one shared Indexer group."""
    config = DeepseekV32Config(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=32, num_key_value_heads=32, q_lora_rank=128, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        index_n_heads=32, index_head_dim=128, index_topk=1024,
        mlp_layer_types=["dense"] * 3, max_position_embeddings=2048,
        attention_dropout=0.0, use_cache=False, tie_word_embeddings=False,
    )
    config.dsa_indexer_types = ["full", "shared", "shared"]
    config.dsa_index_share_loss = mode
    config.dsa_loss_coeff = 0.1
    config.dsa_dense_warm_up = warmup
    config._attn_implementation = "eager"
    model = DeepseekV32ForCausalLM(config)
    for index, layer in enumerate(model.model.layers):
        layer.self_attn = DeepseekV32SparseAttention(module=layer.self_attn, module_fqn=f"model.layers.{index}.self_attn")
    install_indexer_share(model)
    if recompute:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device="npu", dtype=torch.bfloat16).train()


def loss_for(model: torch.nn.Module, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Run the real HF model and FP32 cross entropy, retaining injected KL edges."""
    logits = model(input_ids=tokens, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten())


def run(args: argparse.Namespace) -> dict:
    """Validate exact replay, train three steps, then compare a resumed update."""
    torch.npu.set_device(0)
    torch.manual_seed(1598)
    torch.use_deterministic_algorithms(True)
    model = build(args.mode, args.warmup, False)
    replay = build(args.mode, args.warmup, True)
    replay.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randint(0, 256, (1, args.sequence), device="npu")
    labels = torch.randint(0, 256, tokens.shape, device="npu")
    with patch.object(impl, "dsa_indexer", wraps=impl.dsa_indexer) as indexer:
        expected = loss_for(model, tokens, labels)
        expected.backward()
        selection_calls = indexer.call_count
    if selection_calls != (0 if args.warmup else 1):
        raise AssertionError(f"Unexpected selection calls: {selection_calls}")
    actual = loss_for(replay, tokens, labels)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for (name, param), (_, reference) in zip(replay.named_parameters(), model.named_parameters()):
        if param.grad is None or reference.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        torch.testing.assert_close(param.grad, reference.grad, rtol=0, atol=0, msg=name)
    del model
    replay.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(replay.parameters(), lr=1e-4)
    losses = []
    gradient_norms = []
    leader = replay.model.layers[0].self_attn.indexer
    before = {name: p.detach().clone() for name, p in leader.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(replay, tokens, labels)
        loss.backward()
        norm = 0.0
        for name, parameter in replay.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing/nonfinite gradient: {name}")
            if ".indexer." in name:
                if not parameter.grad.float().norm() > 0:
                    raise AssertionError(f"Zero Indexer gradient: {name}")
                norm += parameter.grad.float().square().sum().item()
        losses.append(loss.item())
        gradient_norms.append(norm**0.5)
        optimizer.step()
    if not any(not torch.equal(before[name], p) for name, p in leader.named_parameters()):
        raise AssertionError("Indexer parameters did not update")
    if any(".indexer." in name and "layers.0." not in name for name in replay.state_dict()):
        raise AssertionError("Shared layers unexpectedly own Indexer parameters")
    checkpoint_path = args.output.with_suffix(".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": replay.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
    resumed = build(args.mode, args.warmup, True)
    restored = torch.load(checkpoint_path, map_location="npu", weights_only=True)
    resumed.load_state_dict(restored["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    for candidate, opt in ((replay, optimizer), (resumed, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        loss_for(candidate, tokens, labels).backward()
        opt.step()
    for (name, parameter), (_, restored_parameter) in zip(replay.named_parameters(), resumed.named_parameters()):
        torch.testing.assert_close(parameter, restored_parameter, atol=0, rtol=0, msg=name)
    result = dict(mode=args.mode, warmup=args.warmup, sequence=args.sequence, topk=1024,
                  selection_calls=selection_calls, losses=losses, indexer_gradient_norms=gradient_norms,
                  replay_all_gradients_exact=True, checkpoint_resume_exact=True,
                  parameters=sum(p.numel() for p in replay.parameters()),
                  indexer_parameters=sum(p.numel() for p in leader.parameters()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("leader", "served"), default="served")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--sequence", type=int, default=1056)
    parser.add_argument("--output", type=Path, default=Path("indexer_share_smoke.json"))
    run(parser.parse_args())
