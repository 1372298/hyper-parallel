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
"""CPU tests for sparse-attention construction, metadata and indexer scaling."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.components.modules.attention.deepseek_sparse_attention import (
    CPDeepseekSparseAttentionModule,
    CPDeepseekV32SparseAttentionModule,
    DeepseekSparseAttention,
    DeepseekSparseAttentionModule,
    DeepseekV32SparseAttention,
    DeepseekV32SparseAttentionModule,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.cp_deepseek_sparse_attention import (
    build_cp_sequence_metadata,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    SparseAttentionSequenceMetadata,
)


def _config() -> SimpleNamespace:
    """Return small native dimensions without a Transformers model dependency."""
    return SimpleNamespace(
        hidden_size=16, num_attention_heads=2, q_lora_rank=8, kv_lora_rank=8,
        qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        index_head_dim=8, index_n_heads=2, index_topk=1024,
        device="cpu", dtype=torch.float32,
    )


FAMILIES = (
    (DeepseekSparseAttention, DeepseekSparseAttentionModule, CPDeepseekSparseAttentionModule),
    (DeepseekV32SparseAttention, DeepseekV32SparseAttentionModule, CPDeepseekV32SparseAttentionModule),
)


class TestDeepseekSparseAttention(unittest.TestCase):
    """Cover family wiring and local numerical behavior without device kernels."""

    def setUp(self) -> None:
        """Keep parameter initialization deterministic and preserve the caller RNG."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(7)

    def test_native_config_selects_local_and_cp_members(self) -> None:
        """Both public factories construct the requested member directly from config."""
        for factory, local_type, cp_type in FAMILIES:
            for cp_size, expected_type in ((1, local_type), (2, cp_type)):
                with self.subTest(family=factory.__name__, cp_size=cp_size):
                    module = factory(config=_config(), context={"cp_size": cp_size})

                    self.assertIs(type(module), expected_type)
                    self.assertEqual(module.linear_qkv.weight.shape, (20, 16))
                    self.assertEqual(module.index_topk, 1024)
                    self.assertEqual({parameter.device.type for parameter in module.parameters()}, {"cpu"})

    def test_cp_promotion_preserves_parameters_and_eval_mode(self) -> None:
        """Promoting a loaded local member reuses its Parameters and training mode."""
        for factory, _, cp_type in FAMILIES:
            with self.subTest(family=factory.__name__):
                source = factory(config=_config()).eval()
                parameters = dict(source.named_parameters())
                values = {name: parameter.detach().clone() for name, parameter in parameters.items()}

                promoted = factory(module=source, context={"cp_size": 2})

                self.assertIs(type(promoted), cp_type)
                self.assertFalse(promoted.training)
                self.assertEqual(set(dict(promoted.named_parameters())), set(parameters))
                for name, parameter in promoted.named_parameters():
                    self.assertIs(parameter, parameters[name])
                    torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)

    def test_sequence_metadata_preserves_packed_documents(self) -> None:
        """Local endpoints and CP causal prefixes retain the same document boundaries."""
        for actual_seq_len, expected_ends in ((None, [8, 16]), ([6, 16], [6, 16])):
            with self.subTest(actual_seq_len=actual_seq_len):
                metadata = SparseAttentionSequenceMetadata.build(
                    batch_size=2, local_seq_len=8, device=torch.device("cpu"), actual_seq_len=actual_seq_len,
                )
                self.assertEqual(metadata.kv_prefix_length, 8)
                for endpoints in metadata.sequence_lengths:
                    torch.testing.assert_close(endpoints, torch.tensor(expected_ends, dtype=torch.int32))

        for rank, query_ends, key_ends in ((0, [4, 8], [4, 8]), (1, [2, 8], [6, 16])):
            with self.subTest(cp_rank=rank):
                metadata = build_cp_sequence_metadata(
                    batch_size=2, local_seq_len=4, device=torch.device("cpu"),
                    cp_size=2, cp_rank=rank, actual_seq_len=[6, 16],
                )
                self.assertEqual(metadata.kv_prefix_length, (rank + 1) * 4)
                torch.testing.assert_close(metadata.query_ends, torch.tensor(query_ends, dtype=torch.int32))
                torch.testing.assert_close(metadata.key_ends, torch.tensor(key_ends, dtype=torch.int32))

    def test_cp_rejects_zero_query_documents_on_every_rank(self) -> None:
        """Unsupported packed partitions fail before any rank starts communication."""
        for rank in (0, 1):
            with self.subTest(cp_rank=rank), self.assertRaisesRegex(ValueError, "zero-query packed documents"):
                build_cp_sequence_metadata(
                    batch_size=1, local_seq_len=4, device=torch.device("cpu"),
                    cp_size=2, cp_rank=rank, actual_seq_len=[4, 8],
                )

    def test_v32_indexer_weights_scale_without_intermediate_bf16_rounding(self) -> None:
        """The 32-head, 128-dimension scale equals an exact division by 64."""
        config = _config()
        config.dtype = torch.bfloat16
        config.index_n_heads, config.index_head_dim = 32, 128
        module = DeepseekV32SparseAttention(config=config)
        raw_weights = torch.linspace(1, 2, 32, device="cpu").to(torch.bfloat16)
        hidden_states = torch.zeros(1, 2, 16, dtype=torch.bfloat16, device="cpu")
        hidden_states[..., 0] = 1
        with torch.no_grad():
            module.indexer.weights_proj.weight.zero_()
            module.indexer.weights_proj.weight[:, 0].copy_(raw_weights)

        index_states, _ = module.prepare_attention(hidden_states, None, use_dense=False)

        expected = (raw_weights.float() / 64).to(torch.bfloat16).expand(1, 2, 32)
        torch.testing.assert_close(index_states[2], expected, rtol=0, atol=0)
