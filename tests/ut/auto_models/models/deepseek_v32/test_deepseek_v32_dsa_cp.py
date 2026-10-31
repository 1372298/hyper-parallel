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
"""Unit tests for DeepSeek-V3.2 DSA Context Parallel length conversion."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention, _dense_flash_attention
from hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel import (
    _resolve_cp_sequence_lengths,
    _run_cp_dsa_kernels,
)


class TestDeepseekV32DSAContextParallel(unittest.TestCase):
    """Validate hardware-independent CP metadata projection."""

    def test_sparse_paths_select_cann_backend(self):
        """Ordinary and CP sparse execution both route indexer, attention and KL to CANN."""
        attention_type = DeepseekV32DSAAttention
        index_states = (torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 2))
        attention_states = (
            torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 1, 2),
        )
        lengths = (torch.tensor([2], dtype=torch.int32),) * 2
        indices = torch.zeros(2, 1, 4, dtype=torch.int32)
        flat_index_states = tuple(tensor.flatten(0, 1) for tensor in index_states)
        expected = torch.ones(1, 2, 2, 6)
        stats = torch.ones(1, 2, 2)
        loss = torch.tensor(0.5)
        module = SimpleNamespace(
            training=True, freeze_dsa=False, dsa_loss_coeff=0.7, scaling=0.5, index_topk=4,
            _project_index_states=lambda *_args: index_states,
            _get_actual_seq_len=attention_type._get_actual_seq_len,
        )
        module._run_indexer = MethodType(attention_type._run_indexer, module)
        module._apply_auxiliary_loss = MethodType(attention_type._apply_auxiliary_loss, module)
        cp_path = "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel"
        attention_path = "hyper_parallel.components.modules.dsa_attention"
        for use_cp in (False, True):
            lookup_path = cp_path if use_cp else attention_path
            with self.subTest(use_cp=use_cp), patch(
                lookup_path + ".dsa_indexer", return_value=(indices, *flat_index_states),
            ) as indexer, patch(
                lookup_path + ".dsa_sparse_attention", return_value=(expected, stats, stats),
            ) as attention, patch(attention_path + ".dsa_kl_loss", return_value=loss) as kl_loss, patch(
                "hyper_parallel.components.modules.dsa_attention.aux_loss_auto_scale", return_value=expected,
            ) as attach_loss:
                if use_cp:
                    output = _run_cp_dsa_kernels(
                        module, index_states, attention_states, lengths, use_dense=False,
                    )
                else:
                    output = attention_type._compute_sparse_attention(
                        module, torch.zeros(1, 2, 8), None, attention_states, None, None, {},
                    )

                self.assertIs(output, expected)
                indexer.assert_called_once()
                attention.assert_called_once()
                kl_loss.assert_called_once()
                self.assertEqual(indexer.call_args.kwargs["backend"], "cann")
                self.assertEqual(attention.call_args.kwargs["backend"], "cann")
                self.assertEqual(kl_loss.call_args.kwargs["backend"], "cann")
                attach_loss.assert_called_once_with(expected, loss)
                self.assertEqual(kl_loss.call_args.args[3].shape, (2, 2, 4))

    def test_dense_lengths_project_to_local_query_and_causal_prefix(self):
        """Dense batches retain one CP-local segment for every input row."""
        query, key = _resolve_cp_sequence_lengths(
            None,
            {},
            batch_size=2,
            local_seq_len=4,
            cp_size=2,
            cp_rank=1,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(query, torch.tensor([4, 8], dtype=torch.int32))
        torch.testing.assert_close(key, torch.tensor([8, 16], dtype=torch.int32))

    def test_packed_lengths_preserve_empty_rank_local_segments(self):
        """Packed documents crossing a CP shard keep aligned Q/KV numbering."""
        rank_zero_query, rank_zero_key = _resolve_cp_sequence_lengths(
            [7, 21, 32],
            {},
            batch_size=1,
            local_seq_len=16,
            cp_size=2,
            cp_rank=0,
            device=torch.device("cpu"),
        )
        rank_one_query, rank_one_key = _resolve_cp_sequence_lengths(
            [7, 21, 32],
            {},
            batch_size=1,
            local_seq_len=16,
            cp_size=2,
            cp_rank=1,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(rank_zero_query, torch.tensor([7, 16, 16], dtype=torch.int32))
        torch.testing.assert_close(rank_zero_key, torch.tensor([7, 16, 16], dtype=torch.int32))
        torch.testing.assert_close(rank_one_query, torch.tensor([0, 5, 16], dtype=torch.int32))
        torch.testing.assert_close(rank_one_key, torch.tensor([7, 21, 32], dtype=torch.int32))

    def test_incomplete_packed_boundaries_are_rejected(self):
        """Global boundaries must cover the complete unsharded token stream."""
        with self.assertRaisesRegex(ValueError, "final query sequence length"):
            _resolve_cp_sequence_lengths(
                [3, 7],
                {},
                batch_size=1,
                local_seq_len=4,
                cp_size=2,
                cp_rank=0,
                device=torch.device("cpu"),
            )

    def test_dense_attention_output_uses_fa_and_skips_target_without_loss(self):
        """Empty CP query segments still advance the key document offset."""
        module = SimpleNamespace(scaling=0.5)
        query = torch.zeros(1, 2, 2, 4)
        key = torch.arange(48.).reshape(1, 6, 2, 4)
        value = torch.zeros(1, 6, 2, 3)
        lengths = (torch.tensor([0, 2]), torch.tensor([2, 6]))
        expected = torch.ones(1, 2, 2, 3)
        with patch(
            "hyper_parallel.components.modules.dsa_attention.npu_fusion_attention_forward",
            return_value=(expected, None),
        ) as fa:
            output, maximum, total = _dense_flash_attention(
                module, query, key, value, lengths, need_stats=False,
            )
        torch.testing.assert_close(output, expected)
        self.assertIsNone(maximum)
        self.assertIsNone(total)
        self.assertEqual(fa.call_args.args[1].shape, (1, 2, 2, 4))
        torch.testing.assert_close(fa.call_args.args[2], key[:, 2:].transpose(1, 2))
        self.assertTrue(fa.call_args.kwargs["is_causal"])
        self.assertEqual(fa.call_args.kwargs["sparse_mode"], 3)
        self.assertFalse(fa.call_args.kwargs["return_softmax_stats"])

    def test_dense_attention_reuses_flash_attention_softmax_output(self):
        """Training reuses FA max/sum and returns fused KL's head/token layout."""
        module = SimpleNamespace(scaling=0.5)
        states = (torch.zeros(1, 2, 2, 4),) * 2 + (torch.zeros(1, 2, 2, 3),)
        lengths = (torch.tensor([2]), torch.tensor([2]))
        expected_output = torch.ones(1, 2, 2, 3)
        maximum = torch.arange(32.).reshape(1, 2, 2, 8)
        with patch(
            "hyper_parallel.components.modules.dsa_attention.npu_fusion_attention_forward",
            return_value=(expected_output, None, maximum, maximum + 1),
        ) as fa:
            output, actual_max, actual_sum = _dense_flash_attention(module, *states, lengths, need_stats=True)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(actual_max, maximum[0, ..., :1])
        torch.testing.assert_close(actual_sum, maximum[0, ..., :1] + 1)
        self.assertTrue(fa.call_args.kwargs["return_softmax_stats"])

    def test_dense_cp_branch_uses_dense_ops_without_sparse_indexer(self):
        """The wrapper's dense selection bypasses sparse indexer/attention calls."""
        module = SimpleNamespace(
            scaling=1.0,
            training=True,
            freeze_dsa=False,
            dsa_loss_coeff=0.1,
        )
        index_states = tuple(torch.zeros(1, 2, 2, 4) for _ in range(2)) + (torch.zeros(1, 2, 2),)
        attention_states = tuple(torch.zeros(1, 2, heads, dim) for heads, dim in (
            (2, 3), (1, 3), (2, 1), (1, 1),
        ))
        lengths = (torch.tensor([2]), torch.tensor([2]))
        dense_output = torch.ones(1, 2, 2, 3)
        with patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel."
            "_deepseek_v32_dense_attention",
            return_value=dense_output,
        ) as dense_attention, patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel.dsa_indexer",
        ) as sparse_indexer:
            output = _run_cp_dsa_kernels(
                module, index_states, attention_states, lengths, use_dense=True,
            )

        self.assertIs(output, dense_output)
        dense_attention.assert_called_once_with(module, index_states, attention_states, lengths)
        sparse_indexer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
