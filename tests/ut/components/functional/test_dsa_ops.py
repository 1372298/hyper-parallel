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
"""Hardware-independent DSA allocation, argument and gradient contracts."""

import builtins
import sys
import unittest
from types import ModuleType
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional import dsa_ops
from hyper_parallel.components.functional import dsa_dense as dense_api
from hyper_parallel.components.functional import _dsa_dense_v1 as dense_v1
from hyper_parallel.components.functional import dsa_indexer, dsa_kl_loss, dsa_sparse_attention


class TestDSAOps(unittest.TestCase):
    """Verify adapter contracts without executing NPU kernels."""

    def test_indexer_adapts_tnd_lengths_and_value_dtype(self):
        """The package receives cu-seqlens and preserves the public value dtype."""
        query = torch.zeros(8, 2, 4, dtype=torch.bfloat16)
        key = torch.zeros(8, 1, 4, dtype=torch.bfloat16)
        weights = torch.zeros(8, 2, dtype=torch.bfloat16)
        lengths = torch.tensor([4, 8], dtype=torch.int32)
        expected_indices = torch.zeros(8, 1, 16, dtype=torch.int32)
        expected_values = torch.zeros(8, 1, 16, dtype=torch.float32)

        def package_indexer(*_args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            """Check packed lengths passed to the package API."""
            torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 4, 8], dtype=torch.int32))
            torch.testing.assert_close(kwargs["cu_seqlens_k"], torch.tensor([0, 4, 8], dtype=torch.int32))
            return expected_indices, expected_values

        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "lightning_indexer", side_effect=package_indexer):
            indices, values = dsa_ops._lightning_indexer(
                query, key, weights, actual_seq_qlen=lengths, actual_seq_klen=lengths,
                layout="TND", sparse_count=16,
            )
        self.assertIs(indices, expected_indices)
        self.assertEqual(values.dtype, query.dtype)

    def test_package_kl_target_uses_document_local_indices(self):
        """Sparse KL target reconstruction resets key offsets at each document."""
        query = torch.tensor([[[1., 0.]], [[2., 0.]], [[3., 0.]], [[4., 0.]]])
        key = torch.tensor([[[1., 0.]], [[2., 0.]], [[10., 0.]], [[20., 0.]]])
        indices = torch.tensor([[[0, -1]], [[0, 1]], [[0, -1]], [[0, 1]]], dtype=torch.int32)
        maximum = torch.zeros(1, 4, 1)
        total = torch.ones(1, 4, 1)
        target = dsa_ops._package_sparse_kl_target(
            query, key, indices, maximum, total, 1.0, None, None,
            torch.tensor([2, 4], dtype=torch.int32), torch.tensor([2, 4], dtype=torch.int32),
        )
        self.assertEqual(target.shape, (4, 1, 2))
        self.assertGreater(target[3, 0, 1], target[3, 0, 0])

    def test_tnd_requires_both_sequence_boundaries(self):
        """Incomplete packed metadata must fail before launching a kernel."""
        with patch.object(dsa_ops, "_validate"):
            with self.assertRaisesRegex(ValueError, "cumulative"):
                dsa_ops._dense_indexer_lse_v2(
                    torch.zeros(4, 8, 128), torch.zeros(4, 1, 128), torch.zeros(4, 8), layout="TND")

    def test_attention_requires_shared_value_and_paired_rope(self):
        """Keep only the CANN-specific shared-KV and paired-RoPE constraints."""
        query, key = torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 1, 4)
        indices = torch.zeros(1, 3, 1, 2048, dtype=torch.int32)
        with patch.object(dsa_ops, "_validate"):
            with self.assertRaisesRegex(ValueError, "alias"):
                dsa_ops._sparse_attention(query, key, key.clone(), indices, .5)
            with self.assertRaisesRegex(ValueError, "together"):
                dsa_ops._sparse_attention(
                    query, key, key, indices, .5, query_rope=torch.zeros(1, 3, 2, 2)
                )

    def test_empty_kl_query_fails_before_launch(self):
        """Empty batches cannot reach CANN or divide a precomputed loss by zero."""
        inputs = (
            torch.empty(0, 1, 4), torch.empty(0, 1, 4), torch.empty(0, 1),
            torch.empty(0, 1, 1), torch.empty(1, 0),
        )
        with patch.object(dense_api, "_kl_loss_grad") as kernel:
            with self.assertRaisesRegex(ValueError, "at least one valid query"):
                dense_api.dsa_dense_indexer_kl_loss(
                    *inputs[:3], inputs[0], inputs[1], inputs[3], inputs[3], inputs[4], scale=0.5,
                )
            kernel.assert_not_called()

    def test_dense_indexer_lse_uses_v2_metadata_and_lse_entrypoint(self):
        """Dense warm-up calls the V2 metadata and LSE package APIs."""
        query = torch.zeros(8, 2, 128, dtype=torch.bfloat16)
        key = torch.zeros(8, 1, 128, dtype=torch.bfloat16)
        weights = torch.zeros(8, 2, dtype=torch.bfloat16)
        lengths = torch.tensor([4, 8], dtype=torch.int32)
        lse = torch.zeros(1, 8)

        def metadata_op(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            """Return metadata arguments for verification by the wrapped op."""
            return kwargs

        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "dense_lightning_indexer_softmax_lse_metadata", side_effect=metadata_op), \
                patch.object(dsa_ops, "dense_lightning_indexer_softmax_lse", return_value=lse):
            result = dsa_ops._dense_indexer_lse_v2(
                query, key, weights, actual_seq_qlen=lengths, actual_seq_klen=lengths, layout="TND",
            )
        self.assertIs(result, lse)

    def test_dense_binding_preserves_lse_and_converts_lengths(self):
        """The fused V1 KL accepts LSE as the equivalent pair (maximum=LSE, sum=1)."""
        query = torch.zeros(5, 2, 128)
        key = torch.zeros(8, 2, 128)
        index_key = torch.zeros(8, 1, 128)
        weights = torch.zeros(5, 2)
        stats = torch.ones(2, 5, 1)
        lse = torch.arange(5.).unsqueeze(0)
        lengths_q = torch.tensor([2, 5], dtype=torch.int32)
        lengths_k = torch.tensor([3, 8], dtype=torch.int32)
        with patch.object(dense_v1, "_extension") as extension:
            dense_v1._kl_loss_grad(
                query, key, query, index_key, weights, stats, stats, lse,
                .5, None, None, lengths_q, lengths_k, "TND",
            )
        args = extension.return_value.kl_loss_grad.call_args.args
        torch.testing.assert_close(args[7], lse)
        torch.testing.assert_close(args[8], torch.ones_like(lse))
        self.assertIsNone(args[9])
        self.assertIsNone(args[10])
        torch.testing.assert_close(args[11], lengths_q.long())
        torch.testing.assert_close(args[12], lengths_k.long())

    def test_attention_accumulates_shared_key_value_and_rope_gradients(self):
        """Aliased K/V receive both derivatives; softmax statistics stay detached."""
        query = torch.zeros(1, 3, 2, 4, requires_grad=True)
        key = torch.zeros(1, 3, 1, 4, requires_grad=True)
        qrope = torch.zeros(1, 3, 2, 2, requires_grad=True)
        krope = torch.zeros(1, 3, 1, 2, requires_grad=True)
        indices = torch.zeros(1, 3, 1, 4, dtype=torch.int32)

        forward = (torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 2, 1), torch.zeros(1, 3, 2, 1))
        backward = tuple(torch.full_like(value, factor) for value, factor in zip(
            (query, key, key, qrope, krope), (2, 3, 5, 7, 11)))
        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "_package_sparse_attention", return_value=forward), \
                patch.object(dsa_ops, "_package_sparse_attention_grad", return_value=backward):
            output, maximum, total = dsa_ops._sparse_attention(
                query, key, key, indices, .5, query_rope=qrope, key_rope=krope)
            (3 * output.sum()).backward()
        torch.testing.assert_close(query.grad, torch.full_like(query, 2))
        torch.testing.assert_close(key.grad, torch.full_like(key, 8))
        torch.testing.assert_close(qrope.grad, torch.full_like(qrope, 7))
        torch.testing.assert_close(krope.grad, torch.full_like(krope, 11))
        self.assertFalse(maximum.requires_grad)
        self.assertFalse(total.requires_grad)

    def test_dense_kl_routes_explicit_indexer_gradients_once(self):
        """Dense KL normalizes only indexer gradients and ignores teacher inputs."""
        inputs = [torch.ones(4, 2, 3, requires_grad=True), torch.ones(4, 1, 3, requires_grad=True),
                  torch.ones(4, 2, requires_grad=True)]
        teacher = torch.zeros(4, 2, 3, requires_grad=True)
        stats = torch.ones(2, 4, 1)
        lse = torch.ones(1, 4)
        lengths = torch.tensor([4], dtype=torch.int32)
        gradients = [torch.full_like(value, factor) for value, factor in zip(inputs, (2, 3, 5))]
        with patch.object(dense_api, "_kl_loss_grad") as kernel:
            kernel.return_value = (*gradients, torch.tensor([8.]))
            loss = dense_api.dsa_dense_indexer_kl_loss(
                *inputs, teacher, teacher, stats, stats, lse, scale=0.5,
                query_lengths=lengths, key_lengths=lengths,
                layout="TND", loss_coefficient=.7,
            )
            (loss * 3).backward()
        torch.testing.assert_close(loss, torch.tensor(1.4))
        self.assertIsNone(teacher.grad)
        for value, gradient in zip(inputs, gradients):
            torch.testing.assert_close(value.grad, gradient * (.7 * 3 / 4))
        kernel.assert_called_once()

    def test_dense_auto_selects_v1_or_v2_by_device(self):
        """A2/A3 use the binding, while both 950 device names use package V2."""
        query = torch.zeros(4, 8, 128)
        key = torch.zeros(4, 1, 128)
        weights = torch.zeros(4, 8)
        for name, use_v2 in (("Ascend910B3", False), ("Ascend910C", False),
                             ("Ascend950PR", True), ("Ascend910_9599", True)):
            dense_api._uses_v2_lse.cache_clear()
            with self.subTest(name=name), patch.object(torch.npu, "get_device_name", return_value=name), patch.object(
                dense_api, "_indexer_stats", return_value=(torch.full((1, 4), 2.), torch.full((1, 4), 3.)),
            ) as v1, patch.object(dense_api, "_cann_dsa_dense_indexer_lse", return_value=torch.ones(1, 4)) as v2:
                result = dense_api.dsa_dense_indexer_lse(query, key, weights)
                self.assertEqual(v2.call_count, int(use_v2))
                self.assertEqual(v1.call_count, int(not use_v2))
                expected = torch.ones(1, 4) if use_v2 else 2 + torch.full((1, 4), 3.).log()
                torch.testing.assert_close(result, expected)
        dense_api._uses_v2_lse.cache_clear()

    def test_sparse_kl_routes_explicit_indexer_gradients_once(self):
        """Sparse KL keeps its own CANN contract and loss normalization."""
        inputs = [torch.ones(4, 2, 3, requires_grad=True), torch.ones(4, 1, 3, requires_grad=True),
                  torch.ones(4, 2, requires_grad=True)]
        teacher_query = torch.ones(4, 2, 3, requires_grad=True)
        teacher_key = torch.ones(4, 1, 3, requires_grad=True)
        stats = torch.ones(2, 4)
        lengths = torch.tensor([4], dtype=torch.int32)
        gradients = [torch.full_like(value, factor) for value, factor in zip(inputs, (2, 3, 5))]
        real_import = builtins.__import__

        def reject_omni_import(name, *args, **kwargs):
            """Ensure the CANN KL path does not import Omni custom operators."""
            if name == "omni_training_custom_ops":
                raise AssertionError("CANN backend attempted to import Omni custom operators")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=reject_omni_import), \
                patch.object(dsa_ops, "_cann_dsa_sparse_indexer_kl_loss_grad") as kernel:
            kernel.return_value = (*gradients, torch.tensor([8.]))
            loss = dsa_kl_loss(
                *inputs, teacher_query, teacher_key, torch.zeros(4, 1, 4, dtype=torch.int32),
                stats, stats, None, None, lengths, lengths, .5, .7,
                backend="cann",
            )
            (loss * 3).backward()
        torch.testing.assert_close(loss, torch.tensor(1.4))
        for value, gradient in zip(inputs, gradients):
            torch.testing.assert_close(value.grad, gradient * (.7 * 3 / 4))
        self.assertIsNone(teacher_query.grad)
        self.assertIsNone(teacher_key.grad)
        kernel.assert_called_once()

    def test_cann_entrypoints_keep_tnd_and_padding_contracts(self):
        """CANN DSA entry points flatten indexer states and pad attention output."""
        query = torch.zeros(2, 3, 2, 4, requires_grad=True)
        key = torch.zeros(2, 3, 1, 4, requires_grad=True)
        weights = torch.zeros(2, 3, 2, requires_grad=True)
        lengths = torch.tensor([3, 6])
        real_import = builtins.__import__

        def reject_omni_import(name, *args, **kwargs):
            """Ensure the CANN branch does not import Omni's custom operators."""
            if name == "omni_training_custom_ops":
                raise AssertionError("CANN backend attempted to import Omni custom operators")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=reject_omni_import), \
                patch.object(
                    dsa_ops,
                    "_cann_dsa_indexer",
                    return_value=(torch.zeros(6, 1, 2048, dtype=torch.int32), torch.zeros(6, 1, 2048)),
                ) as indexer:
            indices, iq, ik, iw = dsa_indexer(query, key, weights, lengths, lengths, 2048, backend="cann")
            self.assertEqual(iq.shape, (6, 2, 4))
            self.assertEqual(ik.shape, (6, 1, 4))
            self.assertEqual(iw.shape, (6, 2))
            self.assertTrue(iq.requires_grad)
            self.assertEqual(indexer.call_args.kwargs["layout"], "TND")
        qrope = torch.zeros(2, 3, 2, 2)
        krope = torch.zeros(2, 3, 1, 2)
        with patch.object(builtins, "__import__", side_effect=reject_omni_import), \
                patch.object(dsa_ops, "_cann_dsa_sparse_attention") as attention:
            attention.return_value = (torch.ones(6, 2, 4), torch.ones(1, 6, 2), torch.ones(1, 6, 2))
            output, _, _ = dsa_sparse_attention(
                query, key, qrope, krope, indices, .5, lengths, lengths, backend="cann",
            )
        self.assertEqual(output.shape, (2, 3, 2, 6))
        torch.testing.assert_close(output[..., :4], torch.ones_like(output[..., :4]))
        torch.testing.assert_close(output[..., 4:], torch.zeros_like(output[..., 4:]))

    def test_sparse_public_omni_default_and_explicit_skip_cann_imports(self):
        """Both Omni selections call the legacy ops without loading CANN adapters."""
        real_import = builtins.__import__
        imported_omni = []

        def reject_cann_import(name, *args, **kwargs):
            """Reject optional CANN imports while allowing the Omni stub."""
            if name == "hyper_parallel.components.functional.dsa_ops" or name.startswith("cann_ops_transformer"):
                raise AssertionError(f"Omni backend attempted to import {name}")
            if name == "omni_training_custom_ops":
                imported_omni.append(name)
            return real_import(name, *args, **kwargs)

        for backend in (None, "omni"):
            with self.subTest(backend=backend):
                query = torch.zeros(1, 2, 1, 4)
                key = torch.zeros(1, 2, 1, 4)
                weights = torch.zeros(1, 2, 1)
                lengths = torch.tensor([2], dtype=torch.int32)
                indices = torch.tensor([[[0, -1]], [[0, 1]]], dtype=torch.int32)
                values = torch.zeros(2, 1, 4)
                kwargs = {} if backend is None else {"backend": backend}
                indexer = Mock(return_value=(indices, values))
                attention = Mock(return_value=(values, torch.zeros(1, 2, 1), torch.ones(1, 2, 1)))
                query_index = torch.ones(2, 2, 3, requires_grad=True)
                key_index = torch.ones(2, 1, 3, requires_grad=True)
                merge_weight = torch.ones(2, 2, requires_grad=True)
                grad_index_query = torch.full_like(query_index, 2)
                grad_index_key = torch.full_like(key_index, 3)
                grad_weights = torch.full_like(merge_weight, 5)
                teacher_query = torch.ones(2, 1, 3, requires_grad=True)
                teacher_key = torch.ones(2, 1, 3, requires_grad=True)
                kl_loss = Mock(return_value=(
                    grad_index_query, grad_index_key, grad_weights, torch.tensor([8.]),
                ))
                omni_module = ModuleType("omni_training_custom_ops")
                with patch.object(builtins, "__import__", side_effect=reject_cann_import), \
                        patch.dict(sys.modules, {"omni_training_custom_ops": omni_module}), \
                        patch.dict(torch.ops.custom.__dict__, {
                            "npu_lightning_indexer_enhance": indexer,
                            "npu_sparse_flash_attention_enhance": attention,
                            "npu_sparse_lightning_indexer_grad_kl_loss_enhance": kl_loss,
                        }):
                    selected_indices, _, _, _ = dsa_indexer(
                        query, key, weights, lengths, lengths, 2, **kwargs,
                    )
                    result, _, _ = dsa_sparse_attention(
                        query, key, torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2),
                        selected_indices, .5, lengths, lengths, **kwargs,
                    )
                    loss = dsa_kl_loss(
                        query_index, key_index, merge_weight,
                        teacher_query, teacher_key,
                        selected_indices, torch.ones(1, 2, 1), torch.ones(1, 2, 1),
                        torch.zeros(2, 1, 2), torch.zeros(2, 1, 2), lengths, lengths, .5, .5,
                        **kwargs,
                    )
                    (loss * 3).backward()

                self.assertEqual(result.shape, (1, 2, 1, 6))
                indexer.assert_called_once()
                attention.assert_called_once()
                kl_loss.assert_called_once()
                torch.testing.assert_close(loss, torch.tensor(2.))
                torch.testing.assert_close(query_index.grad, torch.full_like(query_index, 1.5))
                torch.testing.assert_close(key_index.grad, torch.full_like(key_index, 2.25))
                torch.testing.assert_close(merge_weight.grad, torch.full_like(merge_weight, 3.75))
                self.assertIsNone(teacher_query.grad)
                self.assertIsNone(teacher_key.grad)
                self.assertEqual(imported_omni[-1], "omni_training_custom_ops")

    def test_sparse_backend_rejects_unknown_value(self):
        """Backend selection is explicit instead of silently falling back."""
        for function, argument_count in ((dsa_indexer, 6), (dsa_sparse_attention, 8), (dsa_kl_loss, 14)):
            with self.subTest(interface=function.__name__), self.assertRaisesRegex(
                ValueError, "backend must be 'cann' or 'omni'",
            ):
                function(*((None,) * argument_count), backend="unknown")
