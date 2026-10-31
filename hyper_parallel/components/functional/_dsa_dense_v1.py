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
"""Torch binding for the two dense V1 ACLNN interfaces absent from the package."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import ModuleType

import torch  # pylint: disable=forbidden-backend-import
import torch_npu  # pylint: disable=forbidden-backend-import
from torch.utils.cpp_extension import load  # pylint: disable=forbidden-backend-import

_BUILD_LOCK = Lock()


@lru_cache(maxsize=1)
def _build_extension() -> ModuleType:
    """Compile against the installed Torch NPU and CANN libraries on first use."""
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    npu = Path(torch_npu.__file__).parent
    return load(
        name="hyper_parallel_dsa_dense_v1",
        sources=[str(Path(__file__).parent / "csrc/dsa_dense_v1.cpp")],
        extra_include_paths=[str(cann / "include"), str(npu / "include"),
                             str(npu / "include/third_party/hccl/inc")],
        extra_cflags=["-O2"],
        extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu", f"-L{cann / 'lib64'}", "-lopapi", "-lascendcl"],
    )


def _extension() -> ModuleType:
    """Serialize first-use compilation within one Python process."""
    with _BUILD_LOCK:
        return _build_extension()


def _host_lengths(lengths: torch.Tensor | None) -> torch.Tensor | None:
    """Use the classic ACL int-array ABI for cumulative document ends."""
    return None if lengths is None else lengths.to(device="cpu", dtype=torch.int64).contiguous()


def _indexer_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_lengths: torch.Tensor | None,
    key_lengths: torch.Tensor | None,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V1's float32 indexer max and exponential sum."""
    return _extension().indexer_stats(
        query.contiguous(), key.contiguous(), weights.contiguous(),
        _host_lengths(query_lengths), _host_lengths(key_lengths), layout,
    )


def _kl_loss_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    weights: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    indexer_lse: torch.Tensor,
    scale: float,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    query_lengths: torch.Tensor | None,
    key_lengths: torch.Tensor | None,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute fused teacher, scalar KL, and explicit indexer gradients."""
    values = (query, key, index_query, index_key, weights, softmax_max, softmax_sum,
              indexer_lse, torch.ones_like(indexer_lse), query_rope, key_rope)
    return _extension().kl_loss_grad(
        *(None if value is None else value.contiguous() for value in values),
        _host_lengths(query_lengths), _host_lengths(key_lengths), scale, layout,
    )
