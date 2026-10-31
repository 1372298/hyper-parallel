/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "aclnnop/aclnn_dense_lightning_indexer_grad_kl_loss.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {
using OptionalTensor = std::optional<at::Tensor>;
using Descriptor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using IntArray = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;

aclDataType DataType(at::ScalarType type) {
  switch (type) {
    case at::kHalf:
      return ACL_FLOAT16;
    case at::kBFloat16:
      return ACL_BF16;
    case at::kFloat:
      return ACL_FLOAT;
    default:
      TORCH_CHECK(false, "Unsupported dense DSA dtype: ", type);
  }
}

class Invocation {
 public:
  explicit Invocation(const at::Tensor &query) : guard_(query.device()), stream_(c10_npu::getCurrentNPUStream()) {}

  aclTensor *Tensor(const OptionalTensor &value) {
    if (!value.has_value()) {
      return nullptr;
    }
    const auto &tensor = *value;
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    auto *descriptor = aclCreateTensor(sizes.data(), sizes.size(), DataType(tensor.scalar_type()), strides.data(), 0,
                                       ACL_FORMAT_ND, sizes.data(), sizes.size(), tensor.data_ptr());
    TORCH_CHECK(descriptor != nullptr, "Cannot create dense DSA tensor descriptor");
    tensors_.emplace_back(descriptor, &aclDestroyTensor);
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream_);
    return descriptor;
  }

  aclIntArray *Lengths(const OptionalTensor &value) {
    if (!value.has_value()) {
      return nullptr;
    }
    auto *array = aclCreateIntArray(value->data_ptr<int64_t>(), value->numel());
    TORCH_CHECK(array != nullptr, "Cannot create dense DSA sequence lengths");
    lengths_.emplace_back(array, &aclDestroyIntArray);
    return array;
  }

  template <typename Execute>
  void Run(aclnnStatus status, uint64_t size, aclOpExecutor *executor, Execute execute, const at::Tensor &query) {
    TORCH_CHECK(status == 0, "Dense DSA workspace query failed: ", status, "; ", aclGetRecentErrMsg());
    auto workspace = at::empty({static_cast<int64_t>(size)}, query.options().dtype(at::kByte));
    status = execute(workspace.data_ptr(), size, executor, stream_.stream());
    TORCH_CHECK(status == 0, "Dense DSA execution failed: ", status, "; ", aclGetRecentErrMsg());
  }

 private:
  c10::DeviceGuard guard_;
  c10_npu::NPUStream stream_;
  std::vector<Descriptor> tensors_;
  std::vector<IntArray> lengths_;
};

std::vector<at::Tensor> IndexerStats(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                                     const OptionalTensor &query_lengths, const OptionalTensor &key_lengths,
                                     std::string layout) {
  Invocation call(query);
  std::vector<int64_t> shape = layout == "TND" ? std::vector<int64_t>{key.size(1), query.size(0)}
                                               : std::vector<int64_t>{query.size(0), key.size(2), query.size(1)};
  auto maximum = at::empty(shape, query.options().dtype(at::kFloat));
  auto total = at::empty_like(maximum);
  uint64_t size = 0;
  aclOpExecutor *executor = nullptr;
  constexpr auto window = std::numeric_limits<int64_t>::max();
  auto status = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
    call.Tensor(query), call.Tensor(key), call.Tensor(weights), call.Lengths(query_lengths), call.Lengths(key_lengths),
    layout.data(), 3, window, window, call.Tensor(maximum), call.Tensor(total), &size, &executor);
  call.Run(status, size, executor, aclnnDenseLightningIndexerSoftmaxLse, query);
  return {maximum, total};
}

std::vector<at::Tensor> KLLossGrad(const at::Tensor &query, const at::Tensor &key, const at::Tensor &index_query,
                                   const at::Tensor &index_key, const at::Tensor &weights, const at::Tensor &maximum,
                                   const at::Tensor &total, const at::Tensor &index_maximum,
                                   const at::Tensor &index_total, const OptionalTensor &query_rope,
                                   const OptionalTensor &key_rope, const OptionalTensor &query_lengths,
                                   const OptionalTensor &key_lengths, double scale, std::string layout) {
  Invocation call(query);
  auto dq = at::empty_like(index_query);
  auto dk = at::empty_like(index_key);
  auto dw = at::empty_like(weights);
  auto loss = at::empty({1}, query.options().dtype(at::kFloat));
  uint64_t size = 0;
  aclOpExecutor *executor = nullptr;
  constexpr auto window = std::numeric_limits<int64_t>::max();
  auto status = aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(
    call.Tensor(query), call.Tensor(key), call.Tensor(index_query), call.Tensor(index_key), call.Tensor(weights),
    call.Tensor(maximum), call.Tensor(total), call.Tensor(index_maximum), call.Tensor(index_total),
    call.Tensor(query_rope), call.Tensor(key_rope), call.Lengths(query_lengths), call.Lengths(key_lengths), scale,
    layout.data(), 3, window, window, call.Tensor(dq), call.Tensor(dk), call.Tensor(dw), call.Tensor(loss), &size,
    &executor);
  call.Run(status, size, executor, aclnnDenseLightningIndexerGradKLLoss, query);
  return {dq, dk, dw, loss};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("indexer_stats", &IndexerStats);
  module.def("kl_loss_grad", &KLLossGrad);
}
