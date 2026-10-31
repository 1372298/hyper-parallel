# High-performance functions

`hyper_parallel.components.functional` provides reusable high-performance functions for Ascend NPU. CPU and GPU fallback implementations
are not included.

```python
from hyper_parallel.components.functional import rms_norm

output = rms_norm(x, weight)
```

RoPE functions keep the Transformers call contract:

```python
from hyper_parallel.components.functional import apply_rotary_pos_emb

query, key = apply_rotary_pos_emb(query, key, cos, sin)
```

Use `apply_rotary_pos_emb_interleave` for Transformers models whose source
attention calls the interleaved variant.

## MoE functions

The reusable MoE interfaces include:

- `grouped_matmul`: NPU grouped matrix multiplication with backward support.
- `moe_token_permute`: reorder routed tokens into expert-major order.
- `moe_token_unpermute`: restore expert outputs to token-major order.
- `swiglu`: fused NPU SwiGLU used by grouped experts.

```python
from hyper_parallel.components.functional import (
    moe_token_permute,
    moe_token_unpermute,
)

permuted_tokens, sorted_indices = moe_token_permute(tokens, expert_indices)
expert_outputs = experts(permuted_tokens, tokens_per_expert, routing_weights)
output = moe_token_unpermute(expert_outputs, sorted_indices, routing_probs)
```

In this example, `experts` is an instance of `hyper_parallel.components.modules.GroupedExperts`. It contains only local expert
weights and computation; routing, expert-parallel communication, and shared experts remain outside the module.
`hyper_parallel.components.modules.SharedExpert` provides the parameter-owning dense MLP used as a shared expert.

## Auxiliary-loss functions

`hyper_parallel.components.functional` also exports `aux_loss_auto_scale` and `set_aux_loss_scale` for auxiliary-loss gradient injection.

## DSA causal training operators

The sparse APIs remain in the existing `dsa_indexer`, `dsa_sparse_attention`,
and `dsa_kl_loss` modules. Their signatures preserve the Omni path and add an
optional `backend` keyword. `backend="omni"` is the default; pass
`backend="cann"` to select the CANN implementation. DeepSeek V3.2 explicitly
selects CANN for ordinary sparse execution and CP sparse execution. Other
callers, including generic `DSAAttention`, keep the default Omni behavior.
Dense warm-up remains in `dsa_dense.py`; the CP wrapper's `dsa_dense_warm_up`
state selects it, and `use_dense: true/false` can force one branch for a
complete run. Install the dependencies for the selected backend and matching
native OPP before calling these NPU-only functions.

| Public interface | Positional parameters | Keyword parameters | Backend behavior |
| --- | --- | --- | --- |
| `dsa_indexer` | `index_query, index_key, merge_weight, actual_seq_qlen, actual_seq_kvlen, sparse_count` | `backend="omni"` (default) or `backend="cann"` | Omni indexer or CANN `lightning_indexer` |
| `dsa_sparse_attention` | `query_nope, compressed_kv, query_rope, key_rope, topk_indices, scale, actual_seq_qlen, actual_seq_kvlen` | `backend="omni"` (default) or `backend="cann"` | Omni sparse attention or CANN sparse flash attention |
| `dsa_kl_loss` | `index_query, index_key, merge_weight, query, key, topk_indices, softmax_max, softmax_sum, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen, scale, loss_coeff` | `backend="omni"` (default) or `backend="cann"` | Omni sparse KL or CANN sparse KL |
| `dsa_dense_indexer_lse` | `index_query, index_key, head_weights` | `query_lengths=None, key_lengths=None, layout="TND"` | Auto: V1 ACLNN on A2/A3; package V2 LSE on 950 |
| `dsa_dense_indexer_kl_loss` | `index_query, index_key, head_weights, query, key, softmax_max, softmax_sum, indexer_lse` | `scale` (required), `query_rope=None, key_rope=None, query_lengths=None, key_lengths=None, layout="TND", loss_coefficient=1.0` | `aclnnDenseLightningIndexerGradKLLoss` |

For the public sparse attention/indexer interfaces, `actual_seq_qlen` and
`actual_seq_kvlen` are cumulative document ends without a leading zero. The
indexer accepts BSND inputs and returns int32 top-k plus flattened indexer Q/K/
weights. Attention accepts BSND states and returns output, softmax maximum, and
exponential sum. The loss receives flattened TND tensors and returns a scalar
token-mean auxiliary loss. Dense `query_lengths`/`key_lengths` are cumulative
ends for TND or per-batch valid lengths for BSND.

### Dependencies and initialization

Load the CANN and custom-operator environments before selecting the CANN
backend. The sparse APIs load their optional backend dependency on the selected
branch: `backend="omni"` requires `omni_training_custom_ops`, while
`backend="cann"` requires the `cann_ops_transformer` extension. Deploy matching
native OPP kernels for the target SoC as well as installing the Python
extension. Callers using the default Omni backend do not need the CANN Python
extension for these sparse APIs.

Single-card precision tests passed on Ascend 910B3 with Python 3.10.20,
Torch 2.6.0+cpu, torch_npu 2.6.0.post6, and CANN 9.1.0-beta.1 using the
standard ops-transformer package and its matching native OPP. Other
toolkit/vendor combinations require validation.

### Tensor contracts and training

The sparse interfaces are provided by the existing `dsa_indexer.py`,
`dsa_sparse_attention.py`, and `dsa_kl_loss.py` modules. They accept
`backend="omni"` or `backend="cann"`, with Omni as the default; use the same
backend for indexer, attention, and KL within a sparse training chain. V3.2
passes `backend="cann"` explicitly for ordinary and CP calls. Dense interfaces
remain in `dsa_dense.py` and are CANN-only because Omni has no corresponding
dense operator in this integration. Low-level CANN metadata adaptation lives
in `dsa_ops.py`; its `_package_*` helpers are private implementation details,
not user-facing APIs. Dense KL takes teacher Q/K, FA maximum/sum, optional
rotary states, and detached indexer LSE.
Q/K and rotary tensors use float16 or bfloat16; the tested indexer also requires
weights in the query dtype. Sparse indices are int32. Softmax statistics are
float32, with the following shapes for the tested single-KV-head path:

| Statistics | BSND | TND |
| --- | --- | --- |
| Dense indexer LSE | `[B,1,S]` | `[1,T]` |
| Dense teacher maximum/sum | `[B,Nq,S,1]` | `[Nq,T,1]` |
| Sparse teacher maximum/sum | `[B,1,S,Nq]` | `[1,T,Nq]` |

TND sequence metadata contains cumulative document ends, without a leading zero;
BSND metadata contains per-batch valid lengths. The public CANN APIs expose these
as `query_lengths` and `key_lengths` and build package metadata internally.

The CANN indexer/attention path flattens BSND into TND. For example, with model
tensors already on NPU:

```python
import torch

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_kl_loss import dsa_kl_loss
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention

# q_idx/k_idx: BSND indexer tensors; weights: BSN; q/kv: absorbed MLA BSND.
# q_rope/k_rope: BSND rotary states. ends_cpu: cumulative document ends.
ends_npu = ends_cpu.to(device=q.device, dtype=torch.int32)
indices, iq, ik, iw = dsa_indexer(
    q_idx, k_idx, weights, ends_npu, ends_npu, 2048, backend="cann"
)
output, maximum, total = dsa_sparse_attention(
    q, kv, q_rope, k_rope, indices, scale, ends_npu, ends_npu, backend="cann"
)
aux_loss = dsa_kl_loss(
    iq, ik, iw, q.flatten(0, 1), kv.flatten(0, 1), indices, maximum, total,
    q_rope.flatten(0, 1), k_rope.flatten(0, 1), ends_cpu, ends_cpu, scale,
    loss_coeff, backend="cann")
(task_loss(output) + aux_loss).backward()
```

Dense warm-up requires native OPP support for the dense LSE/KL kernels. The
public dense KL function returns token-mean loss, applies `loss_coefficient`
once, and propagates the upstream loss gradient only to indexer Q/K/weights.
Teacher Q/K, rotary states, statistics and LSE do not receive KL gradients.

Dense has one automatic route and no version/backend switch. `Ascend950` and
`Ascend910_95*` select the package's V2 LSE; A2/A3 select V1. The independent
`dsa_dense_v1.cpp` binding in `functional/csrc/` calls the installed V1
`aclnnDenseLightningIndexerSoftmaxLse` and fused
`aclnnDenseLightningIndexerGradKLLoss` without modifying CANN sources.
It builds on first dense use with `torch.utils.cpp_extension.load`; initialize
`ASCEND_HOME_PATH`, the CANN library path, a C++ compiler and Ninja first.
V1 max/sum becomes `LSE = max + log(sum)`; the fused KL consumes the equivalent
pair `(LSE, 1)`, so both routes share one public interface.

V3.2 dense attention uses multi-head Q/K before absorption, runs FA per packed
document, and passes FA max/sum to fused KL. It does not construct a quadratic
teacher-probability tensor in Python. CP still gathers indexer K, latent KV and
rotary K and backpropagates through these collectives. Per-document FA also
handles empty query segments at CP boundaries; it incurs one FA call per
nonempty local document. The 950 route requires hardware validation.

Attention requires shared K/V storage, shape and strides. Each candidate row
must contain `min(causal_length, TopK)` distinct valid indices, followed by `-1`
padding, as produced by the indexer. Arbitrary shorter lists and independent V
are unsupported. Attention supports first-order Q/shared-KV/rotary gradients;
higher-order gradients are unsupported. The high-level attention output retains
zero padding for the rotary width.

These interfaces accept local Torch tensors. Existing CP/TP hooks must complete
communication and provide local tensors before calling them. Single-NPU
precision and DeepSeek-V3.2 CP1/CP2 module parity were validated; full-model
multi-step convergence, performance, compression, and noncausal execution were
not validated.

Run the hardware-independent tests and the single-NPU precision tests separately:

```bash
python -m pytest tests/ut/components/functional/test_dsa_ops.py \
  tests/ut/auto_models/models/deepseek_v32/test_deepseek_v32_dsa_cp.py -q
ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 \
  python -m pytest tests/torch/components/functional/test_dsa_ops.py -m platform_ascend910b -q
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  python -m pytest tests/torch/context_parallel/test_deepseek_v32_dsa_cp.py -q
```

Select an available device for the second command. The precision suite covers
BSND/TND, FP16/BF16 indexer statistics, real TopK selection, sparse attention
forward/backward, sparse KL, and the public sparse training chain against independent
FP32 Torch references. The two-card test compares the same DeepSeek-V3.2 attention
weights and inputs at CP1 and CP2, including auxiliary KL, packed boundaries,
batch size 2, and sequence length 2080 exceeding TopK=2048;
it checks output, input-gradient, and CP-averaged parameter-gradient parity. This
DSA path has separate APIs and dependencies from the DeepSeek V4.1 fused adapters
described below.

## DeepSeek V4.1 fused operators

The V4.1 adapters use the optional `cann_ops_transformer` PyTorch extension:

| Component | Required entry points |
| --- | --- |
| Indexer | `lightning_indexer` (LI V2), `lightning_indexer_metadata` when required by the device |
| Attention | `sparse_flash_mla`, `sparse_flash_mla_grad`, `sparse_flash_mla_metadata` |
| Indexer KL | `sparse_lightning_indexer_kl_loss_grad`, `sparse_lightning_indexer_kl_loss_grad_metadata` |

### Installation

Follow the official ops-transformer guides:

1. [Build and install the native operator package](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/install/compile.md).
2. [Build and install the PyTorch extension](https://gitcode.com/cann/ops-transformer/blob/master/torch_extension/README.md).

Both the native kernels and the Python extension are required. Load the CANN and
custom-operator environment as described in the guides before training.
HyperParallel imports the standard `cann_ops_transformer` package; use the
standard wheel build (`bash build.sh --torch_extension`), without `--ops` or
`--vendor_name`, which can change the Python package name at the tested revision.

### Tested versions

| Dependency | Tested configuration |
| --- | --- |
| Hardware / CANN Toolkit | Ascend 910B3 / 9.0.0 |
| Python / PyTorch / torch-npu | 3.11 / 2.10.0 / 2.10.0 |
| ops-transformer source | `0b08075141913daeaf1e856faf7480ed393c8949` |
| Python extension | `cann_ops_transformer` 1.0.0, built from the source revision above |

The source revision identifies the tested operator interfaces; the wheel version
alone does not identify them.

### Configuration

Enable the three adapters together through the V4.1 replacement:

```yaml
plan_overrides:
  - match: "model.layers.*.self_attn"
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Attention
    replace_module:
      _target_: >-
        hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
      use_fused_ops: true
```

`false` (default) uses the Torch reference implementation without this extension.
`true` requires the fused entry points used by the model's execution path and
raises on missing entries or unsupported inputs, without automatic fallback.
The fused path supports ordinary causal CSA with compression ratio 1 or 2;
candidate-pool selection is not supported.
