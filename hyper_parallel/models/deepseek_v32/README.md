# DeepSeek-V3.2

**English** | [简体中文](README.zh.md)

HyperParallel trains the native Transformers `DeepseekV32ForCausalLM`, using fused sparse attention,
RMSNorm and grouped experts. It supports FSDP, TP, CP and EP; PP and sequence parallelism are not supported.

## Requirements

- Transformers with `deepseek_v32` support (`transformers>=5.11.0` in the repository requirements).
- Ascend NPU, CANN and `cann_ops_transformer`; see the [operator guide](../../components/functional/README.md).
- A compatible checkpoint and tokenizer. BF16 and released FP8 checkpoints can be loaded; FP8 weights are
  dequantized during loading. MTP weights are ignored.

## YAML configuration

The [eight-device recipe](recipes/train_parallel_offline.yaml) uses TP2 × CP2 × DP2 and EP4 × EDP2:

```yaml
accelerator:
  tp_size: 2
  cp_size: 2
  ep_size: 4
  pp_size: 1
  sequence_parallel: false
  loss_parallel: true

fsdp_config:
  dp_shard_size: 2
  edp_shard_size: 2
```

Merge this topology into a complete recipe; keep its model, data, optimizer and module-replacement settings.

| Parameter | Meaning |
| --- | --- |
| `accelerator.tp_size` | Tensor-parallel degree; splits the main attention heads. |
| `accelerator.cp_size` | Context-parallel degree; splits the sequence. Values above 1 enable KV AllGather attention. |
| `accelerator.ep_size` | Expert-parallel degree for MoE layers. |
| `accelerator.loss_parallel` | Computes the language-model loss with a TP-sharded vocabulary. |
| `fsdp_config.dp_shard_size` | FSDP sharding degree for non-expert parameters. |
| `fsdp_config.edp_shard_size` | FSDP sharding degree for expert parameters. |

The recipes replace `*.self_attn` with `hyper_parallel.components.modules.DeepseekV32SparseAttention`.
Its main options are `cp_strategy: kv_allgather`, `overlap: true` and `dense_warmup: null`;
see the [attention configuration guide](../../components/modules/attention/deepseek_sparse_attention/readme.md)
for their meanings and Indexer training settings.

Keep `pp_size: 1` and `sequence_parallel: false`. For this model, TP > 1 requires EP > 1.
The world size is TP × CP × DP; `n_routed_experts` must be divisible by EP, and
`global_batch_size` by `micro_batch_size × DP`. When changing the model depth, update the recipe's
`when: ep` matching rules to cover only MoE layers (`i >= first_k_dense_replace`).

## Run training

Run the following commands from the repository root. Trailing arguments override YAML fields,
for example `--training.train_iters=200`.

### Checkpoint recipe

[train.yaml](recipes/train.yaml) defaults to one process. The selected checkpoint and optimizer states
must fit the device memory; the released full model is very large.

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_lm.py hyper_parallel/models/deepseek_v32/recipes/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2 \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2
```

### Eight-device crop recipe

The parallel recipe uses an 8-layer crop: 3 dense layers, 5 MoE layers and 16 routed experts.
Create a randomly initialized BF16 checkpoint from the released config and tokenizer:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

source, target = "/path/to/DeepSeek-V3.2", "/path/to/DeepSeek-V3.2-8layer-crop"
released = AutoConfig.from_pretrained(source)
config = AutoConfig.from_pretrained(
    source, num_hidden_layers=8, n_routed_experts=16,
    layer_types=released.layer_types[:8], mlp_layer_types=released.mlp_layer_types[:8],
)
del config.quantization_config
torch.manual_seed(1234)
AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(target)
AutoTokenizer.from_pretrained(source).save_pretrained(target)
```

Prepare an Indexed Dataset (`.bin`/`.idx`) with the same tokenizer, following the
[offline data guide](../../../docs/guide/data/offline_preparation_guide.md).
Pass the checkpoint directory and dataset prefix without the extension:

```bash
bash hyper_parallel/models/deepseek_v32/recipes/run_parallel_offline.sh \
  /path/to/DeepSeek-V3.2-8layer-crop /path/to/dataset_text_document
```

Logs, data cache and checkpoints use `output/deepseek_v32`; override it with `OUTPUT_DIR`.

## Attention constraints

Use causal sequences or explicit packed-document boundaries, without arbitrary padding masks, attention dropout
or KV cache. Every packed document must have query tokens on every CP rank.
With sparse Indexer KL enabled, the current CANN 9.2 operator requires 32, 64 or 128 query heads **per TP rank**.
See the [attention guide](../../components/modules/attention/deepseek_sparse_attention/readme.md#支持范围)
for the remaining constraints.

## Indexer Share

Set `dsa_indexer_types` or `dsa_index_topk_freq` under the Trainer YAML's
`model.config_overrides` to reuse a
Full layer's Top-K in following Shared layers, which own no Indexer parameters.
Both `served` and `leader` supervision and dense warm-up support local and KV
AllGather CP execution. Sharing adds no restrictions to the existing TP, DP, EP
or FSDP support; the model's PP/SP limitations still apply. Use non-reentrant
checkpointing. The Trainer's
`activation_checkpoint.mode: full` selects model-owned non-reentrant regions and
installs the per-forward carrier automatically. See the
[design and runnable validation examples](../../../docs/design/dsa_indexer_share.md)
for the explicit Python installation path, CP2/CP4 precision checks and checkpoint limitations.

The current HF config loader only overrides existing attributes: seed the custom
sharing keys in the local model's `config.json` before applying YAML overrides.
See design section 2 for preparation, field definitions and an effective-layout
check. Merge the [YAML fragment](recipes/indexer_share.fragment.yaml) into an
existing recipe; it is not a standalone training configuration.
