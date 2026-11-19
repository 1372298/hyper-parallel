# DeepSeek-V3.2

[English](README.md) | **简体中文**

HyperParallel 训练 Transformers 原生的 `DeepseekV32ForCausalLM`，使用融合 sparse attention、
RMSNorm 和 grouped experts。支持 FSDP、TP、CP、EP，不支持 PP 和 sequence parallelism。

## 环境要求

- 包含 `deepseek_v32` 的 Transformers，仓库依赖要求 `transformers>=5.11.0`。
- 昇腾 NPU、CANN 和 `cann_ops_transformer`，见 [算子说明](../../components/functional/README.md)。
- 兼容的 checkpoint 与 tokenizer。支持 BF16 和官方 FP8 checkpoint；FP8 权重在加载时反量化，MTP 权重被忽略。

## YAML 配置

[八卡配方](recipes/train_parallel_offline.yaml) 使用 TP2 × CP2 × DP2、EP4 × EDP2：

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

将这个拓扑片段合入完整配方，保留模型、数据、优化器和模块替换配置。

| 参数 | 含义 |
| --- | --- |
| `accelerator.tp_size` | 张量并行度，切分主 attention head。 |
| `accelerator.cp_size` | 上下文并行度，切分序列；大于 1 时启用 KV AllGather attention。 |
| `accelerator.ep_size` | MoE 层的专家并行度。 |
| `accelerator.loss_parallel` | 在词表按 TP 切分的情况下计算语言模型 Loss。 |
| `fsdp_config.dp_shard_size` | 非专家参数的 FSDP 分片度。 |
| `fsdp_config.edp_shard_size` | 专家参数的 FSDP 分片度。 |

配方将 `*.self_attn` 替换为 `hyper_parallel.components.modules.DeepseekV32SparseAttention`。
主要选项为 `cp_strategy: kv_allgather`、`overlap: true`、`dense_warmup: null`；
参数含义与 Indexer 训练设置见 [Attention 配置说明](../../components/modules/attention/deepseek_sparse_attention/readme.md)。

保持 `pp_size: 1`、`sequence_parallel: false`。本模型 TP > 1 时要求 EP > 1。
总卡数为 TP × CP × DP；`n_routed_experts` 须能被 EP 整除，
`global_batch_size` 须能被 `micro_batch_size × DP` 整除。修改模型层数时，
同步调整配方中 `when: ep` 的匹配规则，仅覆盖 MoE 层（`i >= first_k_dense_replace`）。

## 启动训练

以下命令在仓库根目录执行。末尾追加的参数覆盖 YAML 字段，例如 `--training.train_iters=200`。

### Checkpoint 配方

[train.yaml](recipes/train.yaml) 默认使用单进程。所选 checkpoint 和优化器状态需要能放入设备内存；
官方完整模型的内存需求很大。

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_lm.py hyper_parallel/models/deepseek_v32/recipes/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2 \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2
```

### 八卡裁剪模型配方

并行配方使用 8 层裁剪模型：3 个 dense 层、5 个 MoE 层、16 个路由专家。
从官方 config 和 tokenizer 创建随机初始化的 BF16 checkpoint：

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

使用相同 tokenizer 准备 Indexed Dataset（`.bin`/`.idx`），见
[离线数据准备指南](../../../docs/guide/data/offline_preparation_guide.zh.md)。
传入 checkpoint 目录和不带扩展名的数据前缀：

```bash
bash hyper_parallel/models/deepseek_v32/recipes/run_parallel_offline.sh \
  /path/to/DeepSeek-V3.2-8layer-crop /path/to/dataset_text_document
```

日志、数据缓存和 checkpoint 默认写入 `output/deepseek_v32`，可通过 `OUTPUT_DIR` 修改。

## Attention 使用限制

输入使用因果序列或显式 packed 文档边界，不支持任意 padding mask、attention dropout 和 KV cache。
每篇 packed 文档须在每个 CP rank 上都有 query token。
启用 sparse Indexer KL 时，当前 CANN 9.2 算子要求 **TP 切分后每个 rank** 的 query head 数为 32、64 或 128。
其他限制见 [Attention 支持范围](../../components/modules/attention/deepseek_sparse_attention/readme.md#支持范围)。
