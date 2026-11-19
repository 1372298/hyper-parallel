# DeepSeek Sparse Attention

提供 DeepSeek 稀疏注意力及 KV AllGather 上下文并行，支持转换已有模块或从 `config` 直接构造：

- `DeepseekV32SparseAttention`：用于 Transformers DeepSeek-V3.2。
- `DeepseekSparseAttention`：用于已有 `linear_qkv`、`linear_qb`、`linear_kvb` 和扁平 Indexer 的兼容模块，仅支持 sparse 阶段。

## YAML 配置

下面是 TP1、CP2 的配置片段，合入已有模型训练配方；完整示例见
[V3.2 并行配方](../../../../models/deepseek_v32/recipes/train_parallel_offline.yaml)。

```yaml
accelerator:
  tp_size: 1
  cp_size: 2
  sequence_parallel: false

plan_overrides:
  - match: "*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.DeepseekV32SparseAttention
        cp_strategy: kv_allgather
        overlap: true
        dense_warmup: null
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `accelerator.tp_size` | `1` | 张量并行度，按主 attention head 切分；head 数须能被 TP 度整除。 |
| `accelerator.cp_size` | `1` | 上下文并行度；1 使用本地 attention，大于 1 沿序列切分并交换 KV。 |
| `cp_strategy` | `kv_allgather` | CP 通信策略，目前仅支持 KV AllGather，保留本地 query 并收集所需 key bank。 |
| `overlap` | `true` | CP 时先提交各路 key bank 的异步 gather，再等待；`false` 使用同步调度。 |
| `dense_warmup` | `null` | V3.2 训练阶段：`null` 跟随模块的 `dsa_dense_warm_up`，`true` 固定 dense warm-up，`false` 固定 sparse。 |

后三个参数放在 `replace.implementation` 下，与 `_target_` 同级。扁平布局使用
`hyper_parallel.components.modules.DeepseekSparseAttention`，且 `dense_warmup` 只能为 `null` 或 `false`。
`*.self_attn` 匹配要替换的 attention 模块，无需添加 `when: cp` 或单独替换 `*.self_attn.forward`；
框架自动传入源模块与并行上下文。通用规则见 [Attention YAML 配置](../readme.md)。

## 模型配置

以下字段属于模型 `config`，不要直接放到 `replace.implementation` 下：

| 字段 | 含义 |
| --- | --- |
| `index_topk` | 每个 query 选取的稀疏 key 数量。 |
| `dsa_loss_coeff` | Indexer KL 辅助损失系数；默认 0，训练时非零且未冻结才启用。Loss 缩放由框架统一处理。 |
| `dsa_dense_warm_up` | V3.2 的训练阶段标志；仅在 `dense_warmup: null` 时生效。 |
| `freeze_dsa` / `freeze_DSA` | 关闭 Indexer 辅助训练；沿用源模型的字段名，默认 `false`。 |

Python 中也可复用已有模块或直接传入包含模型维度、Indexer 和 RoPE 设置的配置对象：

```python
from hyper_parallel.components.modules import DeepseekV32SparseAttention

attention = DeepseekV32SparseAttention(module=source_attention)
# 或：attention = DeepseekV32SparseAttention(config=config)
```

`module` 与 `config` 二选一；从 `config` 构造不会加载预训练权重。

## 支持范围

- 运行需要 Ascend NPU、CANN 和 `cann_ops_transformer`，见 [算子说明](../../../functional/README.md)。
- 支持因果序列和显式 packed 文档边界；CP 要求序列等长连续切分，且每篇 packed 文档在每个 CP rank 上都有 query token。
- 不支持任意 padding mask、attention dropout、KV cache、PP 或 sequence parallelism。
- 启用 sparse Indexer KL 时，当前 CANN 9.2 算子要求 TP 切分后的本地 query head 数为 32、64 或 128。
- FSDP 包裹完整 attention 或外层 decoder block，不单独包裹 KV / 输出投影。
- 扁平布局的 MOME / parameter-sink 扩展仅支持 TP1、CP1，默认关闭；V3.2 不支持这两个扩展。

## V3.2 Indexer Share

V3.2 local/CP member 支持配置 `dsa_indexer_types`、`dsa_index_topk_freq`、
`dsa_index_skip_topk_offset` 和 `dsa_index_share_loss`。Shared 层无 Indexer 参数，
复用前方 Full 层的选择；served 监督按真实组大小取均值。
CP 下 Full gather 三路 key bank，Shared 仅 gather 本层的 main KV/RoPE key，
复用 leader 已 gather 的 student 状态和本 rank query 的 Top-K。
共享不额外限制已有 TP/DP/EP/FSDP；PP/SP 保留原有限制，不支持 generic 扁平 family 和 Top-K offload。
模型需安装每次 forward 的共享状态入口，并使用非重入 checkpoint；
详细配置与接入代码见 [Indexer Share 设计](../../../../../docs/design/dsa_indexer_share.md)。
