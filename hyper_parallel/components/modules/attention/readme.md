# Attention 模块的 YAML 配置

本目录按注意力算法组织实现。同一目录可以包含不同模型的参数布局和调用约定；
YAML 通过公共 family 工厂选择实现，工厂再根据当前任务的 CP 拓扑选择本地或并行成员。
DeepSeek 两种参数布局的使用方法见 [DeepSeek Sparse Attention](deepseek_sparse_attention/readme.md)。
下面的实现参数与支持范围针对这两种 SparseAttention；其他 attention family 的参数见
[各算法使用文档](../README.md#attention-families)。

## 最小替换配置

以下是 Trainer recipe 的一个片段，应合并到包含模型、数据、优化器等配置的完整 recipe 中：

```yaml
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

这里替换的是完整 attention 模块。`DeepseekV32SparseAttention` 对应 DeepSeek-V3.2 的
`q_a_proj` / `kv_a_proj_with_mqa` 参数布局；已有 `linear_qkv` / `linear_qb` / `linear_kvb`
等扁平布局的源模块使用 `hyper_parallel.components.modules.DeepseekSparseAttention`。
两者不能只凭类名互换，具体差异见算法目录的说明。

完整的 V3.2 recipe 可参考 [单进程配置](../../../models/deepseek_v32/recipes/train.yaml)
和 [并行配置](../../../models/deepseek_v32/recipes/train_parallel_offline.yaml)。

## 替换规则参数

| 参数位置 | 类型或取值 | 作用 |
| --- | --- | --- |
| `plan_overrides[].match` | 字符串或字符串列表 | 匹配模型中的模块全名；支持 glob。`*.self_attn` 可以匹配 `model.layers.0.self_attn`。 |
| `replace.kind` | `module` | 使用完整模块替换。`method` 是另一种方法替换机制，不用于此 family 配置。 |
| `replace.implementation._target_` | 可导入的公共工厂路径 | 选择参数布局和算法 family；无需指定具体 CP 类。 |
| `replace.source_type` | 可选的源类型导入路径 | 除名称匹配外，进一步限制可替换的源模块类型。 |
| `replace.exact_type` | `false`，默认值 | 为 `true` 时仅匹配 `source_type` 本身，不接受其子类；必须同时配置 `source_type`。 |
| `plan_overrides[].when` | 通常省略 | 完整模块替换只允许省略或使用 `low_precision` 条件；不能写 `cp` 或 `ep`。 |

不要给 attention family 的替换规则加 `when: cp`。CP1 也需要本地 family 成员；
是否选择 CP 成员由 `accelerator.cp_size` 决定。也不要再对同一模块增加
`*.self_attn.forward` 的旧 CP 方法替换。

同一条 `kind: module` 规则不能混用 `region_dispatch` 或 `params`、`in_src`、`in_dst`、
`out_src`、`out_dst`、`tp_divide_attrs` 等布局覆盖字段。需要单独覆盖布局时使用另一条规则，
且最终布局仍须满足 family 声明的参数和输入输出约定。

## Attention 实现参数

以下参数位于 `replace.implementation` 中，与 `_target_` 同级：

| 参数 | 默认值 | 含义与取值 |
| --- | --- | --- |
| `cp_strategy` | `kv_allgather` | 当前只支持此策略。查询保留在本 CP rank，交换 Indexer key、latent KV 和 rotary key，再使用因果可见前缀。 |
| `overlap` | `true` | CP 时先提交各个 key bank 的异步 gather，再等待结果；`false` 使用同步调度。它不改变计算结果，也不保证所有通信均被计算覆盖。 |
| `dense_warmup` | `null` | V3.2 在每次前向时读取模块的 `dsa_dense_warm_up` 标志；`true` 固定使用 dense warm-up，`false` 固定使用 sparse 阶段。扁平布局的 `DeepseekSparseAttention` 只支持 sparse 阶段。 |

`dsa_loss_coeff`、`freeze_dsa` / `freeze_DSA`、维度、RoPE、MOME 等是模型 config 或源模块的属性，
不是上述 family 工厂的任意关键字参数。不要把这些属性直接放到 `implementation` 下；
使用所选模型自己的 config 配置入口，或者在 Python 原生构造时传入 `config` 对象。
这些沿用的外部字段名保留原拼写。

## 并行参数放在哪里

并行度属于整个任务的拓扑，放在 recipe 的 `accelerator` 中：

```yaml
accelerator:
  tp_size: 2
  cp_size: 2
  ep_size: 1
  pp_size: 1
  sequence_parallel: false
  loss_parallel: true
```

| 参数 | 默认值 | 与 attention 的关系 |
| --- | --- | --- |
| `tp_size` | `1` | 大于 1 时按完整 attention head 切分；主 attention head 数必须能被 TP 度整除。Indexer 保持 TP 复制。 |
| `cp_size` | `1` | `1` 选择本地成员，大于 1 选择 KV-AllGather 成员。 |
| `ep_size` | `1` | 控制任务中的专家并行，不是 attention 工厂的参数。 |
| `pp_size` | `1` | 当前这两个 attention family 只支持 `1`。 |
| `sequence_parallel` | `false` | 当前这两个 attention family 不支持开启。 |
| `loss_parallel` | `false` | 整个训练任务的损失并行选项；Loss 缩放与参数梯度归约统一由框架管理，attention 不设置 Indexer 专用缩放。 |

`DeepseekSparseAttention` 的 MOME 和 parameter-sink 扩展只支持 TP1、CP1；
关闭这些扩展后才能启用上述 TP / CP 路径。V3.2 成员不支持这两个扩展。

FSDP 的分片、混合精度、预取和 reshard 选项属于 `fsdp_config`，不放到 `implementation` 中。
attention 内部直接读取 KV / 输出投影的普通 Linear 参数，FSDP 应管理完整 attention 模块或其外层
decoder block；不要把这两个投影叶子层作为独立的 FSDP 单元。family 布局准备和并行绑定应先于
FSDP 安装，由标准 Trainer 构建流程统一完成。

## 框架注入参数与 Python 构造

替换现有模型时，框架自动向工厂传入：

- `module`：当前匹配到的源模块，包含已经加载的参数和 `config`。
- `module_fqn`：源模块在模型中的完整名称。
- `context`：本次任务的逻辑并行拓扑。

这三个字段不能手填进 YAML 的 `implementation`。替换已有模块时也不要再传 `config`：
工厂要求 `module` 与 `config` 二选一。

Python 中可以直接使用 `Factory(config=config)` 创建参数，也可以使用 `Factory(module=source)`
复用兼容源模块。`mesh` 是可选的、已经创建好的 `MeshContext` 对象；传入后复用其拓扑完成布局准备和绑定，
不是在 YAML 中填写一个 mesh 名称或再创建一套通信组。单独提供 `context` 仅用于逻辑成员选择，
不能代替运行前的物理 mesh 绑定。具体示例见 [算法使用文档](deepseek_sparse_attention/readme.md)。

参数解析的实现见 [Trainer 配置](../../../trainer/config/parallelism.py)、
[替换执行器](../../../models/replacement.py) 和 [module family](../module_family.py)。
