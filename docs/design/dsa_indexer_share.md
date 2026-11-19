# DeepSeek-V3.2 DSA Indexer Share 设计

## 1. 目标和范围

在重构后的 `DeepseekV32SparseAttention` 上增加跨层 Top-K 共享。一个组的首层为 Full（leader），
后续为 Shared。Full 层拥有 Indexer 参数并计算选择结果；Shared 层没有 Indexer 参数，
保持自己的 Q/K/V、RoPE 和输出投影，只消费 Full 层的选择结果。

实现覆盖 V3.2 的 native config 构造和 HF module 转换、sparse 与 dense warm-up、
leader/served 两种辅助监督、local 和 KV AllGather CP、非重入激活重算，以及单卡同布局 checkpoint 恢复。
共享不额外限制原 DSA 支持的 TP、DP、EP 和 FSDP；它只改变 Indexer 所有权及 CP 中需要交换的 key bank。
PP/SP 保留原 DSA 的不支持边界。本轮没有实现 generic 扁平 DSA、V4.1、MTP、KV cache、
Top-K offload、编译导出或其他 CP 算法。

参考本地 MindFormers PyNative 的 `IndexShareGroup` / `resolve_dsa_index_share_leaders()`，
以及 Transformers demo 的 `GlmMoeDsaAttention`，但不复制其模型 forward 或三值返回 ABI。
参考代码的实际行为优先于分析文档中的概括。

## 2. 配置和布局

### 2.1 YAML 配置入口与必要前置

共享字段属于 **模型 config**。在 Trainer YAML 中放入 `model.config_overrides`，
不要放在 `accelerator`、`activation_checkpoint` 或 attention replacement 的 `implementation` 中。
后者只接受 `cp_strategy`、`overlap`、`dense_warmup` 等已有构造选项。

当前加载路径为 `Trainer YAML → model.config_overrides → _load_pretrained_config()`
`→ AutoConfig.from_pretrained() → resolve_index_share_plan()`。
**当前 HF 配置加载器只覆盖已存在的属性：若原始 config.json 没有这些自定义键，
仅在 YAML 中新增它们会被忽略。** 已用不含共享键的原生 `DeepseekV32Config` 复现：
YAML 的 freq=3 未进入 config，最后仍解析为全 Full。使用下面的 YAML 前，须先完成一次字段准备。

在用于训练的模型配置副本 `config.json` 中补齐以下默认字段；已有字段保留原值：

```json
{
  "dsa_indexer_types": null,
  "dsa_index_topk_freq": 1,
  "dsa_index_skip_topk_offset": 0,
  "dsa_index_share_loss": "served",
  "dsa_index_share_topk_offload": false,
  "dsa_loss_coeff": 0.0,
  "dsa_dense_warm_up": false,
  "freeze_DSA": false
}
```

这是合入原 config 的字段片段，**不能用它替换完整 config.json**。
原有 `model_type`、`num_hidden_layers`、维度、MoE、RoPE 等配置必须保留。
若原文件仅有旧字段 `freeze_dsa`，新增 `freeze_DSA` 时沿用其值；大写字段在 V3.2 中优先。
字段准备只建立可覆盖的属性，之后每个实验的布局和模式均在 YAML 中选择。
未来若修复通用 loader 对新增 `config_overrides` 属性的支持，可取消此前置；本文按当前实现说明。

### 2.2 推荐配置：按频率分组

将下面片段合入已有 V3.2 Trainer 配方，例如
[离线并行配方](../../hyper_parallel/models/deepseek_v32/recipes/train_parallel_offline.yaml)。
保留原 `model._target_`、模型路径、dataset、optimizer、并行和 FSDP 配置；
在已有 `plan_overrides` 中修改 `*.self_attn` 对应条目，保留 RMSNorm、experts 和 EP 的其他条目。
也可复制 [YAML 配置片段](../../hyper_parallel/models/deepseek_v32/recipes/indexer_share.fragment.yaml)。
该片段不是独立训练配方，当前 CLI 不会自动把多个 YAML 合并。

```yaml
model:
  config_overrides:
    dsa_indexer_types: null          # 清除 config.json 中可能已有的显式布局
    dsa_index_topk_freq: 3           # 每组最多三层：Full + Shared + Shared
    dsa_index_skip_topk_offset: 0    # 从零起始第 0 层开始分组
    dsa_index_share_loss: served    # 组内每层监督 leader，按实际组大小平均
    dsa_index_share_topk_offload: false
    dsa_loss_coeff: 0.1             # 示例辅助损失系数；缺省为 0，不训练 Indexer
    dsa_dense_warm_up: false        # sparse 阶段
    freeze_DSA: false               # 允许 Indexer 辅助梯度
    use_cache: false

activation_checkpoint:
  mode: full                       # V3.2 adapter 接入非重入重算和每次调用的共享状态

plan_overrides:
  - match: "*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.DeepseekV32SparseAttention
        cp_strategy: kv_allgather
        overlap: true
        dense_warmup: null         # 跟随模型 config 的 dsa_dense_warm_up
```

例如原模型 `num_hidden_layers=8` 时，得到 `F S S F S S F S`，
leader 层号为 `[0, 0, 0, 3, 3, 3, 6, 6]`；最后一组只有两层，served 除以 2，不除以 3。
这里不需要为了共享修改 `num_hidden_layers`，否则还需重新核对原模型的 MoE 布局和权重形状。

### 2.3 显式布局与关闭共享

不规则分组使用显式列表。以下例子要求原模型恰好有八层，用它替换上面的布局字段，
其余配置保持不变：

```yaml
model:
  config_overrides:
    dsa_indexer_types: [full, shared, shared, full, shared, full, shared, shared]
    dsa_index_share_loss: served
```

对应三个组 `{0,1,2}`、`{3,4}`、`{5,6,7}`，各组独立共享。
`dsa_indexer_types` 非 null 时优先于 freq/offset，这两个字段不再参与布局计算。
字符串 `FSS`、字段 `indexer_types` 或列表中的缩写 `F/S` 都不是本接口。

关闭共享需同时解除显式布局并恢复频率 1；仅设置 freq=1 不能覆盖已有显式列表：

```yaml
model:
  config_overrides:
    dsa_indexer_types: null
    dsa_index_topk_freq: 1
    dsa_index_skip_topk_offset: 0
```

这会恢复“每层都有自己的 Indexer”；`dsa_loss_coeff: 0` 只关闭辅助监督，**不会关闭共享**。
切换角色会改变参数集合，不能把 Shared checkpoint 当作全 Full checkpoint 直接 strict 加载。

### 2.4 共享字段逐项说明

以下默认值指实现没有读到字段时的缺省值；原 config.json 已有的值不会自动恢复成默认值。
YAML 的 `null` 对应 Python `None`，布尔值使用 `true/false`，枚举使用表中的小写拼写。

| `model.config_overrides` 字段 | 默认值 | 类型和合法值 | 含义及相互作用 |
| --- | --- | --- | --- |
| `dsa_indexer_types` | `null` | 非空列表或 null；列表长度等于 `num_hidden_layers`，元素仅 `full/shared`，首项必须 full | 显式指定每层角色。Full 新建组并拥有 Indexer；Shared 复用最近的前方 Full。全 full 列表不启用共享。非 null 时覆盖 freq/offset。 |
| `dsa_index_topk_freq` | `1` | 整数 ≥1，不接受 bool/字符串 | 没有显式列表时，控制 Full 层的间隔，而非 Top-K 个数或训练 step 频率。1 为每层 Full；3 通常得到 FSS。尾组可不足该长度。 |
| `dsa_index_skip_topk_offset` | `0` | 整数 ≥0，不接受 bool/字符串 | 没有显式列表时的零起始分组起点。第 0 到 offset 层均为 Full，offset 层开始后续周期；不是“跳过前几层 attention”。 |
| `dsa_index_share_loss` | `served` | `served` / `leader` | served 用组内每层自己的 teacher 监督同一 leader student，按真实组大小取均值；leader 只用 Full 层监督。两者都让 Shared 没有 Indexer、共享同一 Top-K。served 不减少 KL kernel 调用次数。 |
| `dsa_index_share_topk_offload` | `false` | 布尔值；当前只支持 false | Top-K offload 尚未实现；启用共享时设 true 会报错，不能作为已有优化开关使用。 |

频率布局的 Full 判定为 `max(layer_idx - offset, 0) % freq == 0`。
例如 N=6、freq=3、offset=1 得到 `F F S S F S`，即组 `{0}`、`{1,2,3}`、`{4,5}`；
offset ≥ N−1 时所有层均 Full，实际上没有共享。
本实现接受不足频率大小的尾组，不使用 GLM 公式中的 `+1`。

`resolve_index_share_plan()` 是构造与模型接入共同使用的布局来源，返回不可变的
`IndexSharePlan(leaders, loss_mode)`。native 构造开启共享时必须给每层 config 设置
正确的零起始 `layer_idx`；HF 转换自动读取源模块层号，用户不在全局 YAML 中设置该值。
布局和监督模式是构建期配置，修改后需重建模型，不能直接改变已训练模块的角色。

### 2.5 相关字段与优先级

| 配置位置/字段 | 默认或来源 | 作用 |
| --- | --- | --- |
| `model.config_overrides.dsa_loss_coeff` | DSA 缺省 `0.0` | Indexer KL 总体系数。served 再除以组大小；leader 不除。0 时仍共享选择，但不注入辅助梯度。`0.1` 是上述示例值，不是缺省值。 |
| `model.config_overrides.freeze_DSA` | 缺省 false；兼容 `freeze_dsa` | true 关闭 Indexer 辅助监督，不关闭共享，不自动冻结主模型。V3.2 优先读大写字段；避免两个字段取相反值。 |
| `model.config_overrides.dsa_dense_warm_up` | 缺省 false | true 为 dense warm-up，共享 leader 的 student 投影，不产生 Top-K；false 为 sparse。不是按 step 自动切换的调度器。 |
| `model.config_overrides.index_topk` | 继承原模型配置 | 每个 query 选取的 key 数量 K，与共享层间隔无关。组内保持一致，且须满足本机 kernel 约束；已验证 sparse S1056/K1024。 |
| `model.config_overrides.num_hidden_layers` | 继承原模型配置 | 全局层数，用于检查显式列表长度和计算尾组。为开启共享无需修改它。 |
| `model.config_overrides.use_cache` | 继承 HF 配置 | 示例显式设 false；当前 DSA 不支持 KV cache。 |
| `plan_overrides[].replace.implementation.dense_warmup` | `null` | null 跟随模型的 warm-up 选项；true/false 强制对应阶段，优先于 config 的 `dsa_dense_warm_up`。组内各层必须一致。 |
| 同位置 `cp_strategy` | `kv_allgather` | 当前唯一支持的 CP 算法，保留本地 query、交换 key bank。 |
| 同位置 `overlap` | `true` | CP 时先发出本层多路异步 gather，再 wait；false 同步执行。不会改变共享布局或损失定义。 |
| `activation_checkpoint.mode` | 本示例 `full` | V3.2 adapter 使用非重入区域并安装共享 context；不需要在 YAML 中传 `index_share_context` 或 `use_reentrant`。本轮验收采用 full。 |
| `accelerator.cp_size` | 原训练配方 | CP1 使用 local member，CP>1 使用 CP member。不是共享开关。TP/DP/EP/FSDP 无需为了共享新增特殊参数。 |

例如切换 dense warm-up，可将 `model.config_overrides.dsa_dense_warm_up` 设 true，
并保持 replacement 的 `dense_warmup: null`。若后者仍为 false，则仍会走 sparse。
选择 leader 监督只需修改 `dsa_index_share_loss: leader`，不会增加/减少 Shared 层数。

已有可运行的 CP 配方直接合入共享字段即可。若单独搭建两卡示例，拓扑可使用
TP1/CP2/EP1/PP1、sequence_parallel=false、FSDP/EDP shard=2；完整模型、数据和 batch
参数仍由原训练配方提供。PP/SP 保持基线限制，TP>1 时还需遵守 V3.2 原有 EP 约束。
开启 sparse KL 时，当前 CANN 9.2 要求 TP 切分后的 main query heads 为 32、64 或 128。

### 2.6 启动前检查是否真正生效

下面只读取 YAML 和本地 config，不加载模型权重、不创建分布式组、不运行训练。
在已经配置好本 checkout 的环境中，令 `TRAIN_YAML` 指向合并后的完整训练 YAML：

```bash
python - "$TRAIN_YAML" <<'PY'
import sys
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.models._transformers.config_resolver import get_hf_config
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import resolve_index_share_plan

training = parse_training_args([sys.argv[1]]).to_dict()
model = training["model"]
overrides = model.get("config_overrides") or {}
config = get_hf_config(
    model["pretrained_model_name_or_path"],
    model.get("attn_implementation", "eager"),
    model.get("torch_dtype", "auto"),
    local_files_only=True,
    **overrides,
)
keys = (
    "dsa_indexer_types", "dsa_index_topk_freq", "dsa_index_skip_topk_offset",
    "dsa_index_share_loss", "dsa_index_share_topk_offload", "dsa_loss_coeff",
    "dsa_dense_warm_up", "freeze_DSA",
)
for key in keys:
    if key in overrides and (not hasattr(config, key) or getattr(config, key) != overrides[key]):
        raise ValueError(f"YAML 字段未生效：{key}；先检查 config.json 是否有该属性")
plan = resolve_index_share_plan(config)
print("enabled:", plan.enabled, "loss:", plan.loss_mode)
print("roles:", ["full" if i == leader else "shared" for i, leader in enumerate(plan.leaders)])
print("leaders:", list(plan.leaders))
PY
```

八层频率示例应输出 `enabled: True`、`loss: served` 和
`leaders: [0, 0, 0, 3, 3, 3, 6, 6]`；若仍是 `[0,1,2,3,4,5,6,7]`，不要把它当作共享训练。
字段解析检查不代替训练或 checkpoint 兼容验收；原全 Full checkpoint 文件直接迁移为共享布局
尚未验证，随机初始化的验证配方使用 `model.load_base_model: false`。

## 3. 参数所有权和兼容性

- Full 层保留 `indexer.wq_b/wk/k_norm/weights_proj`。
- Shared 层 `indexer=None`，其 `named_parameters()`、optimizer 与 `state_dict()` 都不含 Indexer。
- 从 config 构造 Shared 时不分配 Indexer；从 HF module 转换时不保留源 Indexer。
- 主 Attention 参数命名、融合 QKV 权重转换与二值返回 `(output, None)` 不变。
- 参数分片描述只枚举实际存在的参数，因此无需给 Shared 创建占位参数。
- native meta 构造后的 `to_empty()` / `reset_parameters()` 支持 Shared 层。
- 同共享布局的 state_dict 可以 strict 加载，恢复时模型必须使用相同布局。
- 将原全 Full HF 模型对象转换为共享模型，会丢弃 Shared 的源 Indexer；这不是无损转换。
  本轮未验收原全 Full checkpoint 文件经通用加载器直接迁移、HF 原格式导出或 DCP 分布式恢复。
  不应使用 `strict=False` 静默掩盖缺失的 Full 权重。

## 4. 每次 forward 的状态

`IndexShareContext.groups` 按 leader 全局层号保存 `IndexShareState`：

- leader 的 BSND/TN 变换前 Indexer query/key/weights，保留 autograd 连接；
- sparse Top-K（kernel TND int32 索引，detach）；dense 阶段为 None；
- 当前 dense/sparse 阶段与 Top-K 配置，供消费者检查。

共享状态不放在 Attention 模块属性、全局变量或线程局部缓存中。
HF decoder 模型的 forward pre-hook 为每次调用创建全新的 context，HF 原生 decoder loop
通过现有 `**kwargs` 将同一对象传至各层。各组独立存放，连续 `FSSS` 不会在第一层 Shared 后丢失选择。

Full 在 `run_attention` 选择一次并发布；Shared 在投影阶段跳过 Indexer，读取对应 leader。
缺 context、缺 leader，以及阶段、Top-K、batch/sequence 或 device 不匹配都会报错。
模型安装函数还检查整个 decoder stack 的布局与层号一致。
所有共享层必须处理相同 token 布局、packed 边界与位置坐标；手工调用方负责遵守这一合同，
本版不支持在同一组中改变 packed 文档边界。

直接调用层时，每个 forward 自行创建 context，按层顺序传递；不得跨 batch 复用：

```python
from hyper_parallel.components.modules.attention.deepseek_sparse_attention import IndexShareContext

context = IndexShareContext()
for attention in attentions:
    hidden, _ = attention(hidden, index_share_context=context)
```

## 5. 数学与梯度

令组首 Indexer 的 student 分布为 P，组内第 i 层 detached teacher 为 Q_i，组大小为 g：

- `leader`：`L_aux = coefficient * KL(Q_leader || P)`，仅 Full 层附加辅助梯度。
- `served`：`L_aux = coefficient / g * sum_i KL(Q_i || P)`，每层独立调用现有 KL kernel。

served 不会减少 KL kernel 的调用次数；它省去 Shared 的 Indexer 投影和 Top-K 选择。
每项 loss 在本层区域内按真实组大小缩放，再用现有 `aux_loss_auto_scale` 附着到输出。
没有跨层累加器，也没有改全局梯度倍率、loss scale 或公共 KL 数学。
Indexer 输入仍 detach，teacher 仍 detach；共享投影不能 detach，否则后续层监督无法训练 leader。

在 dense warm-up 中不执行 Top-K，共享的是 leader 的 Indexer 投影。
每个被监督层使用自己的 dense Q/K 与 softmax 统计。dense 的归一化仍按实际组大小，不能除以 1。
`freeze_dsa`、`dsa_loss_coeff=0` 和 eval 沿用原辅助监督开关。
日志里的主任务 Loss 不等于单独的 Indexer KL 数值，因为公共辅助接口只注入梯度。

## 6. 激活重算与模型入口

只支持 **非重入** checkpoint。重入 checkpoint 的原 forward 在 no_grad 下执行，
不适合把可微投影通过 Python carrier 供后续层复用；共享辅助训练在该情况下明确报错。
eval + no_grad 可以正常执行。

非重入 checkpoint 闭包保留同一次 forward 的 context。Shared replay 可读取原图的 leader 投影，
leader replay 重新执行自己的投影与选择。状态随 forward/backward 生命周期释放。
CPU 验证覆盖多个 group、两个 forward 交错后逆序 backward；NPU 验证覆盖真实模型的重算梯度一致性。
本轮不保证重入重算、编译重排、pipeline 调度或任意自定义执行顺序。

模型接入有两个自动路径：

1. meta 构建时，V3.2 materialization adapter 安装 carrier；
2. 普通 Trainer 的 `activation_checkpoint.mode: full` 使用 V3.2 的 `RecomputePolicy`，
   按 decoder 层包装非重入区域，并通过 `on_applied` 安装 carrier。

这也意味着 V3.2 的默认 full checkpoint 改由模型声明的非重入区域实现，即使关闭共享也是如此。
未修改公共 checkpoint 或 FSDP 算法。关闭重算的非 meta 构建、selective 等其他入口，
应显式调用 `install_indexer_share(model)`；本版仅验收 full 重算 Trainer 配方。

普通 HF Python 使用：

```python
from transformers import DeepseekV32ForCausalLM
from hyper_parallel.components.modules import DeepseekV32SparseAttention
from hyper_parallel.models.deepseek_v32.adapter.indexer_share import install_indexer_share

config.dsa_indexer_types = ["full", "shared", "shared"]  # 对应 num_hidden_layers=3
config.dsa_index_share_loss = "served"
config.dsa_loss_coeff = 0.1
config.use_cache = False
model = DeepseekV32ForCausalLM(config)
for layer in model.model.layers:
    layer.self_attn = DeepseekV32SparseAttention(module=layer.self_attn)
install_indexer_share(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

## 7. CP：从 MindFormers 映射到阶段接口

参考 MF `mindformers/pynative/distributed/style.py` 的 `HPDSAContextParallelAdapter`：
Shared 层跳过不存在的 Indexer CP 边界，但每层 main attention 仍安装 CP；
`build_hp_dsa_cp_style()` 让 sparse Indexer、attention 和 KL 保持 Colossal 的 sequence-shard/KV-gather 布局。
MF 的 `pynative/transformers/experimental_attention_variant/dsa_attention.py` 则负责发布/复用 Top-K，
并在 served 模式使用每层自己的 teacher 监督同一 leader student。

HP 将相同的所有权和坐标规则放入已有 `prepare_attention → exchange_attention → run_attention`：

| 层角色 | 本层投影 | CP gather 顺序 | 共享状态 |
| --- | --- | --- | --- |
| Full | Indexer + main attention | Indexer key → latent KV → RoPE key | exchange 后发布可微 student 状态和本 rank query 的 Top-K |
| Shared | 仅 main attention | latent KV → RoPE key | 读取 leader 已 gather 的 student 状态，省去 Indexer gather/选择 |

query 始终是本 CP rank 的连续片段。Top-K 的 key 索引指向 gather 后的 bank；它不是 rank 内局部 key 索引，
Shared 不增加 rank offset，也不再次 gather Top-K。每层自己的 main KV 使用原 `kv_prefix_length`
截取因果前缀，组内必须使用相同 CP mesh、query 切分、position 和 packed 文档边界。
这些跨层条件由统一 decoder 执行保证；手工逐层调用必须遵守，不能只满足张量 shape。

同步路径逐路 gather；异步路径先 launch 本层拥有的所有 bank，再逐一 wait，之后才消费结果。
角色在构建时确定且所有 rank 一致，通信顺序不会依赖本地 Top-K 或数据内容。
Shared 的 `index_states=()` 是“本层没有 Indexer”的合同，不代表缺少 main attention 输入。

served 梯度仍按真实组大小平均。Shared KL 到 leader student 的图保留不变；多个 consumer 的
梯度先累积到同一可微 gather 图，再由原 gather adjoint SUM 返回源 shard。
不手工乘除 CP/TP degree，也不增加第二套 FSDP 梯度归约。
TP 的主头分片、Indexer 的复制布局和辅助梯度倍率沿用原 family；EP/DP 不读取共享 context。

MF 旧边界实现有 `DSASequenceReplicateCache`，需要在 Shared attention 后释放其持有的引用。
HP 的 `SparseAttentionInputs` 和异步 receipt 本来就属于一次调用，没有模块级 replicate cache；
因此没有移植缓存清理 hook，也不新增持久 KV 缓存。context 只保留跨层实际使用的 leader student 图。
MF dense teacher 的 Ulysses/Hybrid 属于额外算法，本实现只沿用 HP 已有的 `kv_allgather`。

## 8. 文件职责

| 文件 | 职责 |
| --- | --- |
| `components/modules/attention/deepseek_sparse_attention/indexer_share.py` | 布局校验与 per-invocation 数据合同 |
| `deepseek_sparse_attention.py` | 参数裁剪、共享选择、sparse/dense 辅助系数 |
| `deepseek_sparse_attention_family.py` | 新 stage 参数、并行/通用 family 边界 |
| `cp_deepseek_sparse_attention.py` | 按 Full/Shared 角色交换三路/两路 key bank，保留原 gather adjoint |
| `models/deepseek_v32/adapter/indexer_share.py` | HF kwargs carrier 与非重入重算策略 |
| `adapter/materialization.py`、`adapter/registration.py` | 注册到已有模型生命周期 |
| `tests/ut/components/test_dsa_indexer_share.py` | CPU 合同、均值梯度、生命周期与重算回归 |
| `examples/torch/deepseek_v32/indexer_share_smoke.py` | 真实单卡 HF 训练、重算、保存恢复验收 |
| `examples/torch/deepseek_v32/indexer_share_cp_precision.py` | CP2/CP4 输出、全参数梯度和 Top-K 对照 |

以上路径相对于 `hyper_parallel/`，tests 和 examples 除外。

## 9. 单卡复跑与验收边界

安装本 checkout 的 editable 包，准备与现有 DSA 一致的 Torch/torch_npu、CANN 和
`cann_ops_transformer` 环境。仓库根目录执行：

```bash
python -m pytest -q tests/ut/components/test_dsa_indexer_share.py
python examples/torch/deepseek_v32/indexer_share_smoke.py --mode served --output /tmp/sparse_served.json
python examples/torch/deepseek_v32/indexer_share_smoke.py --mode leader --output /tmp/sparse_leader.json
python examples/torch/deepseek_v32/indexer_share_smoke.py --warmup --sequence 64 --output /tmp/dense_served.json
python examples/torch/deepseek_v32/indexer_share_smoke.py --warmup --mode leader --sequence 64 --output /tmp/dense_leader.json
```

默认 sparse 使用 S=1056/K=1024，确有超过 K 的 causal query；辅助 KL 的 K 满足本环境 kernel 约束。
脚本不下载模型，创建三层 Dense HF 小模型；输出 JSON 和同名 `.pt` 保存点。
验收包括：首轮 Top-K 调用数、三步有限且非零 Indexer 梯度、参数更新、
重算与不重算的全部梯度严格一致、同布局保存恢复后的下一步参数严格一致。
CPU 的 KL mock 检查组合系数和 detach 合同，不替代原生 KL kernel 数学精度验证。

本轮不声明性能提升、相对 MF/GLM 全模型数值等价、完整大模型收敛或生产长序列验收。
共享会改变被服务层的离散选择，本来也不要求与未共享模型逐元素相同。

## 10. CP 精度复跑

在满足 CANN DSA 算子要求的 NPU 环境，仓库根目录执行（两条命令顺序运行）：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/torch/deepseek_v32/indexer_share_cp_precision.py --output /tmp/share_cp2
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  examples/torch/deepseek_v32/indexer_share_cp_precision.py --output /tmp/share_cp4
```

每 rank 运行九例：FSS 的 sparse/dense × served/leader ×（同步无重算、异步非重入重算），
另加 FSFS sparse served 异步重算。权重和含真实 RoPE 的输入 hash 完全相同，reference 使用 CP1。
比较完整输出、input gradient、所有 main/Indexer 参数梯度及 global mean loss；
每项须 finite、max_abs ≤ 0.008 且 relative L2 ≤ 0.05，有效 causal Top-K 集合须完全相同。
模块比较在 CP 各 rank 的参数梯度求和后进行；这只是验证夹具，生产梯度仍由原 Trainer/FSDP 管理。

验证包含 S=1056、K=1024 的真实 sparse 选择和 S=64 的 dense warm-up，
不覆盖 packed、多节点、不等长切分、所有重算/Validate 组合或 DCP 续训。
多卡精度是同一共享模型在 CP1/CPn 下的自一致性，不是与 MF 或无共享模型的全架构等价证明。
