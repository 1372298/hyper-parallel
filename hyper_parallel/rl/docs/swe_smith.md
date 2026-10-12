# SWE-Smith Code Agent：实现与限制

使用步骤见 [示例 README](../examples/swe_smith/README.md)。本文维护已实现的边界、
阶段验收结论和后续开发项；不保留一次性实验脚本或待执行的历史规划。

## 代码组织

与 Search-R1 一样，任务通过 `program_callable` 接入，使用公共 `SyncTrainer`，不另建训练器。

| 位置 | 职责 |
| --- | --- |
| `examples/swe_smith/agent.py` | 仓库准备、Codex 生命周期、调用预算、冻结提交、评分和逐调用轨迹 |
| `workspace.py`、`container_execution.py` | 固定问题分支、可信测试恢复和容器内会话中继 |
| `artifacts.py`、`reward.py` | 源码补丁投影、保护测试/配置、独立 pytest 评分与失败归因 |
| `prepare_data.py`、`prepare_images.py` | 子集/全量转换、任务镜像映射、架构和身份核验 |
| `infer.py`、`configs/` | 无训练的推理入口和模型配方 |
| `rl/agentic/codex/`、`rl/agentic/envs/docker_workspace.py` | 通用 Codex 协议、原始采样证据、本地/远端工作区 |
| `models/qwen3_5/adapter/selected_log_probs.py` | 模型侧动作位置概率、分块词表投影与重计算 |
| `rl/roles/`、`rl/weight_sync/`、`rl/checkpoint.py` | 训练构造、流式发布、图刷新及完整恢复 |

任务数据、业务评分和仓库编排留在示例。Search-R1 示例无需改动；共享层新增能力由显式配置开启。
共置内存释放、CPU offload 梯度同步和 checkpoint 校验属于公共修复，仍需覆盖已有配方回归。

RL 外仅保留 `core/fully_shard/hsdp_param.py::reset_sharded_param` 的 `@torch.no_grad()` 修复：
加载后的 CPU/pinned storage 刷新不应建立 autograd 链或保留旧参数 shard。
对应 core UT 检查旧 leaf 可释放、缓冲区无 `grad_fn`、Parameter 身份不变且后续梯度正确。
这属于通用 post-load 存储生命周期修复，不能用只覆盖 RL 调用点的上下文替代。

## 已验证范围

阶段结论截至 2026-10-11。训练推理机为 ARM64 Ascend，任务在远端 amd64 Docker 上执行；
A—C—B SSH 转发、候选/relay/grader 同 endpoint、隔离与冻结传输均已实际验证。

| 验证 | 结果与含义 |
| --- | --- |
| 环境与判题 | schema、boltons、oauthlib 三个镜像，九条任务均通过问题基线失败/可信修复成功对照 |
| 9B 短预算 RL | eager；每 rank 两会话、每会话两次调用；训练、发布、真实评估和完整保存通过 |
| 非零更新与恢复 | 真实训练 reward 全零时，独立实验回调明确注入组内 0/1；梯度约 10.06，参数变化约 1.91e-6，版本 0→1；新进程完整恢复后再更新到 2 |
| 流式发布 | 130 个 bucket 全部确认/释放，最多一个在途；无需聚集全模型再传输 |
| 9B 参考预算推理 | 原始权重、图执行；三个仓库九例各一次，真实 reward 为 2/9（train 2/6，validation 0/3） |
| 9B 真实混合奖励 | 同一 schema 任务先四会话得到 `[0,1,1,1]`；随后两个四会话组、40 次调用预算，共 315 次调用，得到 `[0,1,0,1]` 和 `[0,1,1,0]` |
| 长预算真实奖励训练 | 用户因耗时取消；新八会话只完成生成与独立重判，未执行 optimizer 更新、发布或恢复 |
| 27B | 模型与 RL 接入保留；本次 SWE-Smith 迁移后未重新执行完整 27B 验收 |

受控测试奖励只证明功能，不能算作真实求解成功或有效学习。2/9 是小子集的单次结果，
不是全量求解率，也未证明与 agent-lightning 相同效果。
参考预算为 100 次调用、每次最多 12,288 输出 token、上下文 81,920、timeout 5,400 秒、
temperature 1.0、thinking 关闭。短预算训练配方开启 thinking，两种设置用途不同。

九例身份和参考预算原始权重成绩如下；不混入修正前的旧结果或受控可信修复。

| 实例 ID | split | reward | 调用数 |
| --- | --- | --- | --- |
| `keleshev__schema.24a30457.func_basic__vev5jdcq` | train | 0 | 100 |
| `keleshev__schema.24a30457.func_basic__5hkx4ybr` | train | 0 | 100 |
| `keleshev__schema.24a30457.func_basic__fok6de12` | validation | 0 | 100 |
| `mahmoud__boltons.3bfcfdd0.combine_file__c5rs0ihw` | train | 0 | 100 |
| `mahmoud__boltons.3bfcfdd0.func_pm_remove_cond__u12zyhw5` | train | 0 | 44 |
| `mahmoud__boltons.3bfcfdd0.func_pm_remove_cond__xcq6er4x` | validation | 0 | 100 |
| `oauthlib__oauthlib.1fd52536.func_basic__jlkmpgou` | train | 1 | 32 |
| `oauthlib__oauthlib.1fd52536.func_basic__yd8bq2j5` | train | 1 | 100 |
| `oauthlib__oauthlib.1fd52536.func_basic__ktl2ecvh` | validation | 0 | 82 |

## 当前采样组与训练 DP 的限制

公共 Trainer 按训练 DP rank 分配提示，`rollout.num_return_sequences` 是每个提示的会话数；
组内 GRPO 优势在该 rank 内计算。GSM8K、Search-R1 和 SWE-Smith 共用这一组织方式。
目前没有将同一个提示组的不同会话分发到多个训练 DP rank 的重分配能力。

推理 TP 与该限制独立：vLLM DP1 / TP2 是一个两卡推理副本，可以生成一个提示的八个会话。
两卡不等于两个采样组，也不自动将八个会话分摊到两个训练 rank。
Qwen3.5 家族训练限定 TP=1；双卡 FSDP2 需要两个可分配提示。
用 `--repeats 2` 创建同题副本后，每 rank 四会话得到两个独立四样本组、共八会话，
不是一个全局八样本组；每提示改为八会话则全局变为十六会话。

agent-lightning 的协调器先按共同 `data_id` 聚合会话并计算优势，再由 VERL 分发训练数据；
本仓库尚未实现对应的跨训练 DP 组织方式。后续应在公共 RL 层解决，覆盖单轮和多轮任务，
保留唯一 episode 身份、真实奖励、policy version 和零损失 padding，避免只对 SWE-Smith 打补丁。

## 其他限制与后续开发

- **模型范围**：Qwen3.5-9B / Qwen3.8-27B 仅文本 GRPO、训练 TP=CP=1、
  colocated native vLLM、流式 `full_gather`、consistency off。不声明 BF16 bit-exact。
- **图范围**：固定栈支持含 `torch.compile` 的 FULL_DECODE_ONLY；9B 图推理已运行，
  本次迁移后未追加 9B 图模式的两次权重发布刷新验收，不能用纯推理替代该验收。
- **训练性能**：29,581-token 单次调用预热后前后向约 137.6 秒、峰值约 43.1 GiB。
  40 次调用八会话已产生最长 50,807-token 输入；其完整前后向尚未测量。
  多轮轨迹按每次真实上下文重新训练，推理耗时不能代表训练耗时。
  全量 log-prob 诊断、CPU offload、长上下文 GDN 和 DP padding 都有优化空间；尚未接入 FLA 训练后端。
- **长程运行**：外部清理会中断进程；支持完整训练 checkpoint 恢复，
  不支持对进行中的 Codex 会话或未完成 optimizer step 无损断点续跑。
- **数据规模**：子集和全量共用转换与执行路径，但仅实测三个仓库九任务。
  更多仓库的镜像架构、分支布局、测试依赖、pytest 入口及资源需求仍须检查。
- **源码范围**：冻结补丁仅支持普通 Python / Python stub；现有测试、配置、hook 和执行权限受保护。
  其他语言或需要修改构建配置的任务不在当前验收范围内。
- **求解行为**：部分失败会话持续重复命令。当前依靠调用数、输出和时间预算终止，
  没有宣称与 Mini-SWE-Agent 相同的提示、工具、压缩策略或求解效果。
- **设备环境**：本机曾因旧控制器设备绑定返回 HAL 87（RESOURCE_OCCUPIED）；
  只看 `npu-smi` 无计算进程不足以确认设备已释放。结束实验应停止自有空闲控制器，不占位。

后续优先顺序：公共采样组与训练 DP 解耦、训练性能与长输入显存验证、
真实混合奖励更新/恢复、9B 图发布刷新及 27B 回归，再扩大仓库与数据规模。
新实验重新检查设备、镜像空间、输出/checkpoint 路径和预算，不沿用已取消的长训练规划。

## 回归入口

CPU 任务合同在 `tests/ut/rl/agentic/swe_smith/`；通用协议与工作区分别在
`tests/ut/rl/agentic/codex/`、`tests/ut/rl/agentic/test_docker_workspace.py`。
模型数学合同在 `tests/ut/auto_models/models/qwen3_5/`，
训练、发布和恢复沿用 `tests/ut/rl/` 对应模块。
Search-R1、旧文本网关、Hermes 证据和完整 checkpoint 回归不能因精简而删除。
CPU 通过不等同于新增硬件组合、长预算学习或全数据集通过。
