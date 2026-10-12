# Code Agent：SWE-Smith

本例沿用 Search-R1 的 `program_callable` 接入方式，由 `agent.py::SWESmithProgram` 组织
仓库准备、Codex 执行、冻结提交、独立评分和训练轨迹构造；训练、发布、评估与恢复使用主线 `SyncTrainer`。
数据和问题分支语义参考 [agent-lightning SWE-Smith](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith)。
本文是使用入口，按环境、数据、镜像、推理和训练顺序组织。
实现边界、已验证结果及后续开发项见 [开发说明](../../docs/swe_smith.md)。

## 环境准备

训练推理使用 Ascend controller，任务工作区使用 Docker；两者可在同机，也可通过 SSH 使用远端 Docker。
先按 [运行环境](../../docker/README.md#仓库型-code-agent-与-qwen35-家族) 准备固定版本依赖、
设备权限、模型、Docker 客户端和可写结果/checkpoint 挂载。本文命令在 controller 的仓库根目录执行。
`/models`、`/data`、`/results`、`/checkpoints` 分别指已配置的模型、原始数据、准备结果和检查点挂载。

```bash
source /usr/local/Ascend/cann/set_env.sh
python -m pip install --no-deps --no-build-isolation -e . -e hyper_parallel/rl
export PYTHONPATH="$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import acl, hyper_parallel; print(hyper_parallel.__file__)'
codex --version
```

保留 CANN 原有 `PYTHONPATH`；editable 导入必须指向当前 checkout，Codex 为 `0.152.1`。
远端 controller 还需已验证 host key、有效 SSH-agent 和匹配 UID 的系统用户记录；
先用 `ssh -p PORT USER@HOST true` 与 `docker --host ssh://USER@HOST:PORT info` 检查连接。
任务机无需模型、NPU 或训练代码；仅需 Docker、已准备的任务镜像和 SSH 访问。

## 模型配方

| 模型                     | 配方                         | 训练 / rollout          |
| ------------------------ | ---------------------------- | ----------------------- |
| Qwen3.5-9B，首轮实验模型 | `configs/qwen3_5_9b.yaml`  | FSDP2 / native TP2、DP1 |
| Qwen3.8-27B              | `configs/qwen3_8_27b.yaml` | FSDP4 / native TP4、DP1 |

两者使用公共 `qwen3_5` text AutoModel，纯文本 GRPO、训练 TP=CP=1、CPU offload、
activation checkpoint、consistency off，开启 thinking，默认 eager。
无需安装 SWE-bench evaluator。

## 准备子集或全量数据

使用 agent-lightning 给出的 [筛选数据包](https://drive.google.com/file/d/1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1/view)，
其中两个 JSONL 为 `train_dataset_mixed.jsonl` 和 `val_dataset_filtered.jsonl`。
本机源文件直接放在 `/home/xhy/Project-hw/data`，controller 挂载为 `/data`；
准备结果与实验输出放在独立结果挂载 `/results`。
JSONL 提供问题描述、实例、镜像和测试节点；仓库源码、依赖和测试代码由对应镜像提供。

从仓库根目录运行。先编辑 `/data/swe_smith_selected_ids.txt`，每行一个选中的实例 ID，
包括需要的训练和验证实例，然后执行：

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.swe_smith.prepare_data \
  --train-jsonl /data/train_dataset_mixed.jsonl \
  --validation-jsonl /data/val_dataset_filtered.jsonl \
  --instance-ids-file /data/swe_smith_selected_ids.txt \
  --output /results/swe_smith
```

输出 `train.parquet`、`validation.parquet`、`instances.jsonl`、私有 `instances/` 和 `selection.json`。
`instances.jsonl` 与 `instances/` 只供 controller 使用，不进入候选容器。
训练数据只含公开问题和可信任务引用；每条引用通过 SHA256 校验自己的私有记录。

省略 `--instance-ids-file` 和 `--max-instances` 即处理完整输入 split，后续执行路径保持相同。
显式过滤后再应用数量上限；未知 ID、重复记录和 split 重叠会报错。
训练和验证独立选择，不自动复制数据。新输出目录必须为空，准备成功以 `selection.json` 为标志。
少量训练实例不足以满足分布式 batch 时，可以显式使用 `--repeats 2` 创建采样副本；
副本有独立 prompt identity，不计作新的任务，也不证明独立泛化表现。

### 当前采样组与训练 DP 的限制

`rollout.num_return_sequences` 是每个提示的会话数，GRPO 采样组由所属训练 DP rank 管理。
双卡 FSDP2 需要至少两个可分配提示；一个任务可显式 `--repeats 2`，但得到两个独立采样组。
例如每 rank 四会话时全局八会话，不是一个全局八样本组。
推理 DP1 / TP2 只是一个两卡服务，不会改变训练分组。
跨训练 DP 的组内会话分发尚未实现，详见 [限制及后续开发](../../docs/swe_smith.md#当前采样组与训练-dp-的限制)。

## 镜像准备与身份

先查看选中实例所需镜像，以下命令只读本地 metadata，不下载或构建：

```bash
python -m examples.swe_smith.prepare_images \
  --instances /results/swe_smith/instances.jsonl \
  --output /results/swe_smith/images.json --plan-only
```

准备入口按真实 `image_name` 去重，保留每个实例自己的映射。
缺失镜像默认报错，只有显式 `--pull` 才允许拉取；拉取前检查目标架构。
构建复用本地 CLI 运行镜像，使用本例 `Dockerfile` 离线加入 CLI 与 relay 的独立 Python stdlib。
官方 Conda 镜像默认使用已有 `testbed` 环境；已有 `origin` 问题分支转为本地引用，保留原 commit，
不在候选执行期间联网抓取分支或安装依赖。无此 Conda 路径的原生镜像继续使用其原有 Python。
CLI 镜像与任务镜像必须匹配目标 Docker daemon 的架构。首次构建须用 `--cli-image` 显式指定
本地 CLI 运行镜像，包含 `/opt/codex` 和 `/usr/local/python3.12.13`；不使用机器特定的默认镜像 ID。

已有本地镜像时，执行准备：

```bash
python -m examples.swe_smith.prepare_images \
  --instances /results/swe_smith/instances.jsonl \
  --output /results/swe_smith/images.json --cli-image LOCAL_CLI_IMAGE
```

缺失任务镜像时，确认空间和平台后显式加 `--pull`；没有该参数不会下载。
通过 `--overrides /data/image_overrides.json`
可将数据中的镜像名称映射到经独立验收的本地等价任务镜像。
原生重建的来源和环境差异须记录并验证，不能只替换镜像标签。
平台须以实际 manifest 为准；ARM64 原生重建需独立验收。

override 可为镜像字符串，或含 `image` 和 `test_path_map` 的对象。
个别输入使用相对于项目子目录的测试路径；须在可信镜像描述中显式映射到仓库内的规范路径。
该映射参与镜像清单摘要校验，保留原始数据记录；不自动猜测或删除 `../` 前缀。

`images.json` 记录固定任务镜像 ID、原始来源、架构、Python 路径、测试路径映射与 CLI 身份。
逐镜像保存准备进度，重新运行复用成功结果，只有全部完成才写入 `complete=true`。
准备完成后将输出的 `images_sha256` 填入配方的 `agentic.codex.task.images_sha256`。
同时填写 `data.train_path/test_path`、`agentic.codex.task.data_dir/images_path`、
模型路径、`session_root` 和 `train.checkpoint.output_dir`；使用独立结果目录避免覆盖旧实验。
运行时核验清单身份，按当前实例选择镜像；候选与 grader 使用同一个固定 ID。

实际占用 NPU 前还须核验实例分支，并完成问题基线失败、可信修复通过的环境对照。
镜像平台、分支、依赖或评估入口缺失时明确报错，不静默跳过数据。

## 远端沙盒和参考预算推理

远端执行时，在 `agentic.codex.workspace` 设置 `docker_host: ssh://user@host:port`。
候选、relay 和 grader 均使用该 endpoint；SSH 客户端、已验证的 host key 和 SSH-agent 只在可信控制器可见。
连接错误直接传播，不切换本地 Docker。准备镜像时也使用相同目标：

```bash
python -m examples.swe_smith.prepare_images \
  --instances /results/swe_smith/instances.jsonl \
  --output /results/swe_smith/images.json \
  --docker-host ssh://user@host:port --cli-image TARGET_AMD64_CLI_IMAGE
```

推理入口 `infer.py` 复用原生 vLLM、CodexRuntime、SWESmithProgram 和独立评分，不创建 Actor 或优化器。
使用 `configs/qwen3_5_9b_infer.yaml`，填入与训练相同的数据/镜像清单、摘要和远端 endpoint 后执行：

```bash
torchrun --standalone --nproc_per_node=2 -m examples.swe_smith.infer \
  hyper_parallel/rl/examples/swe_smith/configs/qwen3_5_9b_infer.yaml \
  --dataset /results/swe_smith/train.parquet \
  --dataset /results/swe_smith/validation.parquet \
  --output /results/reference-infer.jsonl
```

可用重复的 `--instance-id ID` 选择单例或子集；不提供则处理输入中的全部记录。
每条记录运行一次独立会话，结果逐条写入 JSONL；基础设施错误保留会话证据并报错。
输出目录沿用事先批准的运行目录，完整 token、实际请求、补丁和测试状态保存在 session 目录。

参考预算对应本机 Agent Lightning `d3819953` 的实际 Job 和 `_query`：Qwen3.5-9B、
100 次模型调用、每次最多 12,288 输出 token、服务上下文 81,920、5,400 秒、温度 1.0、thinking 关闭。
图模式采用已固定栈的 FULL_DECODE_ONLY；工具协议保留 Codex/Qwen3Coder。
`max_observation_chars=6000` 仅对模型收到的工具观察保留首尾并按字符截断，原始工具证据不变。
自动压缩阈值为 65,536，为单次输出保留上下文空间；压缩请求也计入 100 次调用。
CLI 自己的 10,000 token 输出上限仍存在；两种 Agent 的观察包装和压缩策略有差异，不能声称提示完全一致。
本轮使用配方 seed 派生的固定会话种子；参考 `_query` 未显式传 seed，不声称逐 token 复现其轨迹。
独立 grader timeout 600 秒，资源 1 CPU / 6 GiB，真实 reward 不注入人工差异。
若全错，检查实际请求、预算结束、工具执行、冻结补丁和测试评分，外部 kill 不计作求解失败。

## 任务与评分合同

候选检出实例的问题分支，核验“Bug Patch → Remove F2P Tests”的结构并解析固定 commit。
随后移除 Git 历史，保留 controller 捕获的可信问题基线。
容器无网络、Docker socket 和 NPU；管理凭据不进入候选，relay 仅绑定其会话。

Responses 中同一回复的说明文字和函数调用合并为一个 assistant Chat 消息，避免增加虚假的回合结束标记。
来源校验允许解码串与 engine text 仅相差一个精确的末尾 EOS；原始采样 token 保留不变。
未闭合 XML 仍属于不可执行的模型格式错误，其他来源不一致继续拒绝会话。
预算终止响应确认交付后 relay 停止接收新请求，随后由控制器关闭；CLI 重试不能重新打开已结束的会话。
Codex 正常结束或经网关证实调用预算耗尽后，停止候选并导出真实代码。
`artifacts.py` 在 controller 上生成补丁，允许普通 Python / Python stub 源码修改，
保护已有测试、项目配置、特殊文件和执行权限；不信任候选 Git 元数据。
新增非 Python 的复现输入、备份等中间文件不进入提交补丁；新增 Python 测试/hook 仍按受保护路径检查。
最终评分版本为 `swe-smith-pytest-v3`，历史冻结工件可以重评，无需重新生成模型轨迹。
工件 byte/file 上限可在任务配置中调整，不由首个实例决定整个数据集的容量。

grader 从新容器启动，校验相同问题/测试 commit，应用冻结补丁并恢复可信测试。
默认 `f2p_only=true`：全部 F2P 与相同测试文件内的 P2P；设为 false 可检查完整 P2P。
所有选中测试均有可信状态且全部为 PASSED/XFAIL 才得到奖励 1；完整判题失败为 0。
已独立确认的提交或语法错误可形成零奖励。候选源码导致测试无法收集时，
通过 pytest 钩子核验异常来自修改的源码，并确认原问题基线能够收集全部选中测试，才记为零奖励；
保留真实异常和已报告状态，不为未运行节点伪造 FAILED 状态。
采样、容器、传输、无法归因的收集失败和其他不完整判题错误仍拒绝更新。

## 短预算训练与恢复

填写实际模型路径、准备结果目录、镜像清单摘要及独立 checkpoint 挂载后运行：

```bash
torchrun --standalone --nproc_per_node=2 hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/swe_smith/configs/qwen3_5_9b.yaml
```

27B 使用另一份配方和四个训练进程，当前优先验证 9B。
配方每 rank 一个提示、每提示两个会话，每会话最多两次模型调用、单次最多 512 output token。
两个会话用于 GRPO 组内比较。上下文为服务 16384 / Codex 14336，训练一步并评估、保存。
真实奖励相同时可能零梯度；短预算验证功能，不保证真实求解。
人工奖励差异仅用于受控验收测试，默认评分仍使用真实测试结果。

`train.response_mini_batch_size` 的单位是逐调用训练行；它不等于会话数。
增大调用预算时须同时检查该参数和显存、耗时；不能只放大 `max_turns` 后假定更新成本不变。
若要求真实非零学习验收，启用 `train.learning_gate` 并使用真实混合奖励；默认示例不注入人工奖励。

检查点写到 `/checkpoints`，完整 checkpoint 空间须独立确认；本轮 9B 单份约 51 GiB。
恢复设置 `train.checkpoint.load_path` 指向完整 `step_1`，`train.max_steps=2`，使用新的输出目录。
恢复后先发布恢复版本，再继续采样和更新。数据、镜像清单及评估范围须保持原实验身份。

## 图执行与验证范围

固定 vLLM 0.23.0 / vLLM-Ascend 0.23.0.post1、rollout DP1 支持显式 FULL_DECODE_ONLY：
设置 `enforce_eager=false`，在 `rollout.vllm` 下加入以下配置，并保持 `max_num_seqs=4`。

```yaml
compilation_config:
  mode: 3
  cudagraph_mode: FULL_DECODE_ONLY
  cudagraph_capture_sizes: [1, 2, 4]
  max_cudagraph_capture_size: 4
```

该模式包含 `torch.compile` 和图重放。流式 full-gather 完整发布并唤醒 KV 后，
在开放采样前重新捕获 decode 图，刷新失败不得提交版本。
不为本例修改上游算子，也不声明两种模型 bit-exact。

CPU 合同按所有权分布在 `tests/ut/rl/agentic/swe_smith/`、`agentic/codex/`、
`agentic/test_docker_workspace.py` 及现有模型、Trainer、checkpoint 和 weight-sync 测试中。
子集/全量转换、多实例/多镜像、评分和兼容性通过 CPU 检查后，仍需真实镜像和 NPU 验收。
9B 短预算远端 eager RL 闭环已验收；参考预算图推理九例真实求解 2/9。
长预算真实混合奖励已经出现，但该批训练因耗时取消，不算有效学习验收。
9B 图发布刷新复验、27B 本轮回归和全数据集仍未完成；
详细结果、分组限制和后续优先项见 [开发说明](../../docs/swe_smith.md)。
