# Hyper-RL 运行镜像

Qwen3 的普通 RL、Codex Agent 和 DeepSeek Harness 共用 v0.22 统一镜像，现有启动脚本和系统测试默认使用此版本。
Qwen3.5-9B / Qwen3.8-27B 使用下文单独发布的 v0.23 Code Agent 镜像。

## 下载与校验

镜像公开可读，无需登录：

```bash
published_image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-unified-arm64
docker pull "${published_image}"
image=hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64
docker tag "${published_image}" "${image}"
docker image inspect --format '{{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${image}"
```

| 项目            | 值                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| 平台            | `linux/arm64`，Ascend NPU                                                                                         |
| Manifest digest | `sha256:450b32a4a1d7818d80b832335665a7d8bece5b91827f8ee80a9f62609e939944`                                         |
| 展开大小        | 约 18.9 GB                                                                                                          |
| 训练与推理      | Torch`2.10.0+cpu`、torch-npu `2.10.0`、Transformers `5.5.4`、vLLM `0.22.1+empty`、vLLM-Ascend `0.22.1rc1` |
| 数值一致性依赖  | `batch_invariant_ops==1.0.0`、`flash-attn-npu==0.2.0b1`                                                         |
| Agent           | Codex CLI`0.152.1`；DeepSeek Harness SDK/runtime `0.1.1rc1`                                                     |

Codex 使用独立二进制包，无需 Node.js。DeepSeek 指 Agent harness，不表示新增模型支持。
云端标签保留旧环境备份；本地缓存清理后，执行上面的拉取和重新标记即可恢复原启动脚本。
检查 Agent 依赖：

```bash
docker run --rm "${image}" /bin/bash -lc '
set -e
codex --version
python -c "import deepseek_harness; from importlib.metadata import version; print(version(\"deepseek-harness-sdk\"))"
'
```

## 单轮 code 的 SandboxFusion 镜像

沙箱服务与训练容器分开运行，不需要分配 NPU。固定的 Python-only ARM64 镜像已上传到同一 SWR 组织；
镜像公开可读，无需登录；已使用空凭证 Docker 配置验证匿名拉取及镜像身份。

```bash
sandbox_image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v1-arm64
docker pull "${sandbox_image}"
docker image inspect --format '{{.Id}} {{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${sandbox_image}"
```

| 项目 | 值 |
| --- | --- |
| 平台 | `linux/arm64`，Python-only 沙箱 |
| Manifest digest | `sha256:2ceb45f3e0a86ced184339573ab2be8ebf7d450b9cb741da405cbab11f462a73` |
| Image ID | `sha256:1ac76247ee612e1ac4a07ae10192056f061d93438f7c16578c7b169938b24131` |
| 展开大小 | 约 18.54 GB；继承训练基础层，新增层展开大小约 113 MB |

上传已复用 23 个不同的已有层，仅上传 14 个新增层。拉取时 Docker 同样复用本地已有层，无需手工拆分或拼接；
没有缓存基础层的机器仍需下载缺失层。展开大小不等于压缩后的网络流量。
`agentic.code.runtime_version` 沿用上面的 Image ID，不能与 Manifest digest 混用。
启动方式、cgroup 与隔离要求见 [code 示例](../examples/code/README.md#sandboxfusion-部署)。
普通 CPU UT 不需要此镜像；真实沙箱 ST 和 code 训练才需要启动服务。

## 宿主要求

以下要求针对训练容器；沙箱的隔离权限与 cgroup 条件见上述 code 部署说明。

- Linux ARM64、Docker、兼容的 Ascend NPU driver；`npu-smi info` 正常。
- 按运行场景准备空闲、健康的 NPU，模型和数据目录，以及至少 30 GB Docker 可用空间。
- 使用匹配的仓库源码；代码、模型、数据和结果由启动脚本挂载，不打入此镜像。

容器入口加载 CANN，启动脚本调用 `docker/install_runtime.sh` 安装 RL 包并注册 vLLM 插件。
手动 `docker exec` 或覆盖 `PYTHONPATH` 后，须在启动模型的同一 shell 中重新执行
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`，保留 CANN Python 路径，确保 `import acl` 可用。
`docker/patches/vllm-dp-coordinator-timeout.patch` 将 vLLM DP Coordinator 的启动等待时间从 30 秒延长到 120 秒；
GSM8K 的 TP 与一致性启动脚本会在容器中应用它。更新 vLLM 后应先确认补丁仍适用。
具体运行步骤见 [Hyper-RL README](../README.md)，系统测试见 [ST 指南](../README.md#系统测试)。
需要自定义镜像时，保留各启动脚本的 `HYPER_*_IMAGE` 和测试的 `RL_ST_*_IMAGE` 覆盖方式。

## 仓库型 Code Agent 与 Qwen3.5 家族

已发布的 ARM64 controller 镜像公开可读，无需登录：

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64
docker pull "${image}"
docker image inspect --format '{{.Id}} {{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${image}"
```

| 项目 | 值 |
| --- | --- |
| 平台 | `linux/arm64`，Ascend 训练/推理 controller |
| Manifest digest | `sha256:0d1e8af0cbc8eb558497958b827260fa6d9a635fc20e4ec74916587b0348731d` |
| Image ID | `sha256:9d796c354631648532be7c5c775860dab3653359fd249423f074abf35341a2e1` |
| 大小 | 展开约 18.9 GB，压缩层合计约 7.25 GB；实际传输量取决于已有层 |
| Agent | Codex CLI `0.152.1`、DeepSeek Harness SDK `0.1.1rc1` |

镜像包含 RL 运行依赖；仓库源码、模型和数据由运行时挂载提供，并按当前 checkout 安装 editable 主包及 RL 插件。
远端 amd64 任务机的 CLI/仓库镜像仍按 [SWE-Smith 说明](../examples/swe_smith/README.md#镜像准备与身份) 单独准备。
该标签面向 Qwen3.5 家族接入，不改变上面的 Qwen3 默认统一镜像及严格一致性验收范围。

9B、27B 的 controller 基于官方 `quay.io/ascend/vllm-ascend:v0.23.0.post1`，
`Dockerfile.qwen3_8` 是仅加入固定 Codex CLI 的最小叠层入口；发布版本还包含已准备的 RL 依赖，复用官方推理栈。
已验收运行环境为 Torch 2.10.0+cpu、torch-npu 2.10.0.post4、CANN 9.1、
Transformers 5.5.4、vLLM 0.23.0、vLLM-Ascend 0.23.0.post1、Codex CLI 0.152.1；
SWE-Smith 不依赖 SWE-bench evaluator。原有 Qwen3 严格一致性环境保持独立。

本地环境以已发布标签为入口，不依赖开发阶段的 `agent`、`consistency`、`rl` 或 `swe4.1.0` 镜像。
旧 SWE-bench 的 pytest-10051、pytest-10081 任务镜像及其派生镜像不属于当前 SWE-Smith 的依赖，
可随旧实验清理；当前任务机上的 SWE-Smith 镜像按实际选中的数据准备并保留。
清理镜像前先核对运行中和已停止容器的引用；共享基础层由 Docker 管理，标签大小不能直接相加为可释放空间。

[SWE-Smith Dockerfile](../examples/swe_smith/Dockerfile) 为选中仓库的固定任务镜像加入 CLI 和独立 Python stdlib。
任务镜像与 CLI 必须匹配宿主架构；`prepare_images.py` 默认不拉取缺失镜像。可信 controller 持有 Docker 客户端及 socket，候选与 grader
无网络、无 NPU、无 socket，使用同一固定镜像。宿主 Docker 客户端可能还需 `libltdl.so.7`。

源码/wheel 放在 `/home/xhy/Project-hw/reference`，原始模型放在 `/home/xhy/Project-hw/models`。
先核对磁盘空间、依赖版本及 editable 导入路径，再构建或启动；代理使用宿主 8990，
国内模型下载显式禁用代理。任务数据直接放在 `/home/xhy/Project-hw/data`。运行配置中的模型、逐任务镜像清单、摘要和输出挂载按实际环境填写。

非特权 NPU controller 映射所需 `davinci`、`davinci_manager`、`devmm_svm` 和 `hisi_hdc`，
只读挂载驱动；用户须有设备及 socket 的属组权限、匹配 UID 的用户记录和可写家目录/缓存。
同时核对 CANN 和算子目录读取权限。`ASCEND_RT_VISIBLE_DEVICES` 使用容器内枚举编号；
本宿主只映射物理卡 4–7 时，容器中为 0–3。启动前用同一用户核验小算子及实际占卡。

候选和 grader 的 `pids_limit` 使用 1024，为 CLI 与 pytest 线程留出空间。
纯 CPU controller 不分配 NPU，设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
SWE 准备、训练/恢复命令、图配置和验收边界统一见
[Code Agent 示例](../examples/swe_smith/README.md)。
