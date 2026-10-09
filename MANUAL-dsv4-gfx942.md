# DeepSeek-V4-Flash FP8 强化学习训练操作手册（2 台 × 8 卡 gfx942）

本手册说明如何用 `Dockerfile.gfx942-dsv4` 构建的镜像，在两台 MI300 系列（gfx942）服务器上跑起 DeepSeek-V4-Flash FP8 的完整 RL 配方。
训练参数全部由 launcher 自动选择，启动时只需要给出机器相关的信息：IP、网卡、模型和数据路径。

---

## 0. 适用范围与验证情况

| 项目 | 要求 / 说明 |
|---|---|
| GPU | 2 台 × 8 张 gfx942。已验证：MI308X（192 GB）。MI300X（192 GB）会自动选同一套内存配置。MI325X（256 GB）会自动选 288 GB 卡的配置，**未验证** |
| 主机内存 | 验证环境每台 3 TB。训练稳定期每台峰值约占 2.2 TB（最低还剩 824 GB） |
| 网络 | 两台之间需要一张互通的网卡。已验证用 25G TCP（RCCL 走 socket）。RDMA 可选，**未验证** |
| 共享存储 | 模型和数据必须在两台上以**相同路径**可见（NFS / Lustre 等）。FP8 checkpoint 274 GB，`torch_dist` 530 GB |
| 本地磁盘 | 每台留 ≥ 200 GB 给 Ray 日志和临时文件。保存训练 checkpoint 每次约 4 TB（默认不存，见第 5 节） |
| 训练配方 | DAPO-Math-17k，每个 rollout 32 道题 × 8 个采样，回答上限 8192 token，thinking 模式。每 20 步在 AIME-2024 上评测一次 |

**验证情况：**
- 本配方用"upstream miles 2026-09-23 版本 + 同样的修复"在 2 × 8 MI308X 上连续训练了 27 步以上，全部 rank 有效，没有失败。
- 训推一致性：`train_rollout_kl` 0.0065–0.0070，`logprob_abs_diff` 0.038–0.041。
- AIME-2024（4096 token 上限）：第 0 步 0.392，第 20 步 0.475。
- 本镜像把 miles 换成了 2026-09-26 的 upstream（加同样的修复），代码层面的检查都通过了，渲染出的训练参数跟上面这次跑次逐项一致。
- 用本 Dockerfile 构建的镜像 `miles-dsv4-gfx942:0927` 已在 2 × 8 MI308X 上端到端跑了 step 0–5，没有失败：第 0 步 AIME 0.429，`train_rollout_kl` 0.0062–0.0068，`logprob_abs_diff` 0.037–0.040。自己构建的镜像请按第 6 节的指标做一次验收。

---

## 1. 构建镜像

构建机需要能访问 Docker Hub、GitHub 和 PyPI，Docker 需要开启 BuildKit（Docker 23 及以上默认开启）。不需要 miles 源码做构建上下文。

```bash
mkdir -p ~/dsv4-image && cd ~/dsv4-image
# 把 Dockerfile.gfx942-dsv4 放到这个目录
DOCKER_BUILDKIT=1 docker build -f Dockerfile.gfx942-dsv4 -t miles-dsv4-gfx942:0927 .
```

- 耗时主要花在从源码编译 Transformer Engine 和 `sgl_kernel`（按 gfx942 编译）上。
- 默认 `MAX_JOBS=128`。构建机内存或核数较少时，加 `--build-arg MAX_JOBS=32`。
- 构建完成后查看镜像里的代码版本：

```bash
docker run --rm miles-dsv4-gfx942:0927 cat /root/VERSIONS.txt
```

两台服务器要用**同一个镜像**。可以各自构建，也可以把构建好的镜像文件拷过去再导入：

```bash
sha256sum -c miles-dsv4-gfx942-0927.tar.gz.sha256      # 可选：校验文件完整性
docker load -i miles-dsv4-gfx942-0927.tar.gz            # 约 29 GB，导入后 106 GB
```
也可以使用我们编译好的镜像： `amdagi/miles-dsv4-flash:rocm7.0-gfx942-2node-20260927`

---

## 2. 启动容器（两台都做）

```bash
docker run -d --name dsv4 \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 512g --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -v /shared:/shared \
  miles-dsv4-gfx942:0927 sleep infinity

docker exec -it dsv4 bash
```

- `--network host` 必须加：Ray、NCCL 和 SGLang 的跨节点通信都直接走主机网卡。
- `-v /shared:/shared` 换成你们的共享存储，两台的挂载路径要一致。
- 进容器后用 `rocm-smi` 确认能看到 8 张卡。

---

## 3. 准备模型和数据（在共享存储上做一次）

> 已经有现成的 FP8 checkpoint、`torch_dist` 和两份数据集时，跳过本节，直接在第 5 节把路径指过去（不会触发任何下载）。

下文用 `/shared/models` 和 `/shared/datasets` 作示例路径。准备好之后的目录结构：

```
/shared/models/DeepSeek-V4-Flash-FP8/               HF 格式 FP8 checkpoint（274 GB）
/shared/models/DeepSeek-V4-Flash-FP8_torch_dist/    Megatron 格式（530 GB），训练从这里加载
/shared/datasets/dapo-math-17k/dapo-math-17k.jsonl  训练数据
/shared/datasets/aime-2024/aime-2024.jsonl          评测数据
```

### 3.1 下载 checkpoint 和数据集（需要能访问 HuggingFace）

```bash
cd /root/miles
python3 scripts/amd/run_deepseek_v4.py prepare-download \
  --model-name DeepSeek-V4-Flash-FP8 \
  --model-dir /shared/models --data-dir /shared/datasets
```

这一步会下载 `sgl-project/DeepSeek-V4-Flash-FP8`，以及数据集 `zhuzilin/dapo-math-17k`、`zhuzilin/aime-2024`。

### 3.2 生成 `torch_dist`

`torch_dist` 是 Megatron 的分布式 checkpoint，格式与硬件无关，在任何机器上转换好的都能用。如果已经有现成的，直接拷到上面的路径即可。

自己转换分两步：先把 FP8 转成 BF16（单节点），再把 BF16 转成 `torch_dist`（使用 GPU）。

```bash
cd /root/miles
python3 scripts/amd/run_deepseek_v4.py prepare-single \
  --model-name DeepSeek-V4-Flash-FP8 --model-dir /shared/models --data-dir /shared/datasets \
  --hf-checkpoint /shared/models/DeepSeek-V4-Flash-FP8
python3 scripts/amd/run_deepseek_v4.py prepare-spmd \
  --model-name DeepSeek-V4-Flash-FP8 --model-dir /shared/models --data-dir /shared/datasets \
  --num-nodes 1 --num-gpus-per-node 8
```

> 注意：这两步转换是 launcher 原有的功能，**我们没有在 gfx942 上验证过**。验证用的 `torch_dist` 是之前在别的机器上转换好的。转换中间会生成约 550 GB 的 BF16 checkpoint。

---

## 4. 启动 Ray 集群（每台服务器的容器里各做一次）

先定下两台的 IP 和网卡：用两台之间互通的那张网卡，下面记为 `NIC`。

**head 节点（节点 A）：**

```bash
export NIC=ens50f0                                   # 换成实际网卡
export HEAD_IP=$(ip -o -4 addr show $NIC | awk '{print $4}' | cut -d/ -f1)
ray stop --force
ray start --head --node-ip-address=$HEAD_IP \
  --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --num-gpus=8 --object-store-memory=17179869184 --disable-usage-stats
```

**worker 节点（节点 B）：**

```bash
export NIC=ens50f0
export HEAD_IP=<节点 A 的 IP>
export NODE_IP=$(ip -o -4 addr show $NIC | awk '{print $4}' | cut -d/ -f1)
ray stop --force
ray start --address=$HEAD_IP:6379 --node-ip-address=$NODE_IP \
  --num-gpus=8 --object-store-memory=17179869184 --disable-usage-stats
```

在 head 节点上运行 `ray status`，应能看到 2 个节点、`16.0 GPU`。

- 6379 / 8265 端口被占用时（比如多人共用的机器），换成别的端口，第 5 节的 `RAY_ADDRESS` 也要跟着改。
- `--object-store-memory` 限制为 16 GiB 是验证时的设置，建议保持。

---

## 5. 启动训练（在 head 节点的容器里）

```bash
export NIC=ens50f0
export HEAD_IP=$(ip -o -4 addr show $NIC | awk '{print $4}' | cut -d/ -f1)
export RUN_ID=dsv4-gfx942-$(date +%m%d-%H%M)
export OUT=/shared/runs/$RUN_ID
mkdir -p $OUT
cd /root/miles

MILES_SCRIPT_EXTERNAL_RAY=1 \
RAY_ADDRESS=http://$HEAD_IP:8265 \
MASTER_ADDR=$HEAD_IP \
NCCL_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC \
nohup python3 -u scripts/amd/run_deepseek_v4.py train \
  --mode normal --num-nodes 2 --num-gpus-per-node 8 \
  --model-dir /shared/models --data-dir /shared/datasets \
  --hf-checkpoint /shared/models/DeepSeek-V4-Flash-FP8 \
  --run-id $RUN_ID --output-dir $OUT --save-dir $OUT --skip-saving \
  > $OUT/train.log 2>&1 &

tail -f $OUT/train.log
```

- 所有训练参数由 launcher 自动决定。它会探测到 gfx942，然后选择：
  - tensorwise FP8；
  - 192 GB 卡的内存配置；
  - TP1 / PP2 / CP8 / EP8 的并行方式；
  - 两机全量 CPU offload；
  - 跟 SGLang rollout 对齐的精度设置。

  启动日志开头会打印选中的配置。
- `--skip-saving` 表示不保存训练 checkpoint。需要保存时去掉它，并确保 `--save-dir` 所在的盘每次有约 4 TB 空间。**保存 checkpoint 这条路径没有在 gfx942 上验证过。**
- 训练作为 Ray job 独立运行，launcher 进程退出不影响训练。

**查看和停止：**

```bash
ray job list                     # 找到 submission id
ray job logs -f <submission_id>  # 跟踪日志
ray job stop <submission_id>     # 停止训练
ray stop --force                 # 两台都执行，彻底清理
```

---

## 6. 验收：怎样判断跑得正常

**时间线**（验证环境，从提交开始算）：

| 阶段 | 大约耗时 |
|---|---|
| 4 个 SGLang 引擎就绪 | ~10 分钟 |
| 第 0 步 AIME 评测（240 个样本） | ~15 分钟 |
| 每个 rollout（256 个样本） | 20–25 分钟 |
| 每个训练 step | 第 0 步约 3 小时（含评测），之后约 2.2–2.4 小时，一天约 10 步 |

**日志里的关键指标**（`grep "step N:"`、`grep "eval/aime"`）：

| 指标 | 正常范围（验证值） |
|---|---|
| 第 0 步 `eval/aime` | 0.39–0.43（两次验证分别为 0.392、0.429） |
| `train/train_rollout_kl` | 0.0062–0.0070 |
| `train/train_rollout_logprob_abs_diff` | 0.037–0.041 |
| `train/grad_norm` | ≈ 0.09 |
| `train/ess_ratio` | ≈ 1.0 |
| `rollout/raw_reward` | 验证中 28 个 rollout 在 −0.08 到 0.27 之间，每批题不同，波动很大，看趋势不看单点 |
| 第 20 步 `eval/aime` | ≈ 0.47 |

`train_rollout_kl` 明显高于 0.01、或 `logprob_abs_diff` 明显高于 0.05，说明训练和推理的数值没有对齐，请先查第 7 节。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `train.log` 不再更新，launcher 进程已退出 | launcher 跟踪训练日志的连接可能中断（比如 step 末尾 head 节点很忙时），训练作为独立的 Ray job 会继续跑。用 `ray job status <submission_id>` 看状态，用 `ray job logs -f <submission_id>` 继续看日志 |
| NCCL 初始化卡住或超时 | `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` 没指到两台互通的网卡；检查防火墙 |
| 日志里有 `Driver ionic does not support the kernel ABI` | 容器里的 RDMA 库跟宿主机驱动版本不匹配，RCCL 会自动改走 TCP，可以忽略。要用 RDMA 需要宿主机侧处理 |
| `ray status` 只有 8 张卡 | worker 没加入集群：检查 `HEAD_IP`、端口，以及 worker 的 `ray start` 输出 |
| 主机内存 OOM | 每台峰值约需 2.2 TB。先确认没有其他大内存进程 |
| 引擎启动时报 KV cache 显存不足 | 192 GB 卡上 SGLang 默认占 0.75，可以用 `--sglang-mem-fraction-static` 调整（在 launcher 参数里加） |
| `eval/aime` 接近 0 | 数据路径不对（应为 `aime-2024/aime-2024.jsonl`），或用了旧版代码 |
| `train_rollout_kl` 偏高 | 确认镜像是用本 Dockerfile 构建的（`cat /root/VERSIONS.txt`），其中包含 sglang 的数值修复和 `sgl_kernel` 重编 |

需要支持时，请附上 `/root/VERSIONS.txt`、`$OUT/train.log` 以及 `ray status` 的输出。
