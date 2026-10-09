# DeepSeek-V4-Flash FP8 RL 快速复现（2 台 × 8 卡 gfx942）

- 镜像：`/apps/xiaohong/image/miles-dsv4-gfx942-0927.tar.gz`（导入后为 `miles-dsv4-gfx942:0927`）
- 模型：`/apps/mingjiel/FADA/models/dsv4-flash-full`（FP8 checkpoint + `torch_dist`，现成的）
- 数据：`/apps/mingjiel/data`（训练用 `dapo-math-17k`，评测用 `aime-2024`，现成的）
- 前提：两台都挂载了 `/apps`，两台之间通过网卡 `ens50f0` 互通。换机器时用 `ip -o -4 addr` 找到互通的网卡，替换下面的 `NIC`
- 选机器（每台都要满足）：
  - 8 张卡显存空闲（`rocm-smi --showmemuse`）。被别人占着的话，引擎会 OOM
  - `free -g` 的 available ≥ 2200 GB。实测训练每台额外占用约 1.8 TB，Ray 在主机内存用到 95% 时会杀进程
  - docker 盘剩余 ≥ 120 GB（镜像 106 GB）
- 已验证：2026-09-27，gpufc58（节点 A）+ gpuf2c5（节点 B）

## 1. 加载镜像（两台都做）

```bash
cd /apps/xiaohong/image
sha256sum -c miles-dsv4-gfx942-0927.tar.gz.sha256   # 可选，应输出 OK
sudo docker load -i miles-dsv4-gfx942-0927.tar.gz    # 校验加导入约 6 分钟，导入后 106 GB
```

`sudo docker images | grep miles-dsv4-gfx942` 已经能看到 `0927` 的机器，跳过这一步。

## 2. 启动容器（两台都做）

```bash
sudo docker run -d --name dsv4_0927 \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 512g --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -v /apps:/apps \
  miles-dsv4-gfx942:0927 sleep infinity

sudo docker exec -it dsv4_0927 bash
```

后面的命令都在容器里执行。

## 3. 启动 Ray

节点 A（head）：

```bash
export NIC=ens50f0
export HEAD_IP=$(ip -o -4 addr show $NIC | awk '{print $4}' | cut -d/ -f1)
echo $HEAD_IP    # 节点 B 要用这个 IP
ray stop --force
ray start --head --node-ip-address=$HEAD_IP --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --num-gpus=8 --object-store-memory=17179869184 --disable-usage-stats
```

节点 B（worker）：

```bash
export NIC=ens50f0
export HEAD_IP=<节点 A 的 IP>
export NODE_IP=$(ip -o -4 addr show $NIC | awk '{print $4}' | cut -d/ -f1)
ray stop --force
ray start --address=$HEAD_IP:6379 --node-ip-address=$NODE_IP \
  --num-gpus=8 --object-store-memory=17179869184 --disable-usage-stats
```

回到节点 A 执行 `ray status`，应看到 2 个节点、`16.0 GPU`。6379 或 8265 端口被占用时换一个端口，第 4 步的 `RAY_ADDRESS` 也跟着改。

## 4. 启动训练（节点 A）

```bash
export NIC=ens50f0
export HEAD_IP=$(ip -o -4 addr show $NIC | awk '{print $4}' | cut -d/ -f1)
export RUN_ID=dsv4-gfx942-$(date +%m%d-%H%M)
export OUT=/apps/xiaohong/runs/$RUN_ID
mkdir -p $OUT && cd /root/miles

MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://$HEAD_IP:8265 MASTER_ADDR=$HEAD_IP \
NCCL_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC \
nohup python3 -u scripts/amd/run_deepseek_v4.py train \
  --mode normal --num-nodes 2 --num-gpus-per-node 8 \
  --model-dir /apps/mingjiel/FADA/models/dsv4-flash-full \
  --data-dir /apps/mingjiel/data \
  --hf-checkpoint /apps/mingjiel/FADA/models/dsv4-flash-full/DeepSeek-V4-Flash-FP8 \
  --run-id $RUN_ID --output-dir $OUT --save-dir $OUT --skip-saving \
  > $OUT/train.log 2>&1 &

tail -f $OUT/train.log
```

训练参数由 launcher 按 gfx942 自动选好（tensorwise FP8、192 GB 显存配置等），不需要改。`--skip-saving` 表示不保存 checkpoint。

## 查看和停止（节点 A 的容器里）

```bash
ray job list                          # 拿到 submission id（raysubmit_...）
ray job logs -f <submission_id>       # 跟踪日志

# 每步的训推一致性指标，以及 AIME 评测分数
ray job logs <submission_id> | grep "model.py.*- step [0-9]*:" \
  | grep -oE "step [0-9]+:|'train/train_rollout_(kl|logprob_abs_diff)': [0-9.e-]+" | paste - - -
ray job logs <submission_id> | grep -oE "'eval/aime': [0-9.e-]+"

ray job stop <submission_id>          # 停止训练
ray stop --force                      # 两台的容器里都执行
```

## 正常时应该看到

以下是 09-27 在 gpufc58 + gpuf2c5 上用本镜像跑 step 0–5 的实测值：

| 项目 | 实测 |
|---|---|
| 从提交到开始第一次权重同步 | 约 16 分钟 |
| 第 0 步 AIME（`eval/aime`） | 0.429 |
| 从提交到出第 0 步训练指标 | 约 3.5 小时（含第 0 步评测） |
| 之后每个训练 step | 2.2–2.5 小时 |
| `train_rollout_kl` | 0.0062–0.0068 |
| `train_rollout_logprob_abs_diff` | 0.037–0.040 |

- `train_rollout_kl` 明显超过 0.01，或 `logprob_abs_diff` 超过 0.05，说明训练和推理的数值没对齐。
- `train.log` 不再更新、但 `ray job status <submission_id>` 仍是 RUNNING：属于正常情况。launcher 跟日志的连接断了，训练还在跑，改用 `ray job logs -f` 看。
- 日志里会有十来个 traceback（`freeze_gc` 连接被拒、`git_state` 超时等），还有 `Driver ionic does not support the kernel ABI`，都不影响训练。
