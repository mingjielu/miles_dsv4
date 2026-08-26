# DSV4-Flash 4 层 128K 显存优化 — CP=8 验证

把 CP=4 上得出的七项优化(见 `DSV4-128K-完整显存分析报告.md`、`DSV4-128K-显存分析报告-第二轮.md`、
`DSV4-128K-显存优化-复现指南.md`)逐项搬到 CP=8 上重测,确认哪些在 CP=8 仍然成立。

镜像：`miles-dsv4-flash:rocm7.14-gfx942-cp8-20260821-fada`(容器 `cp8_verify`)
硬件：单节点 8× MI308X (gfx942)，192 GiB/卡
配置：128K 序列 / **CP=8 / TP=1** / EP=8 / `--mem-fraction 0.25` / 3 步

CP=4 与 CP=8 的差别决定了为什么必须重测：CP=8 把 S_local 从 32768 砍到 16384，把 TP 从 2 降到 1。
前者让所有按 `S_local` 缩放的分配减半，后者让每卡拿到全部 64 个 attention head、并且不再按 TP 切分
词表——也就是说，各项优化的分母全变了，CP=4 的 GiB 数字不能直接搬用。

---

## 方法

`porting/apply_mem_opts.py` 把七项改动拆成可独立开关的组，`--set` 先把所有文件恢复原始再精确应用
指定子集(`model.py` 同时承载 `logits` 与 `logprob_gc`，就地单组回退会连带撤销另一组，所以只有
"回到原始再重放子集"这一种组合方式对任意子集都正确)。

两项固定进所有配置，作为控制变量而非被测项：

- **`metrics`** — 测量前提。脚本原先报的 `used_GB` 是六个阶段边界上的瞬时采样，Ray 又会折叠跨 actor
  的重复日志，取"存活行的最大值"是运气不是测量。改报 `max_allocated_GB`(进程级单调高水位)后，
  任何一行存活日志都是该 rank 的真实峰值。
- **`intmax`** — 不是显存优化，而是**固定 backward 选择**。它决定 sparse-MLA 走 kernel 还是 reference，
  两者相差数十 GiB；不固定的话各组之间根本不可比。

其余五项按累加顺序验证，每次只增一项，增量即该项在 CP=8 上的贡献。

---

## 测量前提的验证

第一轮报告在 CP=4 上的论断是 `used_GB` 会低估峰值(83 → 105.49 GiB)。CP=8 上同样成立，而且低估
幅度更大：

| 读数来源 | CP=8 峰值 |
|---|---|
| `used_GB` 采样(旧口径，曾写进镜像 LABEL) | 79.42 GiB |
| `max_allocated_GB` 高水位(本轮) | **107.27 GiB** |

差 27.85 GiB。`RAY_DEDUP_LOGS=0` 生效后 rank 覆盖率 8/8，八个 rank 的高水位分别是
100.78 / 101.21 / 100.25 / 100.57 / 99.67 / 99.48 / 99.94 / **107.27**——rank7 明显高出约 7 GiB，
与第一轮报告"温度除法只在 rank6/rank7 上显现"的观察指向同一处：响应分块落在 CP zigzag 布局的尾部
rank 上。

**镜像 LABEL 里的 79.42 GiB 需要按新口径更正。**

### `intmax` 在 CP=8 上的实证

baseline 一轮日志里三种 topk 的 backward 选择：

| topk | reference 反向的 scatter 元素数 `B·S·topk·D` | 是否超 INT_MAX | 实际走的路径 |
|---|---|---|---|
| 1152 | 9.66e9 | 超 | kernel(guard 拦回) |
| 640 | 5.37e9 | 超 | kernel(guard 拦回) |
| 128 | 1.07e9 | 未超 | reference **与** kernel 都出现过 |

topk=128 的两种记录来自同一轮运行的不同时刻——走哪条取决于当时的空闲显存
(`_reference_is_affordable`)。这正是第二轮报告描述的"同样的代码与配置有时崩有时不崩"，也说明
没有这条 guard 时 topk=640/1152 的层会在 CP=8 上撞
`cub sort does not support sorting more than INT_MAX elements`。

---

## 逐项结果

基线 = `metrics` + `intmax`，3/3 步 PASS，**107.27 GiB allocated / 130.28 GiB reserved**，
步时 134.75 / 124.73 s(首步含 tilelang JIT)。

| # | 累加项 | 峰值 allocated | 相对基线 | 本项增量 | 步时 | CP=4 对照 | 结论 |
|---|---|---|---|---|---|---|---|
| A | 基线(`metrics`+`intmax`) | 107.27 GiB | — | — | 134.8 / 124.7 s | 105.49 GiB | — |
| B | +`logits` | **98.00 GiB** | −9.27 | **−9.27** | 128.5 / 120.6 s | −7.89 GiB | 有效 |
| C | +`temperature` | **96.50 GiB** | −10.77 | **−1.50** | 129.9 / 134.1 s | −5.41 GiB | 有效但受限 |
| D | +`rms_norm` | **87.00 GiB** | −20.27 | **−9.50** | 128.0 / 119.8 s | −6.11 GiB(回放) | 有效 |
| E | +`recompute_gc` | **68.13 GiB** | −39.14 | **−18.87** | 122.2 / 121.3 s | −9.88 GiB | 有效，收益最大 |
| F | +`logprob_gc` | **52.45 GiB** | **−54.82** | **−15.68** | 124.8 / 124.8 s | −23.55 GiB(回放) | 有效 |

**总计 107.27 → 52.45 GiB，−54.82 GiB(−51.1%)**，reserved 130.28 → 63.18 GiB。CP=4 的对照是
105.49 → 56.17 GiB(−46.7%)，所以这套优化在 CP=8 上比在 CP=4 上更划算。五项全部有效，一项都不用丢。

### B — `logits`：−9.27 GiB，CP=8 上比 CP=4 更划算

3/3 步 PASS，无 OOM/NaN，**没有出现 CP 死锁**——`cp_utils` 的 `.float()` 前置修复在 CP=8 上同样是
必需且充分的。

逐 rank 的形态比总数更说明问题：

| | rank0 | rank1 | rank2 | rank3 | rank4 | rank5 | rank6 | rank7 |
|---|---|---|---|---|---|---|---|---|
| A 基线 | 100.78 | 101.21 | 100.25 | 100.57 | 99.67 | 99.48 | 99.94 | **107.27** |
| B +logits | 98.00 | 98.00 | 98.00 | 98.00 | 98.00 | 96.50 | 98.00 | **98.00** |

基线里 rank7 高出其余 rank 约 7 GiB，加上本项后八个 rank 齐平到 98.00。那份全量 fp32 副本正是落在
承担响应分块的尾部 rank 上，去掉它之后峰值不再由某一个 rank 决定。第一轮报告在 CP=4 上观察到
"只有 rank6/rank7 有明显变化"，是同一现象的另一面。

CP=8 收益(−9.27)大于 CP=4(−7.89)，但 fp32 副本本身两种配置下一样大：CP=4/TP=2 是
`[S/4, V/2]`、CP=8/TP=1 是 `[S/8, V]`，元素数相同。差出来的约 1.4 GiB 来自峰值时刻同时存活的其他
分配不同，不是这项本身变大了。

性能无退化，反而略快(step time −6.2 / −4.1 s，`Timer train` 92.6 s 两边一致，wall time 965 → 963 s)。

### C — `temperature`：省下来了，但被峰值的取法吃掉

3/3 步 PASS。逐 rank 看，这项在 CP=8 上的效果和 CP=4 报告的描述完全一致；问题在于峰值是**全局最大值**：

| | rank0 | rank1 | rank2 | rank3 | rank4 | rank5 | rank6 | rank7 |
|---|---|---|---|---|---|---|---|---|
| B(基线+logits) | 98.00 | 98.00 | 98.00 | 98.00 | 98.00 | 96.50 | 98.00 | 98.00 |
| C(+temperature) | 96.50 | **93.32** | 96.50 | 96.50 | 96.50 | 96.50 | 96.50 | **91.93** |

rank1 降 4.68 GiB、rank7 降 6.07 GiB——正是第一轮报告"可能只有 rank6/rank7 下降约 6.7 GiB"的那个
形态。但其余六个 rank 只降 1.50 GiB，而峰值取八个 rank 的最大值，于是整体只体现为 −1.50 GiB。

CP=4 报出 −5.41 GiB 是因为那里的峰值 rank 恰好是受益的那个；CP=8 上前一项(`logits`)已经把八个
rank 拉平到同一水平，受益 rank 不再是峰值所在，收益就被掩盖了。**这不是"这项在 CP=8 失效"，而是
"在 CP=8 上它不再决定峰值"。**

保留它：数值上是精确等价的变换(逐元素除法与切片可交换)，代价为零，而且它降低了非峰值 rank 的压力，
对更大 batch / 更长响应留出余量。

### D — `rms_norm`：−9.50 GiB，CP=8 上的最大单项

3/3 步 PASS，八个 rank 全部 87.00 GiB，步时 128.0 / 119.8 s(无退化)。

手写版本的 `q.float()` 会为每层实体化一份 `[1, S_local, H_local, D]` fp32 副本，backward 一直持有它，
并且让 q 的梯度也变成 fp32。在 CP=8/TP=1 下这个形状是 `[1, 16384, 64, 512]` = 2 GiB/层，4 层共 8 GiB，
与实测的 −9.50 GiB 同量级(余量来自 fp32 梯度的连带)。

注意这个形状在两种配置下一样大：CP=4/TP=2 是 `[1, 32768, 32, 512]`，S 加倍而 H 减半。CP=4 报的
−6.11 GiB 是快照回放峰值，与这里的 `max_allocated` 增量口径不同，不能直接比。

**数值等价性单独验证**：4 层剪枝模型的 reward 恒为 0，因此 loss 与所有梯度都是 0，训练跑得通并不能
说明这项改动数值无害。用 `porting/verify_rms_norm_equiv.py` 在真实形状上以 fp64 为参考对照两种写法的
前向与反向(结果见下文"数值等价性"一节)。

### E — `recompute_gc`：−18.87 GiB，CP=8 上收益最大的一项

3/3 步 PASS，峰值 **68.13 GiB**(reserved 84.69)。CP=4 报的是 −9.88 GiB，CP=8 上接近其两倍。

为什么在 CP=8 上更大：这项回收的是 checkpointed backward 重算出来、且只能通过 autograd
`ctx <-> node` 引用环到达的激活。CP=8/TP=1 下每卡持有全部 64 个 head，重算激活的绝对量比
CP=4/TP=2 更大，能回收的也更多。

开关是 `MEGATRON_RECOMPUTE_BACKWARD_GC_GEN`(-1 关闭)，已在 `run_4layer_1node.sh` 里默认导出为 2，
可从外部覆盖以便 A/B。

**时间代价**：`Timer train` 92.2 / 93.5 → 95.3 / 97.4 s(+4%)，step time 128.0 / 119.8 →
122.2 / 121.3 s(基本相当)。wall time 955 → 1144 s 的差额主要落在首步 JIT 与启动段
(log_probs 首次 188.5 → 196.2 s、train 首次 312.2 → 328.0 s)，稳态步时里看不到相应幅度，
所以更像启动噪声而非 GC 的稳态开销。gen 的取值还有优化空间(见下文"GC generation 的选择")。

### F — `logprob_gc`：−15.68 GiB，八个 rank 完全齐平

3/3 步 PASS，峰值 **52.45 GiB**(reserved 63.18)，八个 rank 全部 52.45——峰值不再由任何单个 rank
决定。步时 124.8 / 124.8 s，wall time 980 s。

`forward_only` 在 `no_grad` 下算 log_probs，返回后所有激活都是垃圾，但其中一部分只能通过引用环到达，
和 `recompute_gc` 是同一个成因、不同的代码路径。CP=4 报的 −23.55 GiB 是快照回放峰值；这里 −15.68 GiB
是 `max_allocated` 高水位的降幅，口径不同但方向一致。

顺带证实了 E 那次 wall time 1144 s 是噪声：F 在 E 的全部改动之上又加了一项 GC，wall time 反而回到
980 s，与基线的 965 s 相当。

---

## 数值等价性

只有 `rms_norm` 触及数值。4 层剪枝模型的 reward 恒为 0，因此 loss 和所有梯度都是 0，训练能跑通并不
构成数值证据。`porting/verify_rms_norm_equiv.py` 在真实形状上以 **fp64 为参考**分别衡量两种写法：

| 形状 | dtype | manual 前向 | fused 前向 | manual 反向 | fused 反向 |
|---|---|---|---|---|---|
| (1, 256, 64, 512) | bf16 | 5.866e-03 | 5.866e-03 | 4.476e-03 | 4.476e-03 |
| (1, 256, 64, 512) | fp32 | 1.461e-07 | 1.461e-07 | 1.906e-07 | 1.796e-07 |
| (1, 256, 32, 512) | bf16 | 5.866e-03 | 5.866e-03 | 4.766e-03 | 4.766e-03 |
| (1, 17, 64, 512) | fp32 | 1.241e-07 | 9.762e-08 | 1.598e-07 | 1.594e-07 |

bf16 下两者对参考的误差**逐位相同**；fp32 下都在 1e-7 量级、互有胜负。误差本身(bf16 5.9e-03)就是
bf16 自身的舍入水平，不是任一实现引入的。

**一处需要修正复现指南的说法**：指南称手写版本"让 q 的梯度也变成 fp32"。实测两种写法的 `q.grad`
dtype 相同(bf16 输入 → bf16 梯度)——手写版本末尾的 `.to(q.dtype)` 已经把输出转回，autograd 也就
把梯度转回。省下来的是那份被 backward 持有的**中间** fp32 副本，与梯度 dtype 无关。

---

## GC generation 的选择

gen2 是全量扫描，看起来是可以省的：recompute 产生的垃圾刚分配不久，"应该"还在 gen0，那么
`gc.collect(0)` 就足以打破引用环而代价小得多。**实测不成立。**

| 配置 | 峰值 allocated | reserved | `Timer train` | step time | wall |
|---|---|---|---|---|---|
| F，gen=2 | **52.45 GiB** | 63.18 | 97.6 / 98.4 s | 124.8 / 124.8 s | 980 s |
| G，gen=0 | 63.63 GiB | 77.12 | 94.7 / 96.5 s | 127.9 / 121.0 s | 964 s |

gen=0 比 gen=2 高 **11.18 GiB**，换来的只有约 2 s/step。**gen=2 是必要的，不是保守取值。**

原因在于 CPython 分代回收的判定方式：一个引用环只有在**环内所有对象都落在被扫描的世代里**时才能被
判定为垃圾。`collect(0)` 只看 gen0，而扫描后存活的 gen0 对象会被晋升到 gen1——于是第一次因为环里
掺了一个更老的对象而没被回收的那批张量，此后就永远落在 `collect(0)` 的视野之外。

`run_4layer_1node.sh` 默认导出 gen=2，保留环境变量覆盖以便复测：

```bash
MEGATRON_RECOMPUTE_BACKWARD_GC_GEN=0 run-dsv4-4layer-1node --cp 8 --steps 3
```

---

## expandable_segments 分段窗口(第三轮方案)

第三轮报告([原文](DSV4-128K-显存分析报告-第三轮.md))给出的做法是：**只在训练步内开启
`expandable_segments`，步末 `empty_cache()` 后关闭**。这同时绕开两个此前让该设置无法使用的障碍——
而其中一个是我之前归因错的：

| 障碍 | 我此前在 `final_report.md` 的说法 | 第三轮实测 |
|---|---|---|
| A. IPC 权重同步 | "VMM 内存导不出 IPC handle" | **导出是成功的**；失败在消费侧 `hipIpcOpenMemHandle`，返回 `hipErrorInvalidValue` |
| B. TMS offload | 未提及 | torch_memory_saver 的 preload 钩子**只拦 `hipMalloc`/`hipFree`**，expandable 走 `hipMemCreate`/`hipMemMap` 完全绕过 → `offload_train` 静默失效，SGLang 启动时要求 `mem-fraction-static ≥ 0.45`（基线只需 0.085） |

窗口把权重、梯度缓冲和 IPC bucket 全部留在窗口外分配，只让步内的临时激活（碎片的来源）用 expandable
段。**两个前提先在本机验证**（`porting/verify_expandable_window.py`，gfx942 / torch 2.11.0+rocm7.14.0）：

```
OK  toggle   start 0/0 expandable; after ON-alloc 1/1; after OFF-alloc 1/2
OK  release  reserved 0.00 -> 2.01 -> 0.00 GiB (leftover +0.00); segments still held: 0 (0 expandable)
```

即运行时切换确实生效（ON 时新分配进 expandable 段、OFF 时进普通段），且关窗后物理页零残留归还。
`memory_snapshot()` 的字段名是 `is_expandable`——拼成 `is_expandable_segment` 会让计数恒为 0，
从而让检查"永远通过"，所以脚本里对该字段存在性做了硬断言。

### 从基线直接开 vs 优化完再开

两组各跑 3 步，都是 3/3 PASS、8/8 rank、零 OOM/NaN/LDS，`update_weights` 3.8–4.3 s
（即**权重同步正常，窗口方案确实绕开了障碍 A**）：

| 配置 | allocated | reserved | 碎片 gap | 步时 |
|---|---|---|---|---|
| A 基线 | 107.27 | 130.28 | 23.01 | 134.8 / 124.7 s |
| **H 基线 + expandable** | 106.96 | **120.39** | 13.43 | **111.9 / 111.3 s** |
| F 全部优化 | 52.45 | 63.18 | 10.73 | 124.8 / 124.8 s |
| **I 全部优化 + expandable** | **52.45** | **60.30** | 7.85 | **115.9 / 116.1 s** |

**expandable 本身的收益，在基线上是优化之后的 3.4 倍：**

| 加在哪个基础上 | reserved 降幅 | 碎片压缩 |
|---|---|---|
| 基线 | **−9.89 GiB**（130.28 → 120.39） | 23.01 → 13.43 |
| 全部优化之后 | **−2.88 GiB**（63.18 → 60.30） | 10.73 → 7.85 |

原因不难理解：碎片量取决于分配器池子里空洞的总量，而前面那五项优化（尤其两个 GC 组）已经大幅减少了
分配/释放活动，池子本身小了一半以上，能压缩的碎片自然也少了。**两类优化不是简单相加，而是互相吃掉
对方的收益空间**——若可加，130.28 − 9.89 − 67.10 应得 53.29，实测 60.30，重叠约 7 GiB。

一个佐证：F（不开 expandable）的碎片 gap 10.73 已经低于 H（只开 expandable）的 13.43。**五项优化本身
就顺带减少了碎片，比 expandable 单独做得更好。**

`allocated` 两组都逐 rank 逐位不变（106.96 vs 107.27 的 0.31 差异来自 rank0/4/5/6 的采样时刻，
rank1/2/3/7 完全相同），确认 expandable 没有改变任何计算路径——它只影响分配器如何摆放同样的张量。

### 该怎么选

如果只能做一件事，**做那五项优化**：设备实际要容纳的是 reserved，基线 130.28 → 63.18 是 −51.5%，
而只开 expandable 只有 −7.6%。两者都做再多拿 2.88 GiB（→ 60.30，累计 −53.7%）。

不过 expandable 还有一项独立价值：**它在两种基础上都把步时压下来约 9–23 s**（基线 134.8 → 111.9，
优化后 124.8 → 115.9），来自碎片减少带来的分配器开销下降。这一项优化组是拿不到的。

因此推荐同时开启，但要清楚二者的性质不同：五项优化减少的是**活跃张量**，expandable 减少的是
**分配器摆放它们的浪费**。

```bash
MILES_EXPANDABLE_SEGMENTS_DURING_TRAIN=1 run-dsv4-4layer-1node --cp 8 --steps 3
# 52.45 GiB allocated / 60.30 GiB reserved / ~116 s/step
```

默认关闭，因为它依赖运行时切换分配器这一相对少见的能力；换环境后应先跑
`verify_expandable_window.py` 确认两个前提再开。

---

## 落地

已同步进 `porting/Dockerfile` 第 3 节：`COPY apply_mem_opts.py` + 无参数应用(全部七组)，随后逐个
sentinel 校验；`porting/verify_image.py` 的镜像 gate 也加了这八处 sentinel 检查，缺任何一处即 build
失败。`run_4layer_1node.sh` 默认导出 `MEGATRON_RECOMPUTE_BACKWARD_GC_GEN=2`。

镜像 LABEL 的 `fada.verified` 从 `79.42 GiB` 更正为 `52.45 GiB (107.27 before)`，并新增
`fada.peak_metric` 说明口径是 `max_memory_allocated` 高水位而非旧的 `used_GB` 采样——否则下一个人
会拿这两个不可比的数字互相对照。

### 复现

```bash
# 最终配置(镜像内默认已全部应用)
run-dsv4-4layer-1node --cp 8 --steps 3
# 预期: 3/3 步、52.45 GiB allocated / 63.18 GiB reserved、8/8 ranks、0 OOM/NaN/LDS

# 逐项 A/B
python3 /opt/miles-rocm/patches/apply_mem_opts.py --status
python3 /opt/miles-rocm/patches/apply_mem_opts.py --set metrics intmax          # 基线
python3 /opt/miles-rocm/patches/apply_mem_opts.py --set metrics intmax logits   # 累加一项
python3 /opt/miles-rocm/patches/apply_mem_opts.py                              # 全部
# 数值等价性
python3 /opt/miles-rocm/tools/verify_rms_norm_equiv.py
```

`--set` 先把所有文件还原成原始再精确应用给定子集。不要用"就地撤销单组"的方式做 A/B：`model.py`
同时承载 `logits` 与 `logprob_gc`，备份是原始文件，撤销其中一组会连带撤销另一组。

### 与 CP=4 的差异汇总

| 项 | CP=4 | CP=8 | 差异原因 |
|---|---|---|---|
| 基线(真实高水位) | 105.49 GiB | 107.27 GiB | 相当 |
| `logits` | −7.89 | −9.27 | fp32 副本同样大(`[S/4,V/2]` vs `[S/8,V]`)，差额来自峰值时刻并存的其他分配 |
| `temperature` | −5.41 | −1.50 | CP=8 上 `logits` 已把各 rank 拉平，受益 rank 不再是峰值所在 |
| `rms_norm` | −6.11(回放) | −9.50 | 口径不同；形状两边一样大 |
| `recompute_gc` | −9.88 | −18.87 | TP=1 时每卡持有全部 64 head，可回收的重算激活更多 |
| `logprob_gc` | −23.55(回放) | −15.68 | 口径不同 |
| 合计 | −49.32(−46.7%) | **−54.82(−51.1%)** | |
