# FADA final report — `miles_spec_20260817_1430`

**Workflow:** `specified_config` · **Framework:** Miles (Megatron train + SGLang rollout, colocate)
**Target:** 1 node × 8 AMD Instinct MI308X (gfx942), ROCm 7.14
**LOCKED base image:** `amdagi/miles-dsv4-flash:rocm7.14-gfx942-20260812-fada` — **used unchanged** ✅
(corrected by the user from `…-20260804` on 2026-08-21; see **D6**. Both tags were used unchanged.)
**Host:** `smc300x-ccs-aus-gpuf2b9...` · **Generated:** 2026-08-18 UTC, last updated 2026-08-25 UTC

**Instruction:** 参考 `exp/miles_accuracy_20260809_0202/cases/dsv4_4layer_train_rollout_consistency`
中的设置，在 `amdagi/miles-dsv4-flash:rocm7.14-gfx942-20260804` 这个镜像中执行任务。要求：单机运行
4layer，miles 框架，输入 128k，要求开 cp。cp 的支持参考
[Primus-dsv4-gfx942 `run_128k_dense_hca_csa.sh`](https://github.com/aaab8b/Primus-dsv4-gfx942/blob/dsv4-gfx942-128k-dense-hca-csa/examples/deepseek-v4/gfx942/run_128k_dense_hca_csa.sh)。

---

## Summary

**It works at CP=8 — the full Primus reference shape — after two gfx942 fixes.**

Miles trains the four-layer DeepSeek-V4-Flash on one 8xMI308X node with a real **131072-token
input** and **context parallelism at CP=8 / TP=1**, inside the locked image: 3/3 iterations,
**52.45 GiB peak** of 192, **~125 s/step**, zero OOM / NaN / hang / RCCL timeout.

Getting there needed two independent fixes, and neither is sufficient alone:

| fix | problem | without it |
|---|---|---|
| `block_H` (forward sparse-MLA head blocking) | `Q_shared` is `[64,512]` bf16 = 65536 B at TP=1 — the entire gfx942 LDS budget | CP=8 cannot **compile** |
| `index_select` instead of `gather` in `sparse_attn_torch` | gather's backward zeroes a tensor shaped like its input, `[B,S,S_kv,D]` = **2048 GiB** | CP=8 compiles, then **OOMs** |

CP=4 / TP=2 also works and needs neither fix, so it remains the fallback for an unpatched image —
but it is no longer a ceiling, and it is slightly *slower*.

**A follow-up memory pass then took the CP=8 peak from 107.27 to 52.45 GiB (−51%).** Two things to
read carefully in that sentence, because the earlier revisions of this report said 79.42 GiB:

- **79.42 GiB was a measurement artefact, not a lower peak.** It came from a max over `used_GB`, an
  instantaneous sample taken at six phase boundaries and then deduplicated across actors by Ray, so
  the reported figure depended on which ranks' lines happened to survive. Measured properly, as a
  `torch.cuda.max_memory_allocated` high-water mark with Ray dedup off and all 8/8 ranks reporting,
  the same configuration peaks at **107.27 GiB**. Every peak in this report is now the high-water
  mark; the two numbers are not comparable and should never be put side by side.
- The **−54.82 GiB** that follows is real, verified one group at a time on CP=8. Details in
  [`DSV4-128K-显存优化-CP8验证.md`](DSV4-128K-显存优化-CP8验证.md) and the section below.

| Result | Cases |
|---|---|
| ✅ PASS | 1 |
| ❌ non-PASS | 0 |

**Instruction fidelity: FULLY MET** for the requested configuration (single node, 4 layers, Miles,
128K input, CP on — at the reference's own CP=8). Deviations remaining are all minor/informational
plus the user's own base_image correction; the major CP shortfall (D1) is **resolved**. No requested
feature was silently dropped and no role ever swapped the locked image.

---

## Case results

| case_id | type | backend / mode | parallelism | verdict | key metrics |
|---|---|---|---|---|---|
| `dsv4_4layer_128k_cp8` | specified_config | megatron + sglang, colocate | **CP=8** / TP=1 / EP=8 / ETP=1 / PP=1, allgather CP | ✅ **PASS** | 3/3 steps · seq **131072** (16384/CP rank) · peak **52.45 GiB** high-water, 8/8 ranks (107.27 before the memory pass) · **~125 s/step** · 8 weight syncs · 0 OOM/NaN/LDS · 9 debug rounds + 7 memory A/B runs |

### What was verified, and how

| requirement | observed | evidence |
|---|---|---|
| context parallelism ON | `context_parallel_size = 8` (requested 8), `allgather_cp True` | Megatron's own resolved argument dump |
| 128K input | `max(rollout/total_lengths)=130599` → padded to **131072** (bucket `128*2*8=2048`), **16384 tokens per CP rank** | corroborated by SGLang `context_len=131072` |
| single node / 4 layers / Miles | 8 GPUs, `--num-layers 4`, Megatron+SGLang colocate | argument dump |
| completes | 3/3 iterations, exit 0, 28 min wall | log |
| healthy | 0 OOM, 0 NaN, 0 GPU fault, 0 assertion, 0 actor death | log grep count = 0 |
| image unchanged | checked against the lock before every attempt | `launch.sh` refuses a mismatched tag |

Miles never prints `max_seq_len` (it is computed inside `data.py`), so it is reconstructed with
`data.py`'s own padding rule from the logged `rollout/total_lengths` and cross-checked against
SGLang's independently reported context length.

**`loss`, `reward` and `advantages` are all exactly 0.0, as expected.** A 4-layer prune cannot
answer correctly → reward 0 → GRPO advantage 0 → no gradient. This run proves the 128K + CP
**pipeline and memory envelope**. It says nothing about convergence.

---

## ⚠️ Deviations from instruction

**Instruction fidelity: FULLY MET** for the requested configuration — CP is on at the reference's own
CP=8, at 131072 tokens, single node, 4 layers, Miles. D1 (the CP shortfall) is **resolved**; what
remains is 3 minor + 1 informational, plus the user's own base_image correction.

**LOCKED `base_image`:** `amdagi/miles-dsv4-flash:rocm7.14-gfx942-20260812-fada`, corrected by the
user on 2026-08-21 from `…-20260804` (**D6**). Both tags were used **unchanged** — no role ever
swapped the image to make a case pass, and `launch.sh` verifies the container's image against the
lock before each attempt.

| id | field | requested | actual | severity |
|---|---|---|---|---|
| ~~D1~~ | `context_parallel_size` | 8 | **8** — two gfx942 patches applied | ✅ **RESOLVED** |
| D2 | custom all-reduce | unspecified | `--sglang-disable-custom-all-reduce` | minor |
| D3 | `sglang-mem-fraction-static` | unspecified (0.7 default) | 0.25 | minor |
| D4 | Primus env knobs | "参考 Primus 脚本" | `HSA_NO_SCRATCH_RECLAIM=0` not adopted; `expandable_segments` **adopted, scoped to the training step** | minor |
| D5 | how 128K is realized | "输入 128k" | long prompt (~130.0–130.2K) + 512 response | informational |
| D6 | `base_image` | `…-20260804` → `…-20260812-fada` | corrected value | critical *by rule*, not a violation — the user changed their own requirement |
| ~~D7~~ | `block_H` patch reverted | — | **withdrawn**: the patch is required after all, it was just not sufficient alone | informational |

**D4 is worth reading even though it is marked minor:** the Primus reference's
`PYTORCH_ALLOC_CONF=expandable_segments:True` is *actively harmful* under Miles **when set
process-wide** — it cost two debug rounds. It is now adopted in a usable form, scoped to the training
step, which avoids both of the blockers; details in bug #2 below.

Full text: [`deviations.md`](deviations.md).

---

## Bugs root-caused

### 1. CP=8 needed two independent fixes — both now applied (major, **FIXED**)

**1a. LDS.** `tilelang_sparse_mla_fwd` stages Q/O as `[H_per_block, 512]` bf16 with
`H_per_block = padded_H` for `heads <= 64`, so at 64 heads `Q_shared` alone is 65536 B — all of
gfx942's budget. `num_stages` cannot help (it only multiplies `KV_shared`, and every fallback already
uses 1). CP=8 forces TP=1 on one node, which forces 64 local heads.

*Fix:* the kernel already blocks over heads via `REPLICATE_H` + the `H0` offset; only the block size
was hardwired to 64. Exposing it as `block_H` is a tiling change, not a numerics change — DSV4
attention is MQA and the online-softmax state is per-head, so head blocks are independent. Verified
byte-identical tilings for heads 32/16/8; heads=64 builds at `(1, 16, 64, block_H=32)`.

**1b. A 2048 GiB allocation in `sparse_attn_torch`'s backward.** With the LDS wall gone, CP=8
compiled and then asked for 2048 GiB with ~70 GiB free:

```python
torch.gather(kv.unsqueeze(1).expand(B, S, S_kv, D), 2, safe_idx...)
```

The forward is cheap (`expand` is a view). The backward is a `scatter_add` into a freshly **zeroed
tensor shaped like gather's input** — `[B, S, S_kv, D]`, exactly 2048 GiB at B=1, S=16384,
S_kv=131072, D=512.

*Fix:* index a flattened KV. `index_select` returns the same values and accumulates into
`[B*S_kv, D]` = 128 MiB. The fp32 cast is also hoisted (it was written twice, ~4 GiB per `.float()`).
fwd+bwd peak: OOM → **18.17 GiB**, `dq` bit-identical.

**Why it only appeared at CP=8, and the diagnostic trap.** Not TP-related and not caused by 1a. CP=8
halves `S_local` to 16384, and `_reference_is_affordable` plans for `8*B*S*topk*D*4`; halving S halves
the estimate, so the two dense+SWA layers (`topk=128`, not the 640 of compressed layers) drop to
~34 GB, slip under the budget, and take the reference path **for the first time**. CP=4's estimate
stays over budget and always used the kernel — which is why CP=4 looked healthy.

And the selection is **silent**: of the two branches into `sparse_attn_torch`, only
`_backward_kernel_fits` logs `falling back to the reference implementation`. Grepping for that message
finds none and wrongly exonerates the path — the mistake made in round 7 and corrected in round 9.
**A missing log line is not evidence.**

Patches + README: [`cases/dsv4_4layer_128k_cp8/patches/`](cases/dsv4_4layer_128k_cp8/patches/).
Baked and gated on both sentinels by
[`porting/Dockerfile.cp8_patches`](porting/Dockerfile.cp8_patches). Provenance:
`/apps/mingjiel/dsv4mem`.
→ [`knowledge_candidates/dsv4-cp8-128k-on-gfx942-two-fixes.md`](knowledge_candidates/dsv4-cp8-128k-on-gfx942-two-fixes.md)

### 2. `expandable_segments` silently breaks colocate weight sync — and the offload too (major, now **usable via a scoped window**)

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` process-wide made every weight sync die in
SGLang at `storage._new_shared_cuda()` with `hipErrorInvalidValue`. It read as an out-of-memory
because HIP reuses that errno for a bad handle, a failed mapping, *and* genuine exhaustion — and the
first occurrence coincided with the trainer having 7.82 GB free. Disproving the memory hypothesis
took a round that re-ran with **137.44 GB** free and got a bit-identical failure.

**Two corrections to what this report previously said**, both from the round-3 investigation:

1. The mechanism is not "VMM memory cannot be exported as an IPC handle." The **export succeeds**;
   the failure is on the consumer side, in `hipIpcOpenMemHandle`, returning `hipErrorInvalidValue`.
   `HSA_ENABLE_IPC_MODE_LEGACY=1` does not help, and `NCCL_CUMEM_ENABLE=1` is irrelevant here (it
   only affects RCCL's own buffers, not PyTorch's IPC export).
2. There is a **second, independent** blocker that this report omitted, and it is the more damaging
   one: torch_memory_saver's preload shim intercepts only `hipMalloc`/`hipFree`, while expandable
   segments go through `hipMemCreate`/`hipMemMap` and bypass it entirely. The training offload
   therefore becomes a silent no-op — SGLang then refuses to start, demanding
   `mem-fraction-static ≥ 0.45` where the baseline needs 0.085.

**The setting is now usable**, by scoping it to the training step: open it on entry to
`train_actor`, `clear_memory()` and close it on exit. Weights, gradient buffers and the IPC bucket
are all allocated outside the window, so both blockers are avoided; only the step's transient
activations — where the fragmentation is — use expandable segments. Verified on gfx942 /
torch 2.11.0+rocm7.14.0 (`porting/verify_expandable_window.py`): the runtime toggle takes effect and
closing the window returns the physical pages with zero leftover. In a real run, `update_weights`
stays at its normal 3.8–4.3 s.

Worth **−9.89 GiB of reserved on the unoptimised baseline, but only −2.88 GiB on top of the five
memory optimisations** — see the memory section below. Off by default
(`MILES_EXPANDABLE_SEGMENTS_DURING_TRAIN=1`).

The Primus script can set it process-wide safely because it is pure Megatron SFT with neither an IPC
weight-transfer path nor a TMS offload.
→ [`knowledge_candidates/expandable-segments-breaks-colocate-ipc-weight-sync.md`](knowledge_candidates/expandable-segments-breaks-colocate-ipc-weight-sync.md)

### 3. AITER custom all-reduce cannot export its buffer under SGLang colocate (minor, worked around)

`aiter/csrc/kernels/custom_all_reduce.cu:395 hipIpcGetMemHandle -> invalid argument` killed every
scheduler before weight load. Same family as #2: SGLang colocate runs with
`enable_memory_saver=True`, whose allocator also serves VMM memory. `probe_hip_ipc.py` proves plain
`hipMalloc` and torch-allocator exports are healthy on this host, so it is the buffer, not the
machine. Worked around with `--sglang-disable-custom-all-reduce`, the same switch Miles' own gemma-4
scripts use.

### 4. SGLang's default KV pool is ~500× oversized at 128K colocate (minor, fixed by config)

At `mem-fraction-static 0.7` SGLang sized `max_total_num_tokens=260763648` (~125 GB) for a workload
that uses 4 × 131072 = 524288 tokens, leaving the colocated trainer 7.82 GB. The static pool is not
released during a weight sync. At 0.25 the pool is still 69.7M tokens and the trainer has 137.44 GB.

---

## Open issues

| issue | severity | reproducer | status |
|---|---|---|---|
| Forward sparse-MLA kernel cannot build at 64 local heads on gfx942 | major | [`reproducer/probe_fwd_kernel_lds.py`](cases/dsv4_4layer_128k_cp8/reproducer/) (~2 min, no model) | **FIXED** locally by `block_H`; still **open upstream** |
| `sparse_attn_torch`'s backward allocates `[B,S,S_kv,D]` (2048 GiB at 128K/CP=8) | major | shape-only repro: fwd+bwd at B=1, S=16384, S_kv=131072, topk=128, D=512 | **FIXED** locally by `index_select`; still **open upstream**, and NOT gfx942- or CP-specific |
| Colocate weight sync breaks under `expandable_segments` with an unattributable HIP errno | major | A/B in `logs/cp8_seq131072_r4_train.log` vs `..._r5_train.log`; `reproducer/probe_hip_ipc.py` | root-caused, avoided by config; upstream should assert or document |
| AITER custom all-reduce vs SGLang colocate memory saver | minor | `logs/cp8_seq131072_r1_train.log:1825`; `reproducer/probe_hip_ipc.py` | worked around |
| Ray GCS collision between containers sharing `--network=host` | minor | `verify/run_4layer_1node_fp8recheck.log` (dies in 26 s with a session-name mismatch naming Redis) | preflight check added to the launcher |

**Carried over from the referenced run, NOT investigated here** (out of scope — this case's question
was 128K + CP):

- The ~7 nat Megatron-vs-SGLang rollout log-probability gap on this image, tied to SGLang's
  unconditional FP8 KV cache for DeepseekV4. Unchanged.
- ~~The non-fatal tilelang FP8-encoding warning (`float8_e4m3fnuz` vs `float8_e4m3fn`)~~ —
  **resolved** by the D6 base switch: `sglang_dsv4_fp8_fnuz.patch` ships in `-20260812-fada`, and the
  FP8 audit goes from 309 mismatch lines to 0. See the FP8 audit table below.

---

## How the answer was reached — 9 debug rounds

| # | change | outcome |
|---|---|---|
| 0 | source study only, no GPU | CP is natively supported for DSV4 (`ops/cp_utils.py`); `--allgather-cp` is **mandatory** (`arguments.py:3174`); 128K must come from **data**, not a flag |
| 1 | first run, `--ipc=host` | CP=8 resolved correctly; SGLang died in AITER custom all-reduce |
| 2 | `--ipc=private` (match the validated container) | identical failure → IPC namespace was not the cause |
| 2b | `probe_hip_ipc.py` | IPC export is healthy on this host → the *buffer* is the problem |
| 3 | `--sglang-disable-custom-all-reduce` | rollout engine starts; now dies in the weight sync at `_new_shared_cuda` |
| 4 | KV pool 0.7→0.25, cudagraph bs≤8 | trainer got 137 GB instead of 7.8 GB, **identical** failure → memory was not the cause either |
| 5 | drop `expandable_segments` | weight sync works; forward sparse-MLA kernel then fails to build (85488 B > 65536 B LDS) |
| 5b | `probe_fwd_kernel_lds.py` | 64 local heads cannot build on gfx942 |
| 6 | CP=4 / TP=2 | 3/3 steps at 131072 tokens, 75.2 GiB peak, clean |
| 7 | `block_H` patch, retry CP=8 | compiles, weight sync + full forward pass OK, then OOMs at 2048 GiB. **Mis-attributed** to TP=1 |
| 8 | revert the patch | correct given what was known; the premise was about to be disproven |
| 9 | root-cause the gather; re-apply `block_H` + add `index_select` | **CP=8 works: 3/3 steps @ 131072, 126–136 s/step** (peak read 79.42 GiB here, later shown to be a `used_GB` sampling artefact — the real high-water mark was 107.27 GiB, and rounds 10–16 took it to 52.45) |

Two rounds went to a setting imported from the Primus reference (`expandable_segments`), and two more
to mis-attributing the 2048 GiB OOM. Each had a transferable cause:

- HIP returns `invalid argument` for a bad IPC handle, a failed IPC mapping, *and* genuine
  exhaustion — so it reads as an OOM. When an IPC call returns `invalid argument`, check **what kind**
  of memory the pointer came from before assuming there is not enough of it.
- Round 7 ruled out the reference attention path because no log line said
  `falling back to the reference implementation` — but only one of the two branches reaching that path
  is instrumented. **A missing log line is not evidence.**

Full narrative: [`cases/dsv4_4layer_128k_cp8/hypothesis_ledger.md`](cases/dsv4_4layer_128k_cp8/hypothesis_ledger.md).

---

## Primus reference → Miles mapping

| Primus knob | Miles equivalent | status |
|---|---|---|
| `PRIMUS_CP=8 / TP=1 / ETP=1 / EP=8` | `--context-parallel-size 8 --tensor-model-parallel-size 1 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1` | ✅ **adopted exactly**, after two gfx942 fixes |
| `PRIMUS_SEQ_LENGTH=131072` | 128K prompt dataset (Miles takes the length from data) | adopted |
| `PRIMUS_TOTAL_LAYERS=4`, `COMPRESS_RATIOS=[0,0,4,128,0]` | image default `(0 0 4 128)` — dense, dense, CSA 4, HCA 128 | already default; all three V4 attention branches exercised |
| `PRIMUS_DSA_BWD_NUM_STAGES=1` (gfx942 LDS, **backward**) | none needed — `bwd_within_shared_mem` auto-retiles | no action |
| `PRIMUS_INDEXER_TRITON_FULL=1` | `V4Indexer` + tilelang indexer is the default path | already default |
| `FUSED_LINEAR_CE=1`, `FUSED_CE_CHUNK=4096` | `--log-probs-chunk-size` | held as a lever; not needed (CP already cuts local logits) |
| `PRIMUS_OVERLAP_PARAM_GATHER=false` | n/a — that overlap is not enabled here | n/a |
| `PRIMUS_SHARD_HEADS=false` | n/a | n/a |
| `PRIMUS_LR=1.0e-6` | `--lr 1e-6` | already default |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | **NOT adopted — breaks colocate weight sync** | **rejected (D4)** |
| `HSA_NO_SCRATCH_RECLAIM=0` | **NOT adopted — image ships 1** | rejected (D4) |

Two differences that change what the numbers mean: the Primus reference is **SFT** (Megatron alone,
no colocated rollout engine sharing the GPUs), and it uses **8 experts top-1 from random init**
versus this run's **256 experts top-6 from a converted checkpoint**. Its "~42 GB peak, ~3.8 s/step"
therefore does not transfer, and no comparison against it is made.

---

## Memory: 107.27 → 52.45 GiB at CP=8 (−51%)

Two earlier rounds of analysis on **CP=4** produced seven changes
([report 1](DSV4-128K-完整显存分析报告.md), [report 2](DSV4-128K-显存分析报告-第二轮.md),
[reproduction guide](DSV4-128K-显存优化-复现指南.md)). Those numbers do not transfer to CP=8 by
inspection: CP=8 halves `S_local` to 16384 and drops TP from 2 to 1, so every allocation that scales
with `S_local` shrinks while each GPU now holds all 64 attention heads and an unsharded vocabulary.
Each group was therefore re-measured on CP=8 by re-running the 3-step 128K case with exactly one more
group enabled. Full detail: [`DSV4-128K-显存优化-CP8验证.md`](DSV4-128K-显存优化-CP8验证.md).

| group | what it changes | CP=8 | CP=4 | verdict |
|---|---|---|---|---|
| `metrics` | report `max_allocated_GB` / `max_reserved_GB` | — | — | **prerequisite** — see below |
| `intmax` | refuse + log a reference backward that overflows a 32-bit element count | — | — | **crash fix**, fixes backward choice so runs are comparable |
| `logits` | drop Megatron's full-vocab fp32 upcast (needs the `cp_utils` `.float()` or all 8 ranks deadlock) | **−9.27** | −7.89 | keep |
| `temperature` | divide by `rollout_temperature` after chunking, not on the full `[T, V]` | **−1.50** | −5.41 | keep |
| `rms_norm` | `F.rms_norm` instead of a manual `q.float()` normalisation | **−9.50** | −6.11 | keep |
| `recompute_gc` | collect after each checkpointed backward | **−18.87** | −9.88 | keep |
| `logprob_gc` | collect around `forward_only` / after `log_probs` | **−15.68** | −23.55 | keep |
| | | **−54.82 (−51.1%)** | −49.32 (−46.7%) | |
| `expandable` | run the training step with `expandable_segments` on, then release | reserved **−2.88** | reserved −7.48 | keep, off by default |

All five savings hold on CP=8; none had to be dropped, and the total is proportionally better than on
CP=4. Peak `reserved` fell 130.28 → 63.18 GiB. Step time is unchanged (134.8 → ~125 s), and the eight
ranks end up at *identical* peaks (52.45 GiB each), i.e. the peak is no longer set by one unlucky rank.

Three findings worth carrying forward, because each contradicts something that looked obvious:

- **The measurement had to be fixed first.** `used_GB` is sampled at phase boundaries and Ray folds
  duplicate log lines across actors, so a max over surviving lines is luck: it read 79.42 GiB where
  the high-water mark is 107.27 GiB. Round 1 saw the same effect on CP=4 (83 vs 105.49 GiB). Rank
  coverage is now asserted (8/8) and a run that reports fewer ranks is flagged as a lower bound.
- **`temperature` did not stop working on CP=8 — it stopped setting the peak.** It still saves 4.7 GiB
  on rank1 and 6.1 GiB on rank7, exactly the shape report 1 described. But `logits` has by then
  levelled all eight ranks, so the *maximum* only moves by the 1.50 GiB that the other six ranks save.
- **`gc.collect(0)` is not enough, though the garbage is freshly allocated.** gen=0 peaks at
  63.63 GiB versus gen=2's 52.45 GiB — 11.18 GiB worse — for about 2 s/step. A reference cycle is
  only collectable when *every* object in it is in the generation being scanned, and survivors of a
  gen-0 pass are promoted out of its reach. `MEGATRON_RECOMPUTE_BACKWARD_GC_GEN` defaults to 2.

`rms_norm` is the only change that touches numerics, and the 4-layer prune cannot vouch for it (reward
is 0 by construction, so loss and every gradient are 0 either way).
[`porting/verify_rms_norm_equiv.py`](porting/verify_rms_norm_equiv.py) compares both writings against
an fp64 reference on the real shapes: at bf16 their errors are *identical* (5.866e-03 forward,
4.476e-03 backward), at fp32 both sit near 1.5e-07. That check also corrects the reproduction guide on
one point — both writings produce a **bf16** `q.grad`, so the saving is the intermediate fp32 copy that
backward holds, not the gradient dtype.

### `expandable_segments`, and why order of application matters

The seven groups above all reduce **allocated** memory — live tensors. A separate axis is the
allocator's own waste: `reserved − allocated`. Scoping `expandable_segments` to the training step
(bug #2 above) attacks exactly that, and leaves allocated bit-identical.

| config | allocated | reserved | fragmentation gap | step time |
|---|---|---|---|---|
| baseline | 107.27 | 130.28 | 23.01 | 134.8 / 124.7 s |
| baseline + expandable | 106.96 | **120.39** | 13.43 | **111.9 / 111.3 s** |
| all 7 groups | 52.45 | 63.18 | 10.73 | 124.8 / 124.8 s |
| all 7 + expandable | **52.45** | **60.30** | 7.85 | **115.9 / 116.1 s** |

**The same change is worth 3.4× more on the unoptimised baseline than after the fact**: −9.89 GiB of
reserved when added to the baseline, −2.88 GiB when added on top of the seven groups. Fragmentation
scales with how much churn the allocator sees, and the two GC groups have already removed most of it,
so there is far less slack left to reclaim. The two axes are not additive — if they were,
130.28 − 9.89 − 67.10 would give 53.29 against an actual 60.30, an overlap of about 7 GiB.

A telling detail: the seven groups *without* expandable already reach a smaller fragmentation gap
(10.73) than expandable alone does (13.43). **If only one of the two can be done, do the seven
groups** — they take the device-side requirement down 51.5% versus expandable's 7.6%.

Enable both anyway, for a reason unrelated to capacity: expandable cuts **9–23 s off the step time**
on either footing (134.8 → 111.9 on the baseline, 124.8 → 115.9 optimised), which none of the seven
groups deliver. Off by default because it depends on toggling the allocator at runtime; re-run
[`porting/verify_expandable_window.py`](porting/verify_expandable_window.py) before enabling it on a
different build.

---

## Full 43-layer model, 4 nodes, CP=8 / TP=1 / PP=4 — PASS

Everything above is the 4-layer prune on one node. The unified image was then taken to the **full
43-layer** DeepSeek-V4-Flash across **4 × 8 MI308X (32 GPUs)** at the requested
**CP=8 / TP=1 / PP=4 / EP=8**. Detail: [`DSV4-43层-4节点-CP8验证.md`](DSV4-43层-4节点-CP8验证.md).

| | |
|---|---|
| steps | **1 / 1**, exit 0 |
| resolved layout | CP=**8** · TP=**1** · PP=**4** · DP=1 · EP=8/ETP=1 · `allgather_cp` True |
| layer split | **11 + 11 + 11 + 10 = 43** |
| peak per GPU | **77.81 GiB allocated / 81.82 reserved** of 192 (41%) |
| rank coverage | **32 / 32** |
| OOM / NaN / RCCL timeout / LDS | **0 / 0 / 0 / 0** |
| weight syncs | 64 |
| wall time | 5703 s (95 min); weight sync 997 s, log_probs 840 s, train 1780 s |

Megatron's own resolved dump confirms the layout independently:
`world size: 32, data-parallel size: 1, context-parallel size: 8, tensor-model-parallel size: 1,
pipeline-model-parallel size: 4`.

**What this does not show.** `rollout/rewards`, `train/loss` and `train/grad_norm` are all 0.0. Under
GRPO a group whose samples earn identical rewards has zero advantage and therefore no gradient, and
`debug_minimal` supplies 4 prompts × 8 samples. So this proves the pipeline, the parallel layout and
the memory envelope at full scale — not convergence. `run_full_4node.sh` now prints that caveat
alongside its PASS so the result cannot be misread.

**43 is prime**, so no pipeline size divides it. Megatron skips its divisibility assert only when
`--decoder-first/last-pipeline-num-layers` is given; 11/11/11/10 matches Miles' own 32-GPU preset,
with one fewer layer on the last stage to offset the LM head it also carries. TP=1 gives every GPU all
64 attention heads, which is only buildable on gfx942 because of the image's `block_H` patch — the
launcher asserts that patch is present before committing to a 95-minute run.

`run_full_6node.sh` had never actually submitted a job (only its preflight had ever run), so its whole
back half was unverified code. Converting it to 4 nodes surfaced five real defects, all now fixed in
both scripts: a lost quote that made every container exit instantly (`bash -lc sleep infinity`), three
levels of shell quoting that truncated the `--extra-env-vars` JSON, Ray's **agent** ports colliding
with a neighbouring container's default-port Ray (the cluster looks perfectly healthy and only
`ray job submit` fails, with a message that never mentions ports), a `--sglang-mem-fraction-static`
floor of 0.3952 set by the full model's own weights, and two verdict bugs that reported a spurious
RCCL timeout and a 0/32 rank coverage on a clean run. Each is written up in the document above.

---

## Unified image + one-click launchers

Built FROM the LOCKED base, which is **unchanged**:
**`miles-dsv4-flash:rocm7.14-gfx942-cp8-20260821-fada`**.
Ships the two CP=8 framework patches **and** the memory patches above, all applied at build time and
gated by sentinel checks. Full provenance: [`porting/unified_image.md`](porting/unified_image.md).

| added | version / pin | why |
|---|---|---|
| **apex** | `1.9.0+rocm7.14.0a20260625` (ROCm nightly wheel, `--no-deps`) | supplies `fused_weight_gradient_mlp_cuda`, so `--no-gradient-accumulation-fusion` is no longer forced |
| mori bitcode warm-up | `MORI_PRECOMPILE=1` | makes `import apex` work (it reaches aiter→mori) |
| **causal-conv1d** | built **1.6.2.post1** from `~=1.5` | Miles prepare hooks |
| **primus_turbo** | commit `edc8d2c…`, ~30 min | requested |
| **flash-linear-attention** | 0.4.2 → **0.5.2** | requested |
| misc | `expecttest nltk tyro torchdata plotext webdataset pyrsmi` | Miles tooling |

**`docker build` cannot produce this image on its own** — it gives the build container no GPU, and
`import apex` (→ aiter → mori device bitcode), apex's fused-extension JIT, and `primus_turbo`'s import
check all need `/dev/kfd`. [`porting/build_with_gpu.sh`](porting/build_with_gpu.sh) runs the identical
steps in a GPU-visible container and commits, so the JIT artifacts are baked in;
[`porting/Dockerfile`](porting/Dockerfile) is the portable record with those three steps deferred.

**Post-release fix — `docker commit` broke Entrypoint/Cmd/Shell.** `docker run <image> /bin/bash`
failed with `/bin/bash: /bin/bash: cannot execute binary file`, because commit snapshots the
*running container's* config: the builder's `--entrypoint bash … -lc 'sleep infinity'` became
`Entrypoint=["bash"]` / `Cmd=["-lc","sleep infinity"]`, so a user's `/bin/bash` landed as bash's
*script* argument and bash read the ELF binary as a script. `Shell` was lost the same way. Every
earlier check had used `docker exec … bash -lc` on a running container, which never goes through the
image Entrypoint, so nothing caught it. Fixed by
[`porting/Dockerfile.metadata_fix`](porting/Dockerfile.metadata_fix) (metadata only, no new
filesystem layer) and by `build_with_gpu.sh` now passing the three `--change` flags at commit time
plus asserting `docker run <tag> /bin/bash -c …` before declaring success. Image id `8adcdc949746` → `4d31ed584c46` → **`6eb08f2c5723`** (the last also drops the reverted patch).

**Out-of-the-box verification** — fresh container, one command
([`porting/verify/run_4layer_1node.log`](porting/verify/)):

```
steps completed 3/3 · context_parallel_size 8 · allgather_cp True
trained seq len 131072 (from max total_lengths 130599) · 8 weight syncs
step time ~125 s · peak 52.45 GiB allocated / 63.18 GiB reserved · 8/8 ranks
OOM/NaN/LDS 0/0/0
PASS — 4-layer DeepSeek-V4-Flash trains at 131072 tokens with CP=8
```

The launcher's own preflight reported `apex fused grad kernel: present`.

Launchers, both baked in:

| script | where it runs | status |
|---|---|---|
| `run-dsv4-4layer-1node` | inside the container | **verified end to end at both CP=4 and CP=8** |
| `run_full_4node.sh` | on the host (drives `docker` + `ssh`) | **verified end to end**: full 43-layer model, 32 GPUs, CP=8/TP=1/PP=4, 1/1 step, 0 failures |
| `run_full_6node.sh` | on the host | **preflight/bring-up logic only** — still never executed at 6 nodes. It now carries the five fixes found by running the 4-node variant (quoting, job staging, Ray agent ports, model_name, asset paths), but that back half remains unverified at PP=6. Prefer `run_full_4node.sh`, or run this one with `--dry-run` first. |

Both refuse to start rather than contend for busy GPUs or run against a mismatched image, and the
1-node verdict checks CP-active / real-sequence-length / completion rather than mere liveness.

## Knowledge candidates

| file | finding | confidence |
|---|---|---|
| [`dsv4-cp8-128k-on-gfx942-two-fixes.md`](knowledge_candidates/dsv4-cp8-128k-on-gfx942-two-fixes.md) | CP=8/128K on gfx942 needs BOTH an LDS fix and a `torch.gather` backward fix; plus the diagnostic trap that a silent code path leaves no log line | high |
| [`expandable-segments-breaks-colocate-ipc-weight-sync.md`](knowledge_candidates/expandable-segments-breaks-colocate-ipc-weight-sync.md) | `expandable_segments` makes allocations non-IPC-exportable, silently breaking colocate weight sync; presents as an OOM | high |
| [`apex-primus-turbo-install-on-rocm7.14-gfx942.md`](knowledge_candidates/apex-primus-turbo-install-on-rocm7.14-gfx942.md) | Verified apex/causal-conv1d/primus_turbo recipe on ROCm 7.14 gfx942, and why `docker build` cannot run it (no GPU → aiter/mori). **Closes the KB's open TODO for apex.** | high |
| [`rl-training-memory-measurement-and-gc-on-rocm.md`](knowledge_candidates/rl-training-memory-measurement-and-gc-on-rocm.md) | Why Ray-folded `used_GB` samples make peak memory unmeasurable, and why reference-cycle GC (gen 2, not gen 0) is worth 34 GiB in a colocated RL trainer | high |

Not promoted into `knowledge_base/` — that is a separate, human-gated step.

---

## Reproducing this run

```bash
cd exp/miles_spec_20260817_1430/cases/dsv4_4layer_128k_cp8
./launch.sh my_run_tag        # drives the container on f2b9 over ssh
```

`launch.sh` refuses to start unless the container's image matches the LOCKED tag and all 8 GPUs are
idle. Artifacts: `logs/<tag>_{driver,train}.log`, `logs/<tag>_metrics.json`.

Inside the unified image the same run is one command, and the memory A/B is reproducible per group:

```bash
run-dsv4-4layer-1node --cp 8 --steps 3          # 52.45 GiB, 3/3 steps

python3 /opt/miles-rocm/patches/apply_mem_opts.py --status
python3 /opt/miles-rocm/patches/apply_mem_opts.py --set metrics intmax   # baseline, 107.27 GiB
python3 /opt/miles-rocm/patches/apply_mem_opts.py                        # all groups
python3 /opt/miles-rocm/tools/verify_rms_norm_equiv.py                   # numerics
```

`--set` restores every file to pristine before applying the given subset. Do not revert a single group
in place: `model.py` carries edits from both `logits` and `logprob_gc` and the backup is of the
pristine file, so reverting one would silently take the other with it.

The 128K dataset is rebuildable:

```bash
PYTHONPATH=/opt/miles python3 build_128k_prompts.py \
  --tokenizer /apps/mingjiel/FADA/models/DeepSeek-V4-Flash-FP8-4layer-bf16 \
  --source /apps/mingjiel/data/dapo-math-17k/dapo-math-17k.jsonl \
  --out /apps/mingjiel/data/dapo-math-128k/dapo-math-128k.jsonl \
  --rows 8 --target-tokens 130200 --tolerance 400
```

Framework sources modified, all recorded as scripted patches rather than loose edits:

| patch | files | purpose |
|---|---|---|
| [`apply_fwd_block_h.py`](porting/apply_fwd_block_h.py) | `tilelang_sparse_mla_fwd.py` | LDS fix — lets CP=8 compile |
| [`apply_ref_gather_fix.py`](porting/apply_ref_gather_fix.py) | `torch_sparse_mla.py` | 2048 GiB gather backward |
| [`apply_mem_opts.py`](porting/apply_mem_opts.py) | `memory_utils.py`, `cp_utils.py`, `model.py`, `actor.py`, `logit_processors.py`, `deepseek_v4.py`, `tilelang_sparse_mla.py`, Megatron `random.py` | measurement + INT_MAX guard + the five memory savings |
