# DeepSeek-V4-Flash on gfx942: the complete image, from the locked base to CP=8 at 128K.
#
#   docker build -t miles-dsv4-flash:rocm7.14-gfx942-cp8mem-20260825-fada -f Dockerfile .
#
# Produces, on one 8x MI308X node, a verified:
#   run-dsv4-4layer-1node --cp 8   ->  3/3 iterations at 131072 tokens,
#                                      52.45 GiB allocated / 63.18 GiB reserved per GPU,
#                                      ~125 s/step, 0 OOM / NaN / LDS overflow, 8/8 ranks reporting
#   + MILES_EXPANDABLE_SEGMENTS_DURING_TRAIN=1
#                                  ->  same, reserved 60.30 GiB, ~116 s/step
#   run-dsv4-4layer-1node          ->  same at CP=4 (the default)
#
# Peak figures are torch `max_memory_allocated` high-water marks with Ray log dedup off. Earlier
# revisions of these notes quoted 79.42 GiB, which was a max over sampled `used_GB` values and is not
# comparable -- the same configuration's true baseline high-water mark is 107.27 GiB. See section 3.
#
# ┌─ PREFER ./build_with_gpu.sh ON A MACHINE WITH GPUs ────────────────────────────────────────┐
# │ `docker build` gives the build container NO GPU devices, and three steps here need one:     │
# │   * `import aiter` probes the GPU arch via rocminfo and raises "Get GPU arch from rocminfo   │
# │     failed" without /dev/kfd. `import apex` reaches it through                              │
# │     apex/transformer/functional/fused_rope.py, so even verifying apex needs a GPU.          │
# │   * mori's device bitcode (libmori_shmem_device.bc), which aiter's flydsl path imports, is   │
# │     JIT-built per architecture. Without it `import apex` dies with                          │
# │     "libmori_shmem_device.bc not found".                                                    │
# │   * apex's fused_weight_gradient_mlp_cuda JITs on first import (~27 s).                      │
# │ Those three are DEFERRED here (marked below), so a GPU-less build succeeds and produces a    │
# │ functionally equivalent image that JITs on first use. build_with_gpu.sh runs the identical   │
# │ steps in a GPU-visible container and commits, baking the artifacts in.                       │
# └────────────────────────────────────────────────────────────────────────────────────────────┘
#
# torch is NEVER pip-installed: PyPI torch wheels are CUDA-only. Every step uses --no-deps or is
# pure-python, and the gate asserts torch is still the base's 2.11.0+rocm7.14.0 build.
#
# Layer order is by cost, cheapest first, so an edit to the package list does not invalidate the
# ~30-minute primus_turbo build.

# ---------------------------------------------------------------------------------------------
# BASE — `-20260812-fada`, not `-20260804`. The -fada tag carries three SGLang patches from the
# referenced run (miles_accuracy_20260809_0202), and one of them matters here:
#
#   sglang_dsv4_fp8_fnuz.patch  removes the FP8 encoding mismatch that -20260804 logs as ~309
#                               "Buffer dtype mismatch for parameter q: expected
#                               torch.float8_e4m3fnuz, got torch.float8_e4m3fn" lines wrapped in
#                               "Exception ignored in ..._check_buffer_dtype". Non-fatal there (the
#                               dtype CHECK was skipped, not the kernel) but noisy enough to bury
#                               real signal. Measured: 309 -> 0 lines after the switch.
#   sglang_moe_weight_replay_lifecycle.patch
#   sglang_skip_incremental_missing_warning.patch
#
# Dependency state is identical between the two tags (apex / causal_conv1d / primus_turbo absent,
# fla 0.4.2), so nothing below changes because of the base switch.
# ---------------------------------------------------------------------------------------------
FROM amdagi/miles-dsv4-flash:rocm7.14-gfx942-20260812-fada

SHELL ["/bin/bash", "-lc"]

# gfx942 (MI300/MI308). Set once so every source build below targets the right arch.
ENV GPU_ARCHS=gfx942 \
    PYTORCH_ROCM_ARCH=gfx942 \
    HIP_ARCHITECTURES=gfx942 \
    HCC_AMDGPU_TARGET=gfx942 \
    MAX_JOBS=32

# Record the torch build we must not disturb, so the gate can diff against it.
RUN python3 -c "import torch; open('/opt/torch_baseline.txt','w').write(torch.__version__)" \
 && cat /opt/torch_baseline.txt

# =============================================================================================
# 1. Dependencies
# =============================================================================================

# ---------------------------------------------------------------------------------------------
# 1.1 misc python packages. fla is upgraded 0.4.2 (in base) -> 0.5.2.
# ---------------------------------------------------------------------------------------------
RUN pip install --no-cache-dir 'flash-linear-attention==0.5.2' \
 && pip install --no-cache-dir expecttest nltk tyro torchdata plotext webdataset pyrsmi

# ---------------------------------------------------------------------------------------------
# 1.2 apex — prebuilt ROCm wheel from AMD's nightly index. Supplies
# fused_weight_gradient_mlp_cuda, i.e. what --gradient-accumulation-fusion needs; without it
# Megatron fails in ColumnParallelLinear unless every run passes --no-gradient-accumulation-fusion,
# and the base logs "Apex is not installed. Falling back to Torch Norm".
#
# --no-deps is REQUIRED, not defensive: without it pip resolves apex's torch requirement against
# PyPI and replaces the image's ROCm torch with a CUDA build.
# ---------------------------------------------------------------------------------------------
RUN pip install --no-cache-dir --no-deps \
      --index-url https://rocm.nightlies.amd.com/whl-multi-arch --pre \
      "apex==1.9.0+rocm7.14.0a20260625"

# DEFERRED (needs a GPU): this JITs on first import (~27 s), and the mori device bitcode it reaches
# through aiter is built per architecture. Non-fatal here so a GPU-less build still succeeds.
RUN python3 -c "import fused_weight_gradient_mlp_cuda; print('fused grad extension OK (baked)')" \
      || echo "fused grad extension DEFERRED to first runtime import (no GPU during docker build)"

# ---------------------------------------------------------------------------------------------
# 1.3 causal-conv1d — required by Miles' prepare hooks; V4 itself does not use it.
# FORCE_BUILD avoids picking up a CUDA wheel. `~=1.5` means <2.0, so the resolved version drifts
# with PyPI: 1.6.2.post1 on 2026-08-19, 1.7.0 on 2026-08-21. Pin it if you need reproducible images.
# ---------------------------------------------------------------------------------------------
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    pip install --no-cache-dir --no-build-isolation 'causal-conv1d~=1.5' \
 && python3 -c "import causal_conv1d; print('causal_conv1d', getattr(causal_conv1d, '__version__', 'present'))"

# ---------------------------------------------------------------------------------------------
# 1.4 primus_turbo — source build, ~30 min, pinned commit. The source tree is removed in the same
# layer so it does not inflate the image.
#
# The `cd /` before the import check is load-bearing: importing from inside the source directory
# shadows the installed package with the un-built source tree and fails with "_C not found".
# ---------------------------------------------------------------------------------------------
RUN git clone --recursive https://github.com/AMD-AGI/Primus-Turbo.git /workspace/Primus-Turbo \
 && cd /workspace/Primus-Turbo \
 && git checkout edc8d2ccb0be4888e80ee7c6e765fd3956026a32 \
 && git submodule update --init --recursive \
 && pip install --no-build-isolation --no-deps --no-cache-dir . -v \
 && cd / \
 && { python3 -c "import primus_turbo.pytorch; print('turbo OK')" \
      || echo "primus_turbo built; import DEFERRED (needs a GPU)"; } \
 && rm -rf /workspace/Primus-Turbo

# =============================================================================================
# 2. The two gfx942 patches that make CP=8 at 128K work
#
# Neither is sufficient alone, and an image with only one is WORSE than an image with neither:
#
#   block_H only     -> CP=8 compiles, then OOMs 20 minutes in at 2048 GiB
#   ref_gather only  -> CP=8 cannot compile at all
#
# so the gate below requires both sentinels, and run-dsv4-4layer-1node --cp 8 does too.
#
# ---- 2a. block_H: the LDS half -------------------------------------------------------------
# tilelang_sparse_mla_fwd stages Q/O in LDS as [H_per_block, D], with H_per_block = padded_H for
# heads <= 64. At D=512 bf16 and 64 heads, Q_shared alone is 65536 B -- the entire gfx942 budget,
# leaving nothing for KV_shared/S_shared/Lse_shared:
#     RuntimeError: Requested dynamic shared memory 85488 exceeds device limit 65536
# num_stages cannot help; it only multiplies KV_shared, and every _FALLBACK_TILINGS entry already
# uses 1 (the HIP branch forces it). CP=8 is what produces 64 local heads: on one 8-GPU node
# TP*PP*CP*DP = 8, so CP=8 forces TP=1.
#
# The kernel ALREADY blocks over heads -- REPLICATE_H folds head blocks into the grid's x axis and
# every head index goes through the H0 offset -- only the block size was hardwired to 64. Exposing
# it as `block_H` is a TILING change, not a numerics change: DSV4 attention is MQA (KV_shared has no
# head axis) and the online-softmax state (m_i, sumexp, alpha) is already per-head, so head blocks
# are independent. Verified: heads 32/16/8 keep byte-identical tilings; heads=64 builds at
# (1, 16, 64, block_H=32).
#
# ---- 2b. ref_gather: the OOM half ----------------------------------------------------------
# sparse_attn_torch collected the top-k KV rows with
#     torch.gather(kv.unsqueeze(1).expand(B, S, S_kv, D), 2, safe_idx...)
# The forward is cheap (expand is a view, gather reads it strided). The backward is a scatter_add
# into a freshly ZEROED tensor shaped like gather's INPUT -- [B, S, S_kv, D], which is exactly
# 2048 GiB at B=1, S=16384, S_kv=131072, D=512. index_select on a flattened KV returns the same
# values and accumulates into [B*S_kv, D] = 128 MiB. Measured: fwd+bwd peak OOM -> 18.17 GiB, dq
# bit-identical. The fp32 cast is also hoisted (it was written twice, ~4 GiB per .float()).
#
# Why this only shows at CP=8, which is what made it hard to attribute: not TP-related and not
# caused by 2a. CP=8 halves S_local to 16384, and tilelang_sparse_mla._reference_is_affordable plans
# for 8*B*S*topk*D*4 bytes; halving S halves the estimate, so the two dense+SWA layers (topk=128,
# not the 640 of compressed layers) drop to ~34 GB, slip under the free-memory budget, and take the
# reference path FOR THE FIRST TIME. CP=4's estimate stays over budget and always used the kernel --
# which is exactly why CP=4 looked healthy.
#
# And that selection is SILENT: of the two branches into sparse_attn_torch, only
# _backward_kernel_fits logs "falling back to the reference implementation". Grepping the log for a
# fallback message finds none and wrongly exonerates the path.
#
# Patch provenance: /apps/mingjiel/dsv4mem, where the gather root cause was found.
# =============================================================================================
COPY apply_fwd_block_h.py    /opt/miles-rocm/patches/apply_fwd_block_h.py
COPY apply_ref_gather_fix.py /opt/miles-rocm/patches/apply_ref_gather_fix.py

RUN python3 /opt/miles-rocm/patches/apply_fwd_block_h.py \
      --diff-out /opt/miles-rocm/patches/miles_dsv4_fwd_block_h.patch \
 && python3 /opt/miles-rocm/patches/apply_ref_gather_fix.py \
      --diff-out /opt/miles-rocm/patches/miles_dsv4_ref_gather.patch \
 && K=/opt/miles/miles_plugins/models/deepseek_v4/ops/kernel \
 && grep -q block_H      "${K}/tilelang_sparse_mla_fwd.py" \
 && grep -q index_select "${K}/torch_sparse_mla.py" \
 && python3 -c "import ast; [ast.parse(open(f).read()) for f in ['${K}/tilelang_sparse_mla_fwd.py','${K}/torch_sparse_mla.py']]; print('both patched sources parse')"

# =============================================================================================
# 3. Memory optimisations — 107.27 -> 52.45 GiB allocated, 130.28 -> 60.30 reserved, at CP=8 / 128K
#
# Eight groups: six savings, one measurement prerequisite, one crash fix. Each was verified
# independently on CP=8 by re-running the 3-step 128K case with exactly one more group enabled
# (nine runs total; see DSV4-128K-显存优化-CP8验证.md):
#
#   metrics       report max_allocated_GB / max_reserved_GB      prerequisite, not an optimisation
#   intmax        guard + log the sparse-MLA backward choice     crash fix, not an optimisation
#   logits        no full-vocab fp32 upcast                      -9.27 GiB
#   temperature   divide after chunking, not on the full [T, V]  -1.50 GiB
#   rms_norm      F.rms_norm instead of a manual q.float()       -9.50 GiB
#   recompute_gc  collect after each checkpointed backward       -18.87 GiB
#   logprob_gc    collect around forward_only / log_probs        -15.68 GiB
#   expandable    expandable_segments, scoped to the train step  reserved -2.88 GiB, ~9 s/step
#
# The first seven reduce ALLOCATED memory (live tensors). `expandable` is a different axis: the
# allocator's own waste, reserved - allocated. It leaves allocated bit-identical and is worth 3.4x
# more BEFORE the others are applied (-9.89 GiB of reserved on the raw baseline vs -2.88 after),
# because fragmentation scales with allocator churn and the two GC groups have already removed most
# of it. It is OFF unless MILES_EXPANDABLE_SEGMENTS_DURING_TRAIN=1, since it needs a runtime
# allocator toggle; tools/verify_expandable_window.py checks that premise plus page release.
#
# `metrics` comes first because without it the rest cannot be measured: the script used to report a
# max over `used_GB`, an instantaneous sample taken at six phase boundaries and then deduplicated
# across actors by Ray, so the reported peak depended on which ranks' lines survived. It read
# 79.42 GiB where the true high-water mark is 107.27 GiB. That is also why the LABEL below now
# carries the max_allocated figure.
#
# Three of these are prerequisites rather than savings, and each fails loudly if dropped:
#   - `logits` must ship with the cp_utils `.float()`. With bf16 logits, a CP rank holding no
#     response token contributes bf16 while the others contribute fp32; the byte counts disagree and
#     the all_reduce deadlocks all 8 ranks until the watchdog fires.
#   - `intmax` refuses the reference backward when its scatter_add would exceed a 32-bit element
#     count. At CP=8 the topk=640 and topk=1152 layers need 5.37e9 / 9.66e9 elements, so without the
#     guard they hit "cub sort does not support sorting more than INT_MAX elements" -- but only
#     sometimes, because whether the reference path is taken depends on free memory at the time.
#     It also logs which backward each shape picked; the two differ by tens of GiB, so without that
#     line two runs are not comparable.
#   - `recompute_gc` is inert unless MEGATRON_RECOMPUTE_BACKWARD_GC_GEN is set. run_4layer_1node.sh
#     exports it as 2 by default and lets the environment override it.
#
# `expandable` exists because expandable_segments cannot simply be left on process-wide: the weight
# sync's IPC bucket becomes unusable (the export succeeds, the consumer's hipIpcOpenMemHandle returns
# hipErrorInvalidValue) AND torch_memory_saver's offload silently no-ops, since its shim hooks only
# hipMalloc/hipFree while expandable segments use hipMemCreate/hipMemMap. Scoping the setting to the
# training step puts weights, gradient buffers and the bucket outside the window.
#
# `rms_norm` is the only one that touches numerics. verify_rms_norm_equiv.py compares both writings
# against a fp64 reference on the real shapes: at bf16 the two have IDENTICAL error (5.866e-03 fwd,
# 4.476e-03 bwd) and at fp32 both sit at ~1.5e-07. The 4-layer prune cannot show this on its own --
# its reward is 0 by construction, so loss and all gradients are 0 either way.
# =============================================================================================
COPY apply_mem_opts.py              /opt/miles-rocm/patches/apply_mem_opts.py
COPY verify_rms_norm_equiv.py       /opt/miles-rocm/tools/verify_rms_norm_equiv.py
COPY verify_expandable_window.py    /opt/miles-rocm/tools/verify_expandable_window.py

RUN python3 /opt/miles-rocm/patches/apply_mem_opts.py \
 && python3 /opt/miles-rocm/patches/apply_mem_opts.py --status \
 && M=/opt/miles/miles \
 && grep -q max_allocated_GB              "${M}/utils/memory_utils.py" \
 && grep -q _half_precision_output_kwargs "${M}/backends/megatron_utils/model.py" \
 && grep -q 'full_resps, dim=0).float()' "${M}/backends/training_utils/cp_utils.py" \
 && grep -q _RECOMPUTE_BACKWARD_GC_GEN    /opt/Megatron-LM/megatron/core/tensor_parallel/random.py \
 && grep -q 'functional.rms_norm'         /opt/miles/miles_plugins/models/deepseek_v4/deepseek_v4.py \
 && grep -q expandable_segments_during_training "${M}/utils/memory_utils.py" \
 && grep -q 'with expandable_segments_during_training()' "${M}/backends/megatron_utils/actor.py"

# =============================================================================================
# 4. One-click launchers and tools, on PATH
# =============================================================================================
COPY run_4layer_1node.sh  /opt/miles-rocm/run_4layer_1node.sh
COPY run_full_6node.sh    /opt/miles-rocm/run_full_6node.sh
COPY verify_image.py      /opt/miles-rocm/verify_image.py
COPY audit_fp8.py         /opt/miles-rocm/audit_fp8.py
COPY bench_sparse_attn.py check_correctness.py /opt/miles-rocm/tools/
RUN chmod 0755 /opt/miles-rocm/run_4layer_1node.sh /opt/miles-rocm/run_full_6node.sh \
 && ln -sf /opt/miles-rocm/run_4layer_1node.sh /usr/local/bin/run-dsv4-4layer-1node \
 && ln -sf /opt/miles-rocm/run_full_6node.sh   /usr/local/bin/run-dsv4-full-6node \
 && bash -n /opt/miles-rocm/run_4layer_1node.sh \
 && bash -n /opt/miles-rocm/run_full_6node.sh \
 && find /opt/miles /opt/miles-rocm -name '__pycache__' -type d -prune -exec rm -rf {} +

# =============================================================================================
# 5. Gate — fail the build rather than ship a broken image
#
# The torch check is the important one: any pip step that quietly pulled a CUDA torch would make the
# whole image useless in a way that only surfaces at runtime, long after the build looked fine.
# verify_image.py additionally requires BOTH patch sentinels and both launchers.
# =============================================================================================
RUN set -e; \
    baseline="$(cat /opt/torch_baseline.txt)"; \
    actual="$(python3 -c 'import torch; print(torch.__version__)')"; \
    echo "torch baseline=${baseline} actual=${actual}"; \
    test "${baseline}" = "${actual}" || { echo "FATAL: torch changed during build"; exit 1; }; \
    python3 /opt/miles-rocm/verify_image.py --no-gpu

# =============================================================================================
# 6. Runtime metadata, declared LAST and explicitly
#
# Not redundant. `docker commit` snapshots the RUNNING CONTAINER's config, not the parent image's,
# so the commit-based path (build_with_gpu.sh, whose builder runs
# `--entrypoint bash … -lc 'sleep infinity'`) would otherwise produce Entrypoint=["bash"] /
# Cmd=["-lc","sleep infinity"] and lose Shell. That breaks the most ordinary invocation there is:
#
#     docker run <image> /bin/bash   ->   bash /bin/bash
#     /bin/bash: /bin/bash: cannot execute binary file
#
# because the user's argument replaces Cmd and lands as bash's *script* argument, so bash reads the
# ELF binary as a shell script. Declaring these here keeps the `docker build` path correct by
# construction; build_with_gpu.sh re-applies them via --change plus a Dockerfile.metadata_fix pass,
# since `docker commit --change` rejects SHELL.
#
# Values are exactly what the base declares.
# =============================================================================================
ENTRYPOINT []
CMD ["/bin/bash"]
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /opt/miles

LABEL fada.base_image="amdagi/miles-dsv4-flash:rocm7.14-gfx942-20260812-fada" \
      fada.run_id="miles_spec_20260817_1430" \
      fada.adds="apex,causal-conv1d,primus_turbo,fla-0.5.2,misc,launchers" \
      fada.framework_patches="dsv4-fwd-block_H,dsv4-ref-gather-fix" \
      fada.memory_patches="metrics,intmax,logits,temperature,rms_norm,recompute_gc,logprob_gc,expandable" \
      fada.memory_env_switches="MEGATRON_RECOMPUTE_BACKWARD_GC_GEN=2 (default on), MILES_EXPANDABLE_SEGMENTS_DURING_TRAIN=1 (default off)" \
      fada.patch_provenance="/apps/mingjiel/dsv4mem" \
      fada.gfx="gfx942" \
      fada.max_cp_1node="8" \
      fada.entrypoints="run-dsv4-4layer-1node,run-dsv4-full-6node" \
      fada.peak_metric="torch max_memory_allocated high-water, 8/8 ranks; NOT the used_GB sample the earlier 79.42 figure came from" \
      fada.verified="8x MI308X: --cp 8 -> 3/3 steps at 131072 tokens, 52.45 GiB allocated / 63.18 reserved per GPU (baseline 107.27/130.28, -51%/-52%), ~125 s/step, 0 OOM/NaN/LDS; with the expandable window reserved 60.30 and ~116 s/step"
