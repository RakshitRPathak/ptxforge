# PTXForge

[![test](https://github.com/RakshitRPathak/ptxforge/actions/workflows/ci.yml/badge.svg)](https://github.com/RakshitRPathak/ptxforge/actions/workflows/ci.yml)

**An MLIR/LLVM GPU kernel compiler, a measurement-driven autotuner, and a roofline studio.**
Lowers `GEMM + bias + GELU` from `linalg` to NVIDIA PTX with custom C++ passes and a generated transform schedule, a C++20 closed-loop tuner (`ptxforge-tune`) with an in-process CUDA Driver API runner, whose "AI advisor" is *ablated against honest baselines* instead of assumed to help.

**Measured on two GPU architectures** (see [Results](#results)): the tuned kernel reaches **85.6% of cuBLAS on a Tesla T4** (Turing) and **79.3% on a GeForce RTX 5050 Laptop GPU** (Blackwell); the MLIR-generated kernel, with its A/B tiles staged in shared memory, reaches **68.3% of cuBLAS on the RTX 5050** (6,508 GFLOP/s, 2.2× the unstaged version) and beats cuBLAS plus a separate bias + GELU pass. On both GPUs the original cost model mis-ranked near-optimal configs (ρ = −0.67 and −0.37), which is why the tuner measures every candidate; a revised model that counts shared-memory issue cycles and exposed load latency ranks the recorded RTX 5050 run at ρ = +0.75 (in-sample, see below).

---

## Design decisions

| Alternative | Chosen design | Why |
| --- | --- | --- |
| MLIR pipeline is the only path | **Two backends behind one interface:** `nvcc-ptx` (day 1) and `mlir` (the real thing) | This gives a working loop, real GPU numbers and a studio in week 1 instead of blocking on an LLVM build. The nvcc kernel then becomes the **oracle** the MLIR output must approach. |
| Three custom passes incl. "NVVM→PTX lowering" | **Two custom optimization passes** (`fuse-epilogue`, `pad-workgroup-memory`) built *on top of* upstream patterns, three small legalization passes (`approximate-math`, `erase-self-copies`, `order-kernel-args`), and a **transform-dialect schedule** for tiling | NVVM→PTX lowering already exists upstream; re-implementing it is not a defensible contribution. Using `populateElementwiseOpsFusionPatterns` with a custom control function, and driving tiling from a generated schedule, is how real MLIR projects work. |
| Recompile passes per tuning trial | Tuner **renders a transform schedule** (`scripts/schedule_template.mlir`) per config | New tile sizes need no C++ rebuild → trials are seconds. |
| "GraphRAG agent finds the optimum" | Analytic **cost model + roofline prior**, neighbor hill-climb, and advisors `none / random / rules / kg / llm` that are **compared in an ablation** | An LLM in the loop is a hypothesis, not a result. Here it is *tested*. The LLM can only suggest; invalid/hallucinated configs are rejected in code, and every suggestion is benchmarked. |
| CUPTI counters from the start | **Driver-API attributes first** (regs, spill bytes, occupancy, blocks/SM, correctness, GFLOP/s, cuBLAS baseline); CUPTI/ncu in milestone M4 | Works on a free Colab T4 where hardware counters are often blocked. |
| Correctness assumed | **Correctness gate before timing** (sampled CPU reference, fp64) | A fast wrong kernel is the classic autotuner bug. |
| Python tuner |  **C++20 tuner** (`core/`); Python only for glue | The project is a compiler/systems project: the search, cost model, KG and runner are C++ (header-only core + one CLI). The original Python tuner is kept in `reference/py_tuner/` purely as a parity oracle. |
| Next.js + gRPC studio | **Zero-dependency studio** (one HTML file, trace embedded) | Ship the insight first, port to Next.js/gRPC later. |

---

## Status

| Component | State |
| --- | --- |
| Bank-conflict model (`compiler/include/PtxForge/BankModel.h`, `tools/bank_sim.cpp`) | ✅ **compiled and run**; known-answer tests pass |
| C++ tuner core (`core/`: space, cost model, roofline, KG, advisors, search, trace writer, JSON) | ✅ **compiled and tested** — 17 C++ unit tests, clean under ASan/UBSan, runs with the *simulated* runner |
| C++ ↔ Python reference parity | ✅ **tested**: identical search space and modeled numbers on all 3,610 configs × 5 GPUs × 3 sizes (`make parity`) |
| External/LLM advisor protocol (C++ side, validation guard) | ✅ **tested** with a fake advisor script; `scripts/llm_propose.py` (the actual API call) |
| Studio (`studio/index.html`) | ✅ **rendered in headless Chromium** with a C++-produced trace, no console errors |
| `CudaRunner` (`core/src/cuda_runner.cpp`: nvcc→PTX, `cuModuleLoadData`, correctness gate, event timing, cuBLAS) | ✅ **built and run on a Tesla T4** (Colab), including the cuBLAS baseline |
| `CudaRunner` with a driver older than the toolkit (cannot JIT newer PTX) | ✅ falls back to `nvcc -cubin` automatically; tested against a fake driver in CI (`make cuda-fallback-test`), see DEVLOG 2026-11-17 |
| CUDA kernel (`kernels/gemm_bias_gelu.cu`) | ✅ **built with nvcc and measured on a T4**: see Results |
| MLIR passes + `ptxforge-opt` + CMake/TableGen | ✅ **built against LLVM/MLIR 19** (Ubuntu's prebuilt packages); 7 passes: `fuse-epilogue`, `pad-workgroup-memory`, `approximate-math`, `erase-self-copies`, `order-kernel-args`, `normalize-workgroup-allocs`, `workgroup-sync` |
| FileCheck tests (`compiler/test/*.mlir`) | ✅ **14 cases pass** under `lit`, including negative cases; a planted bug in each pass (or, for `order-kernel-args`, running without it) makes its test fail |
| `scripts/lower_to_ptx.sh` + `schedule_template.mlir` | ✅ **run end to end**: linalg → fused, tiled, vectorized kernel → PTX for `sm_75`, for 5 tile configs, in about 0.2 s, with no CUDA toolkit or libdevice |
| Numerical check of the MLIR lowering | ✅ **on CPU**: the same schedule minus GPU mapping, compiled with LLVM, matches an fp64 NumPy reference (max error about 1e-6) at 3 sizes and 4 tile shapes (`tests/check_lowering_cpu.py`) |
| MLIR-generated PTX on a GPU | ✅ **run on an RTX 5050 Laptop GPU**: correct at 3,008 GFLOP/s (31.6% of cuBLAS), see Results. The first attempt produced NaN because outlining had reordered the kernel's parameters; fixed by `ptxforge-order-kernel-args` (DEVLOG). Not yet run on a T4 |
| Shared-memory tiles in the MLIR path | ✅ **measured on the RTX 5050**: 1.3–1.9× faster than the unstaged schedule at all five tile sizes swept, best 68.3% of cuBLAS (Results); each K step copies A and B tiles into padded shared memory with float4 stores, two barriers, accumulator in registers (see [The MLIR compiler path](#the-mlir-compiler-path)) |
| Measured GPU results | ✅ 40-trial tuning runs on a T4 and an RTX 5050 Laptop GPU for the `nvcc` backend (see Results); advisor ablation on real hardware not yet run |
| Second GPU architecture: RTX 5050 Laptop (Blackwell, `sm_120`) | ✅ **40-trial tune measured**: best 7,547 GFLOP/s, 79.3% of cuBLAS, ρ = −0.37 (see Results); `--gpu rtx5050`, `scripts/rtx5050_run.sh` |

⚠️ = not done yet, as described. The shipped kernel is FP32 SGEMM; FP16 tensor cores are milestone M3. Every number in Results comes from a measured trace (`source: harness`), never from the simulator.

---

## Results

**Tesla T4 (Google Colab), `nvcc` backend, GEMM + bias + GELU at M = N = K = 2048, FP32, 40 measured trials, `kg` advisor.** The correctness gate checks each candidate against an fp64 reference before it is timed.

| | GFLOP/s | |
|---|---:|---|
| Untuned default config | 2,074.0 | the baseline every speedup is measured against |
| Best config found | **3,688.5** | **1.78×** the default |
| cuBLAS SGEMM, same shapes, same run | 4,307.5 | best is **85.6%** of cuBLAS |

GFLOP/s counts only the 2·M·N·K GEMM FLOPs. cuBLAS runs a plain SGEMM while PTXForge's kernel also applies the bias + GELU epilogue, so 85.6% slightly understates the fused kernel.

**The cost model mis-ranked the configs it was steering towards: Spearman ρ = −0.67** between predicted and measured GFLOP/s over the 40 trials. Two notes on reading it:

- The 40 trials are not a random sample. The search concentrates on configs the prior already rates highly, so this shows the model cannot order *near-optimal* configs on a T4, not that it is wrong across the whole space.
- It is still the key result. The tuner reached 85.6% of cuBLAS because it times every candidate and keeps the measured best, not because the model's ranking was right. That is the case for closed-loop measurement over a static cost model.

### Second architecture: GeForce RTX 5050 Laptop GPU (Blackwell, `sm_120`)

**Same problem, same 40-trial `kg` search, run on 17 November 2026** under WSL2 with the laptop plugged in and in its
performance power mode (`scripts/rtx5050_run.sh`). All 40 candidates passed the correctness gate (max error 4.2e-6).
Traces: [`runs/rtx5050_2048.json`](runs/rtx5050_2048.json), studio page [`studio/rtx5050.html`](studio/rtx5050.html).

| | GFLOP/s | |
|---|---:|---|
| Untuned default config (64×64×8, 4×4 per thread) | 5,026.3 | baseline |
| Best config found (128×64×32, 8×4 per thread, pad 2) | **7,546.6** | **1.50×** the default |
| cuBLAS SGEMM, same shapes, same run | 9,512.1 | best is **79.3%** of cuBLAS |

**The cost model mis-ranked configs again: Spearman ρ = −0.37** (T4: −0.67), so the inversion is not a T4 quirk. This
run shows two places where it goes wrong. The model gives the same prediction (4,946 GFLOP/s) to all eight 128×64
configs that differ only in BK and per-thread tile shape, while they measured anywhere from 3,446 to 7,245:

| 128×64 tile | BK 4 | BK 8 | BK 16 | BK 32 |
|---|---:|---:|---:|---:|
| 4×8 per thread | 4,149 | 3,446 | 5,089 | 5,305 |
| 8×4 per thread | 4,701 | 3,758 | 6,615 | 7,245 |

- **Per-thread tile orientation matters and the model can't see it:** 8×4 beat 4×8 at every BK, by 9–37%. Both do
  32 FMAs per thread per K step and differ in how each thread reads its A and B fragments from shared memory. The
  model's bank-conflict term scores the two shapes the same (A reads 1-way, B reads 2-way), and its intensity and
  occupancy terms are identical too, so it predicts the same number for both.
- **Deeper K tiles helped on Blackwell,** BK 32 being best in both rows, which the model's arithmetic-intensity term
  doesn't separate either.

**The fix (same day).** The model's bank term took the *worst* conflict degree of three access patterns. The revised
model (`core/include/ptxforge/cost_model.hpp`, mirrored in Python and parity-checked on every config) instead:

- **counts shared-memory issue cycles against FMA issue cycles** per K step and warp: every A and B load instruction
  costs its phases times its conflict degree, with loads issued as the widest vector their alignment allows (a float4
  warp load is 4 conflict-free phases, not 4 scalar 4-way-conflicted loads); 128 FP32 lanes per SM on Blackwell, 64 on
  Turing. An 8×4 micro-tile issues 12 cycles of loads per 32 FMAs where 4×8 issues 20, which is the difference measured above.
- **adds exposed load latency:** with no double buffering a block waits about one DRAM round trip (600 cycles, set
  a priori) per K step, and its co-resident blocks hide it with B·C/(B·C + L) where C is a block's FMA work per step.
  Deeper K steps mean more work per wait.

`python3 scripts/model_rank.py runs/rtx5050_2048.json` re-scores the measured trials with the current model:

| Trace | Trials | ρ, model at the time | ρ, revised model | Revised model's top pick |
|---|---:|---:|---:|---|
| RTX 5050, 2048³ tune | 40 | −0.37 | **+0.75** | measured 7,245 (best 7,547) |
| RTX 5050, 1024³ smoke test | 5 | −0.40 | +0.72 | measured 4,914 (best 5,299; tied with it in the model) |

Both traces are **in-sample**: the model was revised while looking at them (DEVLOG lists what was tried). They show
the model can describe these runs, not that it predicts new ones. The test that counts is the next tuning run, which
records the revised model's predictions before measuring.

Two caveats on these numbers:

- **cuBLAS measured 9.5 TFLOP/s here**, in both the tuner (cuBLAS C API) and the CuPy script, against 6.7–6.9 in
  [cuda-gpu-bench](https://github.com/RakshitRPathak/cuda-gpu-bench)'s earlier runs on the same laptop. The likeliest
  difference is the power mode (laptop GPUs follow the power limit), but that is not verified. Only same-run ratios
  (79.3%, 1.50×) are comparable across sessions. The `rtx5050` profile's `sustained` prior (0.50) was set from the
  lower figure; 9.5 / 13.6 suggests about 0.70.
- **This almost certainly ran on cubins, not the driver's PTX JIT.** The laptop's driver refused the toolkit's PTX in
  the two earlier attempts that day, so the runner would have switched to `nvcc -cubin` (DEVLOG 2026-11-17); the
  trace doesn't record which path was taken. Same kernel source and flags; only the step that turns
  PTX into machine code moved from the driver to `ptxas`.

**The MLIR-generated kernel on the same GPU** ([`runs/mlir_rtx5050.json`](runs/mlir_rtx5050.json)): correct (max abs
error 4.9e-6, every output written) at **3,008 GFLOP/s**, 31.6% of cuBLAS SGEMM and 40% of the tuned `nvcc` kernel. It
is also slower than cuBLAS followed by a separate bias + GELU pass (5.71 ms vs 2.75 ms), so fusion doesn't pay off
yet. That is expected for this schedule: it doesn't stage A and B tiles in shared memory, so every thread re-reads
global memory. Its first run on this GPU produced NaN; the cause (kernel parameters in the wrong order) is fixed and
described in DEVLOG.

**With A/B tiles staged in shared memory** (the schedule in [The MLIR compiler path](#the-mlir-compiler-path)), timed
the same evening in the same power mode by `scripts/mlir_sweep.sh`, which first checks that every kernel is the one its
name says ([`runs/mlir_sweep/`](runs/mlir_sweep/)). Both schedules compiled by the same pipeline at each tile size; all
ten correct (max abs error 4.9e-6, every output written); % of cuBLAS SGEMM timed in the same process:

| Block tile, BK, per-thread tile | Unstaged | Staged in shared memory | Speedup |
|---|---:|---:|---:|
| 64×64, 16, 4×4 (the default) | 31.5% | 55.9% | 1.78× |
| 64×64, 32, 4×4 | 32.7% | 62.7% | 1.93× |
| 128×64, 16, 8×4 | 43.0% | 55.8% | 1.30× |
| **128×64, 32, 8×4** | 39.8% | **68.3%** (6,508 GFLOP/s) | 1.71× |
| 128×128, 16, 8×8 | 46.0% | 65.7% | 1.43× |

- **Staging paid off at every tile size**, by 1.3–1.9×. The unstaged default reproduced the first run (2,999 vs 3,008
  GFLOP/s), so the two sessions agree.
- **The best MLIR config is the tuned `nvcc` kernel's best config** (128×64×32, 8×4): 68.3% of cuBLAS against the `nvcc`
  kernel's 79.3% from the tuning run, i.e. the compiler-generated kernel is at about 86% of the hand-written one.
- **Fusion now pays:** 2.64 ms for GEMM + bias + GELU in one kernel against 2.76 ms for cuBLAS SGEMM followed by a
  separate bias + GELU pass. Unstaged, the fused kernel was twice as slow as that.
- Not separated yet: how much of the gain is the staging and how much the padding pass (64×20 / 16×68 instead of
  64×16 / 16×64); and the global loads in the copies are still scalar. Both are next.

A rerun of the *unstaged* kernel on a
later evening ([`runs/mlir_rtx5050_rerun.json`](runs/mlir_rtx5050_rerun.json)) gave the same share of cuBLAS (32.2% vs
31.6%) while cuBLAS itself measured 6,557 GFLOP/s instead of 9,529: the same laptop in a lower power state, close to
cuda-gpu-bench's earlier 6.7–6.9, which supports the power-mode explanation above.

Next experiments: the MLIR schedule's tile config chosen by the tuner instead of a fixed sweep (an `MlirRunner`); padding on/off and
vectorized global loads in the copies; double buffering; a new tuning run with the revised model (its
first out-of-sample ρ) plus ρ on a random sample of configs, on both GPUs; the `none / random / rules / kg` advisor
ablation with `--runner cuda`; double buffering in the MLIR schedule.

---

## The MLIR compiler path

`scripts/lower_to_ptx.sh compiler/samples/gemm_bias_gelu.mlir build/ir sm_75` writes every stage to `build/ir/`:

| Stage | What happens | Pass |
| --- | --- | --- |
| 1 | Bias-add and GELU merge into one `linalg.generic`; it and the matmul are tagged as a pair | `ptxforge-fuse-epilogue` (custom) |
| 2 | Tile to 64×64 per thread block; the block steps K by 16, and `promote` stages each step's 64×16 slice of A and 16×64 slice of B in shared memory; every thread computes a 4×4 micro-tile from them while the two copies are spread over all 256 threads (one float4 each); vectorize; map to `gpu.launch` without barriers, so the accumulator can be hoisted into registers across K and the block-sized accumulator buffer disappears (store-to-load forwarding) | transform schedule rendered by `ptxforge-tune schedule` |
| 3 | Fold `promote`'s byte buffers into typed shared allocations; drop the no-op tile copies bufferization leaves behind; replace `tanh` and `fabs` with arith ops so no libdevice is needed; unroll vectors | `ptxforge-normalize-workgroup-allocs`, `ptxforge-erase-self-copies`, `ptxforge-approximate-math` (custom) |
| 4 | Pad the shared tiles to avoid bank conflicts, with a 16-byte access model so rows stay float4-aligned (64×16 → 64×20, 16×64 → 16×68) | `ptxforge-pad-workgroup-memory` (custom) |
| 5–6 | Outline the kernel and put its parameters back in the source order (outlining orders them by first use); turn the shared tiles into 16-byte-aligned kernel attributions and add the K loop's two barriers (after the copies, at the end of the step); lower to NVVM, emit PTX with LLVM's NVPTX backend, extract it. The script fails if the final launch is out of order | upstream `gpu-kernel-outlining`, `ptxforge-order-kernel-args`, `ptxforge-workgroup-sync` (custom), upstream `gpu-lower-to-nvvm-pipeline` |

The result (`kernels/generated/gemm_bias_gelu_2048_sm75.ptx`) is one kernel taking four pointers in the source order (A, B, bias, C), 16×16 threads per block,
two `.shared .align 16` tiles (5,120 and 4,352 bytes), and per K step: two `st.shared.v4` (each thread's share of the tile copies), `bar.sync`,
32 `ld.shared.v4` and 256 FMAs with the 4×4 accumulator in registers (no local memory), `bar.sync`. CI regenerates it and fails if the committed copy
is stale; `tests/check_lowering_cpu.py` runs the same schedule on the CPU against an fp64 reference for three tile configs.

What was fixed to get here is in `docs/DEVLOG.md`. In short: one TableGen declaration, a lit config for current `lit`, constants turning into kernel
parameters, `tanh`/`fabs`/n-D `fma` falling back to libdevice, and a result buffer being copied instead of written in place.

**Run it on a GPU** (Colab or Kaggle T4; CuPy is preinstalled, no CUDA toolkit needed):
```bash
git clone https://github.com/RakshitRPathak/ptxforge && cd ptxforge
python3 scripts/run_ptx_on_gpu.py --json runs/mlir_t4.json
```
It checks the output against an fp64 reference, then times the kernel against cuBLAS SGEMM and against cuBLAS + a separate bias/GELU pass.
The committed kernel uses the default 64×64×16 tile (55.9% of cuBLAS on the RTX 5050). `bash scripts/mlir_sweep.sh` times both
schedules at five tile sizes (`kernels/generated/sweep/`); the best, 128×64×32 with 8×4 per thread, reached 68.3%.

---

## Architecture

```
            ┌──────────────── ptxforge-tune (C++20, core/) ────────────────┐
 problem ──►│ space → cost model/roofline prior → search loop ◄── advisors │
            │   (3.6k configs)   (rank prior)        │   none|random|rules|kg|llm
            └──────────────┬─────────────────────────┴───────────────────┬─┘
                config     │                                  trace.json │
        ┌──────────────────▼───────────────────┐                         ▼
        │ backend A: nvcc -ptx (day 1, oracle) │                ┌─────────────────┐
        │ backend B: MLIR (the real project):  │                │ studio (HTML)   │
        │  linalg ─ptxforge-fuse-epilogue─►    │                │ roofline · IR   │
        │  transform schedule (tile+fuse) ─►   │                │ stages · trials │
        │  bufferize ─promote to shared─►      │
        │  pad-workgroup-memory ─workgroup-sync│                └─────────────────┘
        │  gpu→nvvm→llvm→PTX                   │
        └──────────────────┬───────────────────┘
                           │ PTX
              ┌────────────▼─────────────────┐
              │ CudaRunner (same C++ binary) │  cuModuleLoadData · cuLaunchKernel
              │ correctness gate → timing    │  regs/spills/occupancy · cuBLAS baseline
              └──────────────────────────────┘
```

**Language split** (lines): C++/CUDA ≈ 2,800 (tuner core + CUDA runner + MLIR passes + tests + tools) · shipped Python ≈ 120 (studio embedding, parity script, optional LLM API bridge) · Python reference tuner ≈ 1,300 (kept only to prove the C++ port is faithful; not needed to run anything).

## Quick start

**1. No GPU, no LLVM (2 minutes) — see the loop and the studio:**
```bash
make test     # C++ unit tests, bank model, C++<->Python parity, Python reference tests
make demo     # simulated tune with ptxforge-tune -> studio/demo_simulated.html (open in a browser)
./build/ptxforge-tune ablate --runner sim --budget 16 --seeds 10
```
Needs only g++ ≥ 12 (C++20) and python3. CMake works too: `cmake -S . -B build && cmake --build build && ctest --test-dir build`.

**2. Real numbers on a free Colab/Kaggle T4 (milestone M0):**
```bash
bash scripts/colab_setup.sh        # builds ptxforge-tune-cuda (nvcc→PTX→driver JIT in-process)
./build/ptxforge-tune-cuda tune --runner cuda --gpu t4 --budget 40 --out runs/t4_2048.json
python3 scripts/make_studio.py runs/t4_2048.json studio/t4.html
```
**3. A second architecture: RTX 5050 Laptop GPU (Blackwell, `sm_120`), under WSL2 or Linux:**
```bash
bash scripts/rtx5050_run.sh        # smoke test, 40-trial tune vs cuBLAS, then the MLIR-generated PTX
```
The profile uses compute capability 12.0 limits (48 warps and 32 blocks per SM, 128 KB shared memory per SM), 384 GB/s
of DRAM bandwidth and a `sustained` prior of 0.50, taken from cuBLAS SGEMM measuring 6.7–6.9 TFLOP/s on this laptop in
[cuda-gpu-bench](https://github.com/RakshitRPathak/cuda-gpu-bench). The question this run answers: the cost model ranked
near-optimal configs backwards on the T4 (ρ = −0.67); does it also do so on a GPU with far more cache and half the SMs?
This has not been run yet, so there are no RTX 5050 numbers in Results.

Optional LLM advisor: `export ANTHROPIC_API_KEY=…; ./build/ptxforge-tune-cuda tune --runner cuda --advisor llm --llm-cmd "python3 scripts/llm_propose.py"` — suggestions are validated in C++ before anything runs.

**3. The compiler.** On Ubuntu 24.04 the prebuilt LLVM/MLIR 19 packages are enough (no source build, no CUDA toolkit):
```bash
sudo apt-get install llvm-19-dev llvm-19-tools libmlir-19-dev mlir-19-tools clang-19 ninja-build
pip install lit numpy
make compiler-test    # builds ptxforge-opt, runs the FileCheck tests, lowers to PTX, checks the numbers on CPU
```
`docker/Dockerfile` has the same toolchain plus CUDA, for running everything on one GPU machine.

## Repo map

```
core/        C++20 tuner: include/ptxforge/*.hpp, src/main.cpp (CLI), src/cuda_runner.cpp, tests/
compiler/    MLIR passes (TableGen), ptxforge-opt, samples, test/*.mlir (FileCheck)
kernels/     parametric CUDA oracle kernel; generated/ holds the MLIR-produced PTX
knowledge/   seed knowledge graph (symptom→remedy→side-effects)
studio/      roofline/IR/trials viewer
scripts/     lower_to_ptx.sh, schedule template, extract_ptx.py, run_ptx_on_gpu.py, Colab setup, LLM bridge
tools/       bank_sim                                    tests/      parity_check.py, check_lowering_cpu.py
reference/   py_tuner — the original Python tuner, parity oracle only
docs/        roadmap, dev log, benchmark protocol, MLIR API notes
```
