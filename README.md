# PTXForge

[![test](https://github.com/RakshitRPathak/ptxforge/actions/workflows/ci.yml/badge.svg)](https://github.com/RakshitRPathak/ptxforge/actions/workflows/ci.yml)

**An MLIR/LLVM GPU kernel compiler, a measurement-driven autotuner, and a roofline studio.**
Lowers `GEMM + bias + GELU` from `linalg` to NVIDIA PTX with custom C++ passes and a generated transform schedule, a C++20 closed-loop tuner (`ptxforge-tune`) with an in-process CUDA Driver API runner, whose "AI advisor" is *ablated against honest baselines* instead of assumed to help.

---

## Design decisions

| Alternative | Chosen design | Why |
| --- | --- | --- |
| MLIR pipeline is the only path | **Two backends behind one interface:** `nvcc-ptx` (day 1) and `mlir` (the real thing) | This gives a working loop, real GPU numbers and a studio in week 1 instead of blocking on an LLVM build. The nvcc kernel then becomes the **oracle** the MLIR output must approach. |
| Three custom passes incl. "NVVM→PTX lowering" | **Two custom optimization passes** (`fuse-epilogue`, `pad-workgroup-memory`) built *on top of* upstream patterns, two small legalization passes (`approximate-math`, `erase-self-copies`), and a **transform-dialect schedule** for tiling | NVVM→PTX lowering already exists upstream; re-implementing it is not a defensible contribution. Using `populateElementwiseOpsFusionPatterns` with a custom control function, and driving tiling from a generated schedule, is how real MLIR projects work. |
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
| C++ tuner core (`core/`: space, cost model, roofline, KG, advisors, search, trace writer, JSON) | ✅ **compiled and tested** — 16 C++ unit tests, clean under ASan/UBSan, runs with the *simulated* runner |
| C++ ↔ Python reference parity | ✅ **tested**: identical search space and modeled numbers on all 3,610 configs × 4 GPUs × 3 sizes (`make parity`) |
| External/LLM advisor protocol (C++ side, validation guard) | ✅ **tested** with a fake advisor script; `scripts/llm_propose.py` (the actual API call) |
| Studio (`studio/index.html`) | ✅ **rendered in headless Chromium** with a C++-produced trace, no console errors |
| `CudaRunner` (`core/src/cuda_runner.cpp`: nvcc→PTX, `cuModuleLoadData`, correctness gate, event timing, cuBLAS) | ✅ **built and run on a Tesla T4** (Colab), including the cuBLAS baseline |
| CUDA kernel (`kernels/gemm_bias_gelu.cu`) | ✅ **built with nvcc and measured on a T4**: see Results |
| MLIR passes + `ptxforge-opt` + CMake/TableGen | ✅ **built against LLVM/MLIR 19** (Ubuntu's prebuilt packages); 4 passes: `fuse-epilogue`, `pad-workgroup-memory`, `approximate-math`, `erase-self-copies` |
| FileCheck tests (`compiler/test/*.mlir`) | ✅ **7 cases pass** under `lit`, including negative cases; a planted bug in each pass makes its test fail |
| `scripts/lower_to_ptx.sh` + `schedule_template.mlir` | ✅ **run end to end**: linalg → fused, tiled, vectorized kernel → PTX for `sm_75`, for 5 tile configs, in about 0.2 s, with no CUDA toolkit or libdevice |
| Numerical check of the MLIR lowering | ✅ **on CPU**: the same schedule minus GPU mapping, compiled with LLVM, matches an fp64 NumPy reference (max error about 1e-6) at 3 sizes and 4 tile shapes (`tests/check_lowering_cpu.py`) |
| MLIR-generated PTX on a GPU | ⚠️ **not yet run**: `scripts/run_ptx_on_gpu.py` checks and times `kernels/generated/gemm_bias_gelu_2048_sm75.ptx` on a Colab T4 |
| Shared-memory tiles in the MLIR path | ⚠️ not yet: A/B are read from global memory (through L1/L2), so `pad-workgroup-memory` has nothing to pad in this pipeline yet (milestone M2) |
| Measured GPU results | ✅ one 40-trial tuning run on a T4 for the `nvcc` backend (see Results); advisor ablation on real hardware not yet run |

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

Next experiments: rank correlation on a random sample of configs (to measure the model globally), the `none / random / rules / kg` advisor ablation with `--runner cuda`, and finding which roofline term (occupancy, register cap, wave quantization) causes the inversion.

---

## The MLIR compiler path

`scripts/lower_to_ptx.sh compiler/samples/gemm_bias_gelu.mlir build/ir sm_75` writes every stage to `build/ir/`:

| Stage | What happens | Pass |
| --- | --- | --- |
| 1 | Bias-add and GELU merge into one `linalg.generic`; it and the matmul are tagged as a pair | `ptxforge-fuse-epilogue` (custom) |
| 2 | Tile to 64×64 per thread block and 4×4 per thread, pull the matmul and its zero-fill into both loops, step K by 16, vectorize each thread's work, bufferize so the result is written straight into `C`, map the loops to `gpu.launch` | transform schedule rendered by `ptxforge-tune schedule` |
| 3 | Drop the no-op tile copies bufferization leaves behind; replace `tanh` and `fabs` with arith ops so no libdevice is needed; unroll vectors | `ptxforge-erase-self-copies`, `ptxforge-approximate-math` (custom) |
| 4 | Pad 2-D shared-memory buffers to avoid bank conflicts (no-op until tiles are staged in shared memory) | `ptxforge-pad-workgroup-memory` (custom) |
| 5–6 | Outline the kernel, lower to NVVM, emit PTX with LLVM's NVPTX backend, extract it | upstream `gpu-lower-to-nvvm-pipeline` |

The result (`kernels/generated/gemm_bias_gelu_2048_sm75.ptx`) is one kernel taking four pointers (A, B, bias, C), 16×16 threads per block,
with each thread's 4×4 accumulator in registers (no local memory) and each K step fully unrolled into 256 FMAs. CI regenerates it and fails if the
committed copy is stale.

What was fixed to get here is in `docs/DEVLOG.md`. In short: one TableGen declaration, a lit config for current `lit`, constants turning into kernel
parameters, `tanh`/`fabs`/n-D `fma` falling back to libdevice, and a result buffer being copied instead of written in place.

**Run it on a GPU** (Colab or Kaggle T4; CuPy is preinstalled, no CUDA toolkit needed):
```bash
git clone https://github.com/RakshitRPathak/ptxforge && cd ptxforge
python3 scripts/run_ptx_on_gpu.py --json runs/mlir_t4.json
```
It checks the output against an fp64 reference, then times the kernel against cuBLAS SGEMM and against cuBLAS + a separate bias/GELU pass.
Expect it to be well below the tuned `nvcc` kernel: this schedule doesn't stage tiles in shared memory yet.

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
        │  bufferize ─pad-workgroup-memory─►   │                └─────────────────┘
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
