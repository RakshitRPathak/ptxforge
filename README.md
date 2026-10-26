# PTXForge

**An MLIR/LLVM GPU kernel compiler, a measurement-driven autotuner, and a roofline studio.**
Lowers `GEMM + bias + GELU` from `linalg` toward NVIDIA PTX, with custom C++ passes, a C++20 closed-loop tuner (`ptxforge-tune`) with an in-process CUDA Driver API runner, whose "AI advisor" is *ablated against honest baselines* instead of assumed to help.

Target audience: NVIDIA compiler / system-software internship interviews (NVVM, CUDA compiler, TensorRT-LLM, MLIR/Triton teams).

---

## What changed from the original plan (and why)

| Original plan | This kit | Why |
| --- | --- | --- |
| MLIR pipeline is the only path | **Two backends behind one interface:** `nvcc-ptx` (day 1) and `mlir` (the real thing) | You get a working loop, real GPU numbers and a studio in week 1 instead of blocking on an LLVM build. The nvcc kernel then becomes the **oracle** your MLIR output must approach. |
| Three custom passes incl. "NVVM→PTX lowering" | **Two custom passes** (`fuse-epilogue`, `pad-workgroup-memory`) built *on top of* upstream patterns + a **transform-dialect schedule** for tiling | NVVM→PTX lowering already exists upstream; re-implementing it is not a defensible contribution. Using `populateElementwiseOpsFusionPatterns` with your own control function, and driving tiling from a generated schedule, is how real MLIR projects work. |
| Recompile passes per tuning trial | Tuner **renders a transform schedule** (`scripts/schedule_template.mlir`) per config | New tile sizes need no C++ rebuild → trials are seconds. |
| "GraphRAG agent finds the optimum" | Analytic **cost model + roofline prior**, neighbor hill-climb, and advisors `none / random / rules / kg / llm` that are **compared in an ablation** | An LLM in the loop is a hypothesis, not a result. The kit makes you *test* it. The LLM can only suggest; invalid/hallucinated configs are rejected in code, and every suggestion is benchmarked. |
| CUPTI counters from the start | **Driver-API attributes first** (regs, spill bytes, occupancy, blocks/SM, correctness, GFLOP/s, cuBLAS baseline); CUPTI/ncu in milestone M4 | Works on a free Colab T4 where hardware counters are often blocked. |
| Correctness assumed | **Correctness gate before timing** (sampled CPU reference, fp64) | A fast wrong kernel is the classic autotuner bug. |
| Resume bullet: "2.4× speedup, 88% of cuBLAS on FP16" | **Removed.** Numbers must come from your own trace files (see `docs/RESUME_BULLETS.md`) | Those figures were placeholders in the original text, not measurements. Interviewers ask for the trace. Also: the shipped kernel is **FP32 SGEMM**; FP16 tensor cores are milestone M3. |
| Python tuner |  **C++20 tuner** (`core/`); Python only for glue | The project is a compiler/systems project: the search, cost model, KG and runner are C++ (header-only core + one CLI). The original Python tuner is kept in `reference/py_tuner/` purely as a parity oracle. |
| Next.js + gRPC studio | **Zero-dependency studio** (one HTML file, trace embedded) | Ship the insight first, port to Next.js/gRPC later if you want the full-stack story. |

---

## Status — be precise about this in interviews

| Component | State |
| --- | --- |
| Bank-conflict model (`compiler/include/PtxForge/BankModel.h`, `tools/bank_sim.cpp`) | ✅ **compiled and run**; known-answer tests pass |
| C++ tuner core (`core/`: space, cost model, roofline, KG, advisors, search, trace writer, JSON) | ✅ **compiled and tested** — 16 C++ unit tests, clean under ASan/UBSan, runs with the *simulated* runner |
| C++ ↔ Python reference parity | ✅ **tested**: identical search space and modeled numbers on all 3,610 configs × 4 GPUs × 3 sizes (`make parity`) |
| External/LLM advisor protocol (C++ side, validation guard) | ✅ **tested** with a fake advisor script; `scripts/llm_propose.py` (the actual API call) is ⚠️ **never run** |
| Studio (`studio/index.html`) | ✅ **rendered in headless Chromium** with a C++-produced trace, no console errors |
| `CudaRunner` (`core/src/cuda_runner.cpp`: nvcc→PTX, `cuModuleLoadData`, correctness gate, event timing, cuBLAS) | ⚠️ **syntax-checked against stub CUDA headers only**; never linked or run |
| CUDA kernel (`kernels/gemm_bias_gelu.cu`) | ⚠️ written and reviewed, **never compiled** (no `nvcc` where this was authored) — first run is M0 |
| MLIR passes + `ptxforge-opt` + CMake/TableGen | ⚠️ written for LLVM 19, **never compiled** (no LLVM available). Expect small API fixes: `docs/MLIR_API_NOTES.md` |
| FileCheck tests (`compiler/test/*.mlir`) | ⚠️ written from the pass source, **never run** — CHECK lines may need adjusting to the real output |
| `scripts/lower_to_ptx.sh`, `schedule_template.mlir` | ⚠️ **skeletons** — they are the work of milestone M1 |
| Any GFLOP/s number anywhere | ❌ **none are real yet.** `studio/sample_trace.json` is flagged `"source":"simulated"` |

Everything marked ⚠️ is exactly the work that makes the project yours. Each fix you make is interview material.

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

**Language split** (lines, this kit): C++/CUDA ≈ 2,700 (tuner core + CUDA runner + MLIR passes + tests + tools) · shipped Python ≈ 120 (studio embedding, parity script, optional LLM API bridge) · Python reference tuner ≈ 1,300 (kept only to prove the C++ port is faithful; not needed to run anything).

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

**3. The compiler (milestones M1+):** `docker build -t ptxforge -f docker/Dockerfile .`, then
`cmake -S . -B build -G Ninja -DPTXFORGE_BUILD_COMPILER=ON -DMLIR_DIR=…`, run `cmake --build build --target check-ptxforge`, and read `docs/MLIR_API_NOTES.md` when the first build complains.

## Repo map

```
core/        C++20 tuner: include/ptxforge/*.hpp, src/main.cpp (CLI), src/cuda_runner.cpp, tests/
compiler/    MLIR passes (TableGen), ptxforge-opt, samples, test/*.mlir (FileCheck)
kernels/     parametric CUDA oracle kernel               knowledge/  seed knowledge graph (symptom→remedy→side-effects)
studio/      roofline/IR/trials viewer                   scripts/    lowering skeleton, schedule template, Colab setup, LLM bridge
tools/       bank_sim                                    tests/      parity_check.py
reference/   py_tuner — the original Python tuner, parity oracle only
docs/        roadmap, interview prep, protocol
```

## What the simulated ablation already taught us

`./build/ptxforge-tune ablate --runner sim --budget 16 --seeds 10` (cold start) gave, on the simulated T4 (mean best GFLOP/s):
`none 3995 ≳ rules 3904 ≈ kg 3894 > random 3248`.
Read it carefully: the simulator is built from the same cost model the advisors reason with, so this says *only* that the harness works and that **guided search beats random**. The cost-model-ordered hill climb (`none`) matching or beating the KG/rules advisors is a warning shot: **the KG/LLM layer must beat that baseline on real hardware, with ≥10 seeds, or you should not claim it helps.** If it does not, say so — "I tested it and a cost-model hill-climb was just as good" is a stronger interview answer than an unmeasured claim.

## License / attribution

Your project; pick a license. Cite the Roofline paper (Williams, Waterman, Patterson 2009), the CUDA Best Practices Guide, and the MLIR docs where you use them.
