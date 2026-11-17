# Roadmap (8 weeks, ~8–10 h/week)

Each milestone has an **exit test** — if you cannot demonstrate it, it is not done.

| M | Weeks | Goal | Exit test |
| --- | --- | --- | --- |
| **M0** ✅ | 1 | Real numbers on a real GPU | **Done** on a Tesla T4 (85.6% of cuBLAS; that trace was not committed) and on an RTX 5050 Laptop GPU (79.3%, `runs/rtx5050_2048.json`), 2026-11-17. Original exit test: `runs/t4_2048.json` with `"source":"harness"`, correctness gate green, cuBLAS baseline recorded, best config reported as **% of cuBLAS SGEMM** (whatever it is) |
| **M1** ✅ | 2–3 | Compiler builds and runs one stage | Docker image builds `ptxforge-opt`; `--ptxforge-fuse-epilogue samples/gemm_bias_gelu.mlir` merges the two generics and tags the pair; the 2 FileCheck tests in `compiler/test/` pass (`cmake --build build --target check-ptxforge`; they were written but never run when this kit was made). **Done 2026-11-02** with Ubuntu's prebuilt MLIR 19 instead of Docker; 7 FileCheck cases pass (see DEVLOG) |
| **M2** ◐ | 3–4 | MLIR → PTX that actually runs | **Partly done 2026-11-02:** `lower_to_ptx.sh` produces `06_kernel.ptx` for the fused, tiled, vectorized kernel and the lowering is checked numerically on CPU. **2026-11-17:** the PTX runs correctly on an RTX 5050 Laptop GPU (3,008 GFLOP/s, 31.6% of cuBLAS) after `ptxforge-order-kernel-args` fixed the kernel's parameter order. **Left:** stage A/B tiles in shared memory so the padding pass has work, and the items below. Original exit test: `lower_to_ptx.sh` produces `07.ptx` for a tiled matmul; `CudaRunner::run_ptx()` loads it and passes the correctness gate; an `MlirRunner` plugs into the same `Runner` interface; `--ptxforge-pad-workgroup-memory` integrated and its effect visible in `ptxas -v`/timing |
| **M3** | 5 | Tensor cores (stretch, high value) | FP16 GEMM via `gpu.subgroup_mma_*` or `nvgpu.mma.sync`, `cp.async` double-buffering; compared with `cublasGemmEx`. `cp.async` needs sm_80+, so use an A100/L4 (Colab Pro / cloud). A T4 (sm_75) has first-generation tensor cores with fewer `mma` shapes and no `cp.async` — fine for a first WMMA experiment, not for the full pipeline |
| **M4** | 6 | Real counters | CUPTI (or `ncu` CSV ingest) for bank conflicts, DRAM bytes, L2 hit; roofline uses **measured** AI; validate each counter against Nsight Compute |
| **M5** | 7 | The science | Calibrate cost model (report Spearman); ablation on **real HW**: ≥10 seeds × ≥3 problem sizes × advisors `none/random/rules/kg(/llm)`; grow the KG from your own logs; write down the result **even if the KG loses** |
| **M6** | 8 | Make it legible | Provenance in studio (MLIR `loc` → LLVM debug loc → PTX `.loc`), 2-page write-up, README GIF, headline numbers filled from traces only |

**Second architecture (added 2026-11-17):** an `rtx5050` profile (Blackwell laptop, `sm_120`) is in the tuner and parity-tested, and `scripts/rtx5050_run.sh` runs M0's exit test on it. Not yet run. It feeds M5: the cost-model rank correlation on two architectures instead of one.

## Cut lines (if time runs out)

1. Cut M3 first (tensor cores), keep FP32 — still a complete story.
2. Cut the LLM advisor; keep `kg`. Cut `kg` last, because the ablation *is* the project's thesis.
3. Never cut M0 and the correctness gate.

## Dev log

`docs/DEVLOG.md` has one entry per surprise: one entry per surprise ("expected X, saw Y, because Z"). Open questions:
- Why does `PAD=1` still leave the kernel's transposed store 4-way conflicted, but `PAD=4` makes it conflict-free? (`bank_sim`, test `bank_known_answers`)
- Why can a 16-byte vector access with 16-byte aligned rows be conflict-free at stride 36 words but 4-way at scalar width? (quarter-warp phases)
- Where does `__launch_bounds__` cap registers, and when does `TM=TN=16` start spilling?

## Ideas after M6

Epilogue variants (ReLU, SiLU, residual add) to show the fusion pass generalizes; FlashAttention-style fused softmax as a second workload; Triton-vs-yours comparison; port the studio to Next.js + gRPC streaming live trials.
