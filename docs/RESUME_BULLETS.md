# Resume bullets — fill from traces, never from hope

**Rule:** every bracketed value must come from a `trace.json` with `"source":"harness"` that you can open in an interview. If you cannot fill a bracket, delete the clause. Keep the traces in the repo (`runs/`).

## After M0 (kernel + harness + tuner on a real GPU)
- Built a C++20 autotuner (cost model, search, advisors, trace writer) with an in-process CUDA Driver-API benchmarking runner (in-process PTX load, `cuLaunchKernel`, correctness gate vs fp64 reference, cuBLAS baseline) — a closed loop over a [3,610]-config GEMM+bias+GELU space; tuned kernel reached [X]% of cuBLAS SGEMM on a [T4] ([Y]× over the untuned default).

## After M2 (MLIR path works)
- Implemented custom MLIR passes in C++20/TableGen: an epilogue-fusion pass (built on `linalg` elementwise-fusion patterns with a custom control function) and a shared-memory padding pass driven by a bank-conflict model verified against a standalone C++ simulator; lowered `linalg` → `gpu` → `nvvm` → PTX and benchmarked the generated PTX at [X]% of a hand-written CUDA oracle.

## After M5 (the science — pick the version that is TRUE)
- *If the KG helped:* Designed a knowledge-graph-guided pass/tile advisor (symptom → remedy → side-effect graph, optional LLM proposer with in-code validity guard); in a [10]-seed ablation on [T4], it reached [X] GFLOP/s vs [Y] for random search and [Z] for a cost-model hill-climb at equal trial budget.
- *If it did not:* Evaluated knowledge-graph and LLM advisors against a roofline-cost-model hill-climb baseline over [10] seeds on [GPU]; found [the baseline matched/beat them], and used the finding to [calibrate the cost model → Spearman ρ = X].

## After M6
- Built a trace-driven roofline/IR studio (static HTML) plotting each tuning iteration against measured/modeled arithmetic intensity, with PTX/LLVM/MLIR stage views.

## Things NOT to write
- Any speedup or %-of-cuBLAS you have not measured. (The original plan's "2.4×" and "88% on FP16" were illustrative placeholders. The kit's kernel is FP32; FP16 is M3.)
- "Zero-overhead" profiler, "NP-hard phase ordering solved", "GraphRAG indexing the PTX ISA" — unless you built exactly that. The honest phrases above are stronger because they are checkable.
- "CUPTI" before M4 is done.
