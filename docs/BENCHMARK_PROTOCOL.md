# Benchmark protocol

1. **Correctness before speed.** The harness gates timing on a sampled fp64 CPU reference (`--samples 512`). A config that fails the gate is recorded as failed and can never be "best".
2. **Same work, same inputs.** Identical shapes, data (fixed seed) and epilogue across variants. cuBLAS SGEMM has no GELU epilogue, so it is a *GEMM-only upper reference*; say so. (For an apples-to-apples epilogue comparison, time GEMM + a separate elementwise kernel.)
3. **Baselines that are not strawmen.** Report: (a) untuned default config, (b) best tuned, (c) cuBLAS. "×N over a naive kernel" alone is weak; "% of cuBLAS" is the number that matters.
4. **Warm-up, then median.** The harness does warm-up launches, then the median of 5 repetitions of 20 launches (events). Report the spread across ≥5 runs of the whole tune, not one lucky run.
5. **Know your clocks.** Cloud T4s throttle; datasheet peak is unreachable. The studio draws both the datasheet peak and the *assumed* sustained peak (`core/include/ptxforge/hw.hpp`, field `sustained`) — they are different lines for a reason. Record `nvidia-smi -q -d CLOCK` once per session. If you can, lock clocks on a machine you control.
6. **Sizes.** Report at least 3 problem sizes (e.g. 1024³, 2048³, 4096³) and one non-square. Kernels have no bounds checks: sizes must divide the tile (the tuner enforces it).
7. **Record the environment** in the trace/README: GPU, driver, CUDA version, nvcc flags, LLVM commit, git SHA.
8. **Roofline honesty.** Pre-M4 the x-axis is *modeled* tile traffic (no L2 reuse credit), not measured DRAM bytes. Label it that way (the studio does). Post-M4 plot both.
9. **Ablation hygiene.** Same budget, same seeds, cold start, ≥10 seeds, report mean ± std and the max. A single seed proves nothing.
10. **Simulated ≠ measured.** `"source":"simulated"` traces are for testing the tool. They must never appear in a README table or slide.
