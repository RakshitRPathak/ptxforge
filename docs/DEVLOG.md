# Dev log

One entry per surprise: expected X, saw Y, because Z.

## 2026-11-02: first build of the MLIR passes, and linalg → PTX end to end (LLVM/MLIR 19.1.1)

Toolchain: Ubuntu 24.04's `llvm-19-dev`, `libmlir-19-dev`, `mlir-19-tools`, `clang-19`. No source build and no CUDA toolkit.

1. **`PadWorkgroupMemory.cpp` didn't compile; `FuseEpilogue.cpp` did, unchanged.** Expected the API drift in
   `MLIR_API_NOTES.md`; saw `unknown type name 'PadWorkgroupMemoryOptions'`. A pass with `options` needs
   `GEN_PASS_DECL_<PASS>` defined before `GEN_PASS_DEF_<PASS>` so the generated `Options` struct exists. One line.

2. **Both FileCheck tests passed on the first run, but one checked almost nothing.** `pad-workgroup.mlir` accepted
   any padded width (`8x{{[0-9]+}}`). The pass picks pad 1 (stride 65 floats) for scalar fp32 and pad 4 (stride 68)
   for 16-byte loads with 16-byte rows, matching `BankModel.h`. Tests now pin both, plus negative cases (already
   conflict-free, dealloc users, rank 3, multi-use matmul, multi-use intermediate). Removing the multi-use check
   from `FuseEpilogue` and adding one element of padding each make their test fail.

3. **New tests failed for a reason unrelated to the pass.** The greedy rewriter erases allocations whose loads go
   unused, so test buffers have to flow into a store.

4. **`lit` refused the config.** `ShTest(execute_external=True)` is deprecated in current `lit`; the internal shell works.

5. **Bufferization allocated a new 2048×2048 output instead of writing C.** Expected the fused epilogue to write into
   the `C` argument; saw `memref.alloc` plus a copy. The elementwise-fusion patterns replace an `outs` operand the
   payload never reads with `tensor.empty`, so the destination is gone, and empty-tensor elimination treats function
   arguments as read-only. Fix: the kernel takes memrefs, inputs come in through `bufferization.to_tensor ... restrict`,
   and the result goes out through `bufferization.materialize_in_destination ... in restrict writable %C`.

6. **Every thread copied its whole block tile onto itself.** Bufferizing `tensor.parallel_insert_slice` writes the
   tile in place and then copies the subview onto the identical subview. MLIR 19 doesn't fold that; inside a kernel
   it's a 64×64 loop of loads and stores per thread. Added `ptxforge-erase-self-copies`.

7. **`tanh` (and then `fabs`) became libdevice calls.** `convert-gpu-to-nvvm` maps them to `__nv_tanhf` / `__nv_fabsf`,
   which need libdevice from a CUDA toolkit; it even marks `llvm.intr.fabs` illegal. Added
   `ptxforge-approximate-math`: upstream's polynomial `tanh`/`erf` (pure arith) and an exact `fabs` (clear the sign
   bit, scalar or vector). The PTX is self-contained; `lower_to_ptx.sh` fails if any `__nv_` call remains.

8. **The kernel had 29 parameters instead of 4.** Constants hoisted outside `gpu.launch` became kernel arguments when
   outlined. `gpu-launch-sink-index-computations` fixes it, but any greedy rewrite after it (canonicalize, or the
   padding pass) hoists them out again, so it has to run immediately before outlining.

9. **The accumulator lived in local memory.** The 4×4 per-thread accumulator was a 64-byte stack buffer, read and
   written every K step. Vectorizing each thread's work (`vectorize_children_and_apply_patterns`), lowering contractions
   to outer products, and unrolling put it in registers: the PTX has no `.local` at all and 256 FMAs per K step.

10. **Vectorizing broke the NVVM lowering.** The GELU epilogue became `vector<4x4xf32>` math, and LLVM 19's
    `convert-gpu-to-nvvm` only lowers scalar and 1-D math ops, leaving `unrealized_conversion_cast`s. `absf` is now
    expanded for vectors too, and `convert-math-to-llvm` (which handles n-D vectors) runs before the NVVM pipeline.

Checked: `tests/check_lowering_cpu.py` runs the same schedule minus GPU mapping on the CPU and matches fp64 NumPy to
about 1e-6 for 3 sizes and 4 tile shapes; a 0.6% change to one GELU constant makes it fail. PTX is generated for
5 tile configs. Not checked here: running the PTX on a GPU (`scripts/run_ptx_on_gpu.py` on a Colab T4).

## 2026-11-17: first run on an RTX 5050 Laptop GPU (Blackwell, sm_120): zero trials ran

**Expected** the 40-trial tune to produce numbers. **Saw** all 45 trials (smoke test and tune) fail at
`cuModuleLoadData` with "the provided PTX was compiled with an unsupported toolchain"
(`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), before the correctness gate and before any timing. **Because** the laptop's
NVIDIA driver supports CUDA 13.1 while `nvcc` in WSL emits PTX for a newer ISA version. A driver can only JIT PTX
whose version it knows. SASS is different: CUDA's minor version compatibility lets a 13.1 driver run a cubin built
by any 13.x toolkit. The T4 never hit this because Colab's driver was newer than its toolkit.

**Fix:** `CudaRunner` still generates and first tries the PTX (the studio shows it, and JIT is the normal path). On
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` it switches, once, to `nvcc -cubin` for the same config and says so on stderr.
Kernel source, flags and timing method are unchanged. A toolkit with a newer *major* version than the driver still
fails; the error then names the driver version.

**Checked without a GPU:** `tests/fake_cuda/` compiles the real runner against NVIDIA's headers and a fake driver
that rejects PTX the same way and computes the kernel on the CPU. With the fix, all trials pass the correctness gate
through cubins and the notice prints once; with the old code it reproduces the laptop failure exactly (0 of 6
trials). CI runs it (`make cuda-fallback-test`). **Not checked:** a real run on the laptop.

## 2026-11-17: the MLIR-generated kernel's first GPU run produced NaN

**Expected** `scripts/run_ptx_on_gpu.py` to pass its fp64 check on the RTX 5050. **Saw** `correct: false`,
`max_abs_err: NaN`. **Because** the kernel's parameters were not in the order everyone assumed. The source function is
`@gemm_bias_gelu(A, B, bias, C)` and the README and the runner said the PTX takes (A, B, bias, C). Decoding the PTX
showed parameter 0 is the one stored to, parameter 3 is read once per column, and parameters 1 and 2 feed the K loop:
the real order was **(C, A, B, bias)**. `gpu-kernel-outlining` orders captured values by first use in the kernel body,
and the body addresses its output tile first. The runner's call therefore read the bias from C, which it pre-fills
with NaN, and wrote the result over A, leaving C untouched.

Why nothing caught it: `tests/check_lowering_cpu.py` runs the schedule without GPU mapping, so it calls the function
before outlining, where the order is still the source order. It validates the arithmetic, not the kernel ABI.

**Fix:** a new pass, `ptxforge-order-kernel-args`, run right after an explicit `--gpu-kernel-outlining`, permutes each
kernel's parameters and its `gpu.launch_func` operands into the host function's argument order (anything that is not
a host argument goes last). The regenerated PTX differs only in which parameter register feeds which base address.
`lower_to_ptx.sh` now fails if the final launch passes descriptor pointers out of order, and the runner records how
many output elements were left unwritten.

**Checked:** FileCheck test `order-kernel-args.mlir` (fails on unpassed IR); decoding the new PTX shows parameter 3 is
stored to, parameter 2 is read 4 times per thread (bias) and parameters 0 and 1 carry the 64 GEMM loads each. CPU
numerical checks unchanged. **Not checked:** the rerun on the GPU.
