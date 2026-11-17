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

## 2026-11-17: second architecture measured (RTX 5050 Laptop, Blackwell)

After the two fixes above, the full run worked: 40 of 40 candidates passed the correctness gate, best 7,546.6 GFLOP/s
(1.50× the default, 79.3% of cuBLAS), and the MLIR-generated kernel passed at 3,008 GFLOP/s.

**Expected** the cost model to rank better on a second GPU, or at least differently. **Saw** ρ = −0.37, inverted again.
The trace shows where: the model gives one prediction (4,946) to eight 128×64 configs that measured 3,446–7,245. The
8×4 per-thread tile beat 4×8 at every BK (9–37%), and BK 32 was best in both shapes. The model gives all eight the
same bank-conflict degrees (A-tile store 1, A reads 1, B reads 2), the same intensity (20.9 FLOP/B) and the same
occupancy (0.5), so nothing in it can separate them. **Next:** make the bank model depend on the per-thread tile shape
and add a K-depth term; measure ρ on a random sample.

**Expected** cuBLAS near the 6.7–6.9 TFLOP/s cuda-gpu-bench measured on this laptop. **Saw** 9.5 TFLOP/s, in both the
C++ runner and the CuPy script. Unverified explanation: power mode (this run was plugged in, performance mode). The
`sustained` prior of 0.50 came from the lower figure. Left unchanged so this trace stays reproducible; worth
re-measuring cuda-gpu-bench in the same power mode before changing it.

## 2026-11-17: shared-memory staging in the MLIR schedule

The MLIR kernel ran correctly at 3,008 GFLOP/s but read A and B from global memory in every thread's own K loop. The
schedule now stages each K step's tiles in shared memory. The parts that did not work the obvious way:

- **`bufferize_to_allocation` on the A/B slices**: in MLIR 19 it only takes op handles, not values. Instead the
  schedule bufferizes first and then runs `transform.structured.promote` (memory space = workgroup) on the block-level
  matmul step, so the K loop moves from per-thread to per-block code.
- **`map_copy_to_threads` crashed** (segfault in `MapCopyToThreadsOp::applyToOne`) on memref copies; it expects tensors.
  The two copies are tiled by hand into float4 pieces over linear thread ids (`#gpu.thread<linear_dim_*>`), sized by a
  new rule (`copy_piece`, the same in the C++ tuner, the Python tuner and the CPU check) so any tile shape fits the block.
- **The accumulator was not hoisted into registers.** Three separate reasons: the barriers `map_nested_forall_to_threads`
  inserts (a `gpu.barrier` has memory effects on everything), the subview the accumulator is accessed through (MLIR 19's
  hoisting refuses view-like sources), and read/write indices that were equal but distinct SSA values. Fix: map with
  `sync_after_distribute = false`, fold the subviews into the transfer indices, CSE, then hoist. The block-sized
  accumulator buffer that bufferization created is then written and read once by the same thread, and
  `memref.erase_dead_alloc_and_stores` removes it.
- **A 64×64 vector load and store of C onto itself, per thread.** Vectorization now runs after bufferization, so the
  self-copies that `ptxforge-erase-self-copies` used to delete got vectorized first. The schedule runs that pass (after
  CSE, so the two identical subviews are one value) right after bufferization.
- **Padding by one float broke the float4 loads**: rows of 17 and 65 floats are not 16-byte aligned, so 32 vector loads
  became 64 scalar ones. With `access-bytes=16 align-bytes=16` the pass pads to 20 and 68.
- **`.shared .align 4`** on the arrays read with `ld.shared.v4`: a misaligned-address fault waiting to happen. gpu.launch
  attributions cannot carry an alignment, gpu.func ones can (`llvm.align`), so `ptxforge-workgroup-sync` runs after
  outlining and asks for 16.

Two new passes: `ptxforge-normalize-workgroup-allocs` (promote's i8 buffer + view + dealloc → one typed allocation) and
`ptxforge-workgroup-sync` (allocations in a kernel → aligned workgroup attributions; a barrier before the first shared
read and one at the end of every loop body that writes shared memory; refuses a body that reads before its last write).

**Checked:** CPU fp64 check for 64×64×16 4×4, 128×64×8 8×4 and 64×32×16 4×2 at n = 256 and n = 192 (max error ≤ 4.9e-7);
FileCheck cases for both passes, which fail without the sync pass; the PTX per K step is 2 `st.shared.v4`, `bar.sync`,
32 `ld.shared.v4`, 256 FMAs, `bar.sync`. **Not checked:** anything on a GPU. Open: the global loads in the copies are
scalar (`ld.global.f32` ×4) because nothing tells LLVM the kernel's pointers are 16-byte aligned.

## 2026-11-17: the cost model's tile-shape and K-depth blind spots

**Expected** the bank-conflict term to separate the eight 128×64 configs (BK × micro-tile shape) the RTX 5050 run
measured from 3,446 to 7,245. **Saw** one prediction for all eight. **Because** the term used the *worst* degree of
three scalar access patterns, and those were the same for every shape; it never counted how many loads each shape
issues. And nothing in the model depended on BK except shared-memory size.

What was tried, all against `runs/rtx5050_2048.json` (so in-sample), with ρ = Spearman over its 40 trials:

| Model | ρ |
|---|---:|
| as recorded | −0.37 |
| shared-load issue cycles vs FMA issue cycles, scalar loads, instead of the bank and ILP terms | +0.21 |
| old terms plus a load-latency term, L = 600 cycles | +0.29 |
| both, latency as min(1, B·C/(C+L)) / B·C/(B·C+L) / C/(C+L) | +0.54 / +0.66 / +0.63 |
| both, B·C/(B·C+L), L = 300 / 600 / 1200 | +0.58 / +0.66 / +0.67 |

L was fixed at 600 cycles from typical DRAM latency before any of this, and B·C/(B·C+L) chosen as the standard smooth
latency-hiding form (and the least sensitive to L), not as the best-scoring one. That model then scored **−0.70** on the
five-trial smoke run (`runs/rtx5050_smoke.json`), which I had not looked at. **Because** its best config is a 4×4
micro-tile 32 lanes wide, which the scalar model calls 4-way conflicted in B; but `Bs[k][tx*TN + j]` for j = 0..3 is four
aligned consecutive floats, which ptxas issues as one 128-bit load, served in four conflict-free phases. Counting each
load as the widest vector its alignment allows: **+0.75 on the 40-trial run, +0.72 on the smoke run** (the two best
smoke configs tie in the model). Both traces are now in-sample.

**Not checked:** that ptxas really emits LDS.128 for these reads (`cuobjdump -sass` on a built kernel will show it), and
how the model ranks configs it has not seen. The next tuning run records the revised model's predictions before
measuring them; that ρ is the first honest one.

## 2026-11-17: a rerun that looked like a result

A run meant to time the shared-memory kernel came back at 2,109 GFLOP/s, 32.2% of same-run cuBLAS, against 31.6% for the
first, unstaged run. It looked like "staging made no difference". **It was the old kernel**: the laptop's checkout was
still at `aab6e79` (switching branches had failed on an untracked `studio/rtx5050.html` that `rtx5050_run.sh` writes),
and that commit's PTX has no `ld.shared`. The max error matching the first run to every digit was the clue. Kept as
`runs/mlir_rtx5050_rerun.json` for what it does show: the same kernel at the same share of cuBLAS while cuBLAS itself
moved from 9,529 to 6,557 GFLOP/s with the laptop's power state.

**Rule from now on:** attribute a GPU result to a kernel only after checking the PTX that ran (`grep -c ld.shared`, or
its hash) and the commit it came from.
