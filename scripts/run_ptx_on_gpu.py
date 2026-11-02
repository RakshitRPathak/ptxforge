#!/usr/bin/env python3
"""Run the MLIR-generated GEMM + bias + GELU PTX on a real GPU, check it, and time it against cuBLAS.

    python3 scripts/run_ptx_on_gpu.py                                   # the committed T4 kernel
    python3 scripts/run_ptx_on_gpu.py --ptx build/ir/06_kernel.ptx --bm 64 --bn 64

Needs CuPy (preinstalled on Colab and Kaggle GPU runtimes). The CUDA driver JIT-compiles the PTX,
so no CUDA toolkit or ptxas is needed. The kernel takes four pointers (A, B, bias, C) and computes
C = gelu_tanh(A @ B + bias) for N x N fp32 matrices; one thread block computes a BM x BN tile.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy is required (it is preinstalled on Colab GPU runtimes): pip install cupy-cuda12x")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PTX = os.path.join(ROOT, "kernels", "generated", "gemm_bias_gelu_2048_sm75.ptx")


def gelu_tanh(x):
    return 0.5 * x * (1.0 + cp.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def time_ms(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx", default=DEFAULT_PTX)
    ap.add_argument("--n", type=int, default=2048, help="must match the size the PTX was compiled for")
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--json", help="also write the results to this file")
    a = ap.parse_args()

    ptx = open(a.ptx).read()
    kernel_name = re.search(r"\.entry\s+(\w+)", ptx).group(1)
    tx, ty, tz = (int(v) for v in re.search(r"\.maxntid\s+(\d+),\s*(\d+),\s*(\d+)", ptx).groups())
    n = a.n
    grid = (n // a.bn, n // a.bm, 1)
    block = (tx, ty, tz)

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    gpu = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(f"GPU: {gpu} (sm_{props['major']}{props['minor']})  kernel: {kernel_name}  grid {grid}  block {block}")

    kernel = cp.RawModule(code=ptx).get_function(kernel_name)

    rng = np.random.default_rng(0)
    A = cp.asarray((rng.standard_normal((n, n)) * 0.1).astype(np.float32))
    B = cp.asarray((rng.standard_normal((n, n)) * 0.1).astype(np.float32))
    bias = cp.asarray(rng.standard_normal(n).astype(np.float32))
    C = cp.full((n, n), cp.nan, dtype=cp.float32)

    run = lambda: kernel(grid, block, (A, B, bias, C))
    run()
    cp.cuda.Device().synchronize()

    # Reference: float64 on the GPU.
    ref = gelu_tanh(A.astype(cp.float64) @ B.astype(cp.float64) + bias.astype(cp.float64))
    unwritten = int(cp.isnan(C).sum())
    max_err = float(cp.abs(C - ref).max())
    ok = unwritten == 0 and bool(cp.allclose(C, ref, rtol=1e-3, atol=1e-4))
    print(f"correctness: unwritten={unwritten}  max_abs_err={max_err:.2e}  -> {'PASS' if ok else 'FAIL'}")

    flops = 2.0 * n ** 3
    ms = time_ms(run)
    cublas_ms = time_ms(lambda: cp.matmul(A, B))
    # What an unfused framework would do: cuBLAS SGEMM, then bias + GELU as a separate elementwise pass.
    unfused_ms = time_ms(lambda: gelu_tanh(cp.matmul(A, B) + bias))
    res = {
        "gpu": gpu, "n": n, "kernel": kernel_name, "grid": grid, "block": block, "correct": ok,
        "max_abs_err": max_err, "mlir_ms": ms, "mlir_gflops": flops / ms / 1e6,
        "cublas_sgemm_ms": cublas_ms, "cublas_sgemm_gflops": flops / cublas_ms / 1e6,
        "cublas_plus_unfused_epilogue_ms": unfused_ms,
    }
    print(f"MLIR kernel (GEMM + bias + GELU fused): {ms:8.3f} ms  {res['mlir_gflops']:8.1f} GFLOP/s")
    print(f"cuBLAS SGEMM alone:                     {cublas_ms:8.3f} ms  {res['cublas_sgemm_gflops']:8.1f} GFLOP/s")
    print(f"cuBLAS SGEMM + separate bias/GELU:      {unfused_ms:8.3f} ms")
    print(f"MLIR kernel reaches {100 * cublas_ms / ms:.1f}% of cuBLAS SGEMM throughput")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
