#!/usr/bin/env python3
"""Check that the compiler's tiling, fusion and bufferization compute the right numbers.

There is no GPU here, so this runs the same schedule as scripts/lower_to_ptx.sh minus the final
"map loops onto blocks and threads" step, compiles the result for the CPU, and compares it with a
float64 NumPy reference. scripts/run_ptx_on_gpu.py does the same check for the real PTX on a GPU.

    python3 tests/check_lowering_cpu.py --opt build-mlir/ptxforge-opt [--n 256] [--tile 64,64,16,4,4]

Needs ptxforge-opt, mlir-translate and clang from LLVM 19, and numpy.
"""
import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLVM_BIN = os.environ.get("LLVM_BIN_DIR", "/usr/lib/llvm-19/bin")


def tool(name, fallback):
    for cand in (os.path.join(LLVM_BIN, name), shutil.which(name), shutil.which(fallback)):
        if cand and os.path.exists(cand):
            return cand
    sys.exit(f"cannot find {name}")


def run(cmd, out=None):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr}")
    if out:
        open(out, "w").write(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default=os.path.join(ROOT, "build-mlir", "ptxforge-opt"))
    ap.add_argument("--n", type=int, default=256, help="M = N = K")
    ap.add_argument("--tile", default="64,64,16,4,4", help="BM,BN,BK,TM,TN")
    a = ap.parse_args()
    bm, bn, bk, tm, tn = (int(x) for x in a.tile.split(","))
    n = a.n
    if n % bm or n % bn or n % bk or bm % tm or bn % tn:
        sys.exit("n must be a multiple of BM, BN and BK, and BM/BN multiples of TM/TN")

    tmp = tempfile.mkdtemp(prefix="ptxforge-cpu-")
    src = open(os.path.join(ROOT, "compiler", "samples", "gemm_bias_gelu.mlir")).read().replace("2048", str(n))
    open(f"{tmp}/in.mlir", "w").write(src)

    sched = open(os.path.join(ROOT, "scripts", "schedule_template.mlir")).read()
    for k, v in dict(BM=bm, BN=bn, BK=bk, TM=tm, TN=tn, TX=bn // tn, TY=bm // tm).items():
        sched = sched.replace("{{%s}}" % k, str(v))
    # Same schedule, minus the GPU mapping: the block and thread loops stay scf.forall and run serially.
    sched = re.sub(r"\n\s*%launch = transform\.gpu\.map_forall_to_blocks.*?: \(!transform\.any_op\) -> !transform\.any_op", "", sched, flags=re.S)
    sched = re.sub(r"\n\s*transform\.gpu\.map_nested_forall_to_threads.*?: \(!transform\.any_op\) -> !transform\.any_op", "", sched, flags=re.S)
    assert "transform.gpu" not in sched, "failed to strip the GPU mapping from the schedule"
    open(f"{tmp}/schedule.mlir", "w").write(sched)

    opt = a.opt
    run([opt, "--ptxforge-fuse-epilogue", f"{tmp}/in.mlir"], f"{tmp}/01.mlir")
    run([opt, f"--transform-preload-library=transform-library-paths={tmp}/schedule.mlir", "--transform-interpreter",
         "--canonicalize", "--cse", "--ptxforge-erase-self-copies", f"{tmp}/01.mlir"], f"{tmp}/02.mlir")
    if "memref.copy" in open(f"{tmp}/02.mlir").read() or "linalg.copy" in open(f"{tmp}/02.mlir").read():
        sys.exit("unexpected copy left after bufferization (the result should be written into C in place)")
    run([opt, "--scf-forall-to-for", "--convert-linalg-to-loops", "--promote-buffers-to-stack=max-alloc-size-in-bytes=1024",
         "--ptxforge-approximate-math", "--canonicalize", "--convert-vector-to-scf=full-unroll=true", "--expand-strided-metadata", "--lower-affine",
         "--convert-scf-to-cf", "--convert-vector-to-llvm", "--finalize-memref-to-llvm", "--convert-math-to-llvm", "--convert-arith-to-llvm",
         "--convert-func-to-llvm=use-bare-ptr-memref-call-conv=1", "--convert-cf-to-llvm",
         "--reconcile-unrealized-casts", f"{tmp}/02.mlir"], f"{tmp}/03.mlir")
    run([tool("mlir-translate", "mlir-translate-19"), "--mlir-to-llvmir", f"{tmp}/03.mlir"], f"{tmp}/k.ll")
    run([tool("clang", "clang-19"), "-O2", "-shared", "-fPIC", "-Wno-override-module", f"{tmp}/k.ll", "-o", f"{tmp}/k.so"])

    lib = ctypes.CDLL(f"{tmp}/k.so")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((n, n)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((n, n)) * 0.1).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    C = np.full((n, n), np.nan, dtype=np.float32)
    ptr = lambda x: x.ctypes.data_as(ctypes.c_void_p)
    lib.gemm_bias_gelu(ptr(A), ptr(B), ptr(bias), ptr(C))

    x = A.astype(np.float64) @ B.astype(np.float64) + bias
    ref = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    unwritten = int(np.isnan(C).sum())
    err = float(np.abs(C - ref).max())
    ok = unwritten == 0 and np.allclose(C, ref, rtol=1e-4, atol=1e-5)
    print(f"n={n} tile={a.tile}: unwritten={unwritten} max_abs_err={err:.2e} -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(tmp)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
