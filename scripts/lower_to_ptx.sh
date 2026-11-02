#!/usr/bin/env bash
# Lower the fused GEMM + bias + GELU kernel from linalg to PTX, stage by stage.
# Every stage writes its IR to $OUT so the studio (and you) can see what each step did.
#
#   scripts/lower_to_ptx.sh compiler/samples/gemm_bias_gelu.mlir build/ir sm_75
#
# Tile sizes come from $OUT/schedule.mlir. If it is missing, the default config (64x64 block tile,
# 4x4 per thread, BK 16) is rendered with `ptxforge-tune schedule`, or with sed if the tuner isn't built.
# Needs: ptxforge-opt (PTXFORGE_OPT) built against MLIR 19, and python3. No CUDA toolkit: the PTX comes
# from LLVM's NVPTX backend, and GELU's tanh is approximated with arith ops, so libdevice isn't linked.
set -euo pipefail

IN=${1:?usage: lower_to_ptx.sh input.mlir [out_dir] [sm_arch]}
OUT=${2:-build/ir}
SM=${3:-sm_75}                      # T4=sm_75, P100=sm_60, A100=sm_80, L4=sm_89
HERE=$(cd "$(dirname "$0")" && pwd)
OPT=${PTXFORGE_OPT:-ptxforge-opt}
TUNE=${PTXFORGE_TUNE:-$HERE/../build/ptxforge-tune}
mkdir -p "$OUT"

if [[ ! -f "$OUT/schedule.mlir" ]]; then
  if [[ -x "$TUNE" ]]; then
    "$TUNE" schedule --BM 64 --BN 64 --BK 16 --TM 4 --TN 4 --out "$OUT/schedule.mlir" >/dev/null
  else
    sed -e 's/{{BM}}/64/g; s/{{BN}}/64/g; s/{{BK}}/16/g; s/{{TM}}/4/g; s/{{TN}}/4/g; s/{{TX}}/16/g; s/{{TY}}/16/g' \
        "$HERE/schedule_template.mlir" > "$OUT/schedule.mlir"
  fi
  echo "(rendered the default schedule into $OUT/schedule.mlir)"
fi

echo "[1/6] fuse epilogue"
"$OPT" --ptxforge-fuse-epilogue "$IN" > "$OUT/01_fused.mlir"

echo "[2/6] tile, fuse, bufferize, map to blocks and threads (transform schedule)"
"$OPT" --transform-preload-library="transform-library-paths=$OUT/schedule.mlir" --transform-interpreter \
       --canonicalize --cse "$OUT/01_fused.mlir" > "$OUT/02_gpu_mapped.mlir"

echo "[3/6] clean up and lower the per-thread work to loops"
"$OPT" --ptxforge-erase-self-copies \
       --convert-linalg-to-loops \
       --promote-buffers-to-stack="max-alloc-size-in-bytes=1024" \
       --ptxforge-approximate-math \
       --convert-vector-to-scf="full-unroll=true" \
       --canonicalize --cse "$OUT/02_gpu_mapped.mlir" > "$OUT/03_loops.mlir"

echo "[4/6] shared-memory padding"
# No-op for this schedule: A/B tiles are not staged in shared memory yet (docs/ROADMAP.md, M2).
"$OPT" --ptxforge-pad-workgroup-memory "$OUT/03_loops.mlir" > "$OUT/04_padded.mlir"

echo "[5/6] outline the kernel, lower to NVVM and emit PTX ($SM)"
# Constants must be inside the launch when it is outlined, or each becomes a kernel parameter.
# Greedy rewrites (canonicalize, the padding pass) hoist them out, so sink them back right before.
# math.fma on 2-D vectors (from the vectorized tanh) is lowered here: convert-gpu-to-nvvm in LLVM 19
# only handles scalar and 1-D math ops.
"$OPT" --gpu-launch-sink-index-computations \
       --convert-math-to-llvm \
       --gpu-lower-to-nvvm-pipeline="cubin-chip=$SM cubin-format=isa opt-level=3 kernel-bare-ptr-calling-convention=true" \
       "$OUT/04_padded.mlir" > "$OUT/05_binary.mlir"

echo "[6/6] extract PTX"
python3 "$HERE/extract_ptx.py" "$OUT/05_binary.mlir" "$OUT/06_kernel.ptx"
if grep -q "__nv_" "$OUT/06_kernel.ptx"; then
  echo "error: the PTX calls libdevice ($(grep -o '__nv_[a-z0-9_]*' "$OUT/06_kernel.ptx" | sort -u | tr '\n' ' '))" >&2
  exit 1
fi
echo "done: $OUT/06_kernel.ptx"
