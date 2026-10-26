#!/usr/bin/env bash
# Staged MLIR -> PTX lowering. Each stage dumps its IR to $OUT/ so the studio can show it.
# STATUS: skeleton. Stages 1-2 are the project's own work; stages 3+ are the standard upstream
# GPU pipeline and need tuning to your LLVM version (milestone M1). Fail loudly, stage by stage.
set -euo pipefail

IN=${1:?usage: lower_to_ptx.sh input.mlir [out_dir] [sm_arch]}
OUT=${2:-build/ir}
SM=${3:-sm_75}                      # T4=sm_75, P100=sm_60, A100=sm_80, L4=sm_89
OPT=${PTXFORGE_OPT:-ptxforge-opt}
MLIR_TRANSLATE=${MLIR_TRANSLATE:-mlir-translate}
mkdir -p "$OUT"

echo "[1/6] fuse epilogue"
$OPT --ptxforge-fuse-epilogue "$IN" > "$OUT/01_fused.mlir"

echo "[2/6] tile + fuse (transform schedule)"
if [[ -f "$OUT/schedule.mlir" ]]; then
  $OPT --transform-interpreter="entry-point=__transform_main" \
       --split-input-file "$OUT/01_fused.mlir" > "$OUT/02_tiled.mlir" 2>/dev/null \
    || { echo "  transform-interpreter flags differ on your LLVM; see MLIR_API_NOTES.md"; exit 1; }
else
  echo "  (no schedule.mlir in $OUT — render one with: ptxforge-tune schedule --out $OUT/schedule.mlir)"
  cp "$OUT/01_fused.mlir" "$OUT/02_tiled.mlir"
fi

echo "[3/6] bufferize"
$OPT --one-shot-bufferize="bufferize-function-boundaries" \
     --canonicalize "$OUT/02_tiled.mlir" > "$OUT/03_buffers.mlir"

echo "[4/6] shared-memory padding"
$OPT --ptxforge-pad-workgroup-memory "$OUT/03_buffers.mlir" > "$OUT/04_padded.mlir"

echo "[5/6] gpu -> nvvm -> llvm dialect"
$OPT --gpu-kernel-outlining \
     --convert-gpu-to-nvvm \
     --convert-nvvm-to-llvm \
     --canonicalize --cse "$OUT/04_padded.mlir" > "$OUT/05_nvvm.mlir"

echo "[6/6] llvm ir -> ptx"
$MLIR_TRANSLATE --mlir-to-llvmir "$OUT/05_nvvm.mlir" > "$OUT/06.ll"
llc -mtriple=nvptx64-nvidia-cuda -mcpu="$SM" "$OUT/06.ll" -o "$OUT/07.ptx"
echo "done: $OUT/07.ptx"
