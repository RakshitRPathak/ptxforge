#!/usr/bin/env bash
# Time the MLIR-generated kernel for several tile configs, with and without shared-memory staging.
#
#   bash scripts/mlir_sweep.sh [out_dir]          # default runs/mlir_sweep
#
# Each kernels/generated/sweep/<BMxBNxBK_tTMxTN>_<smem|global>.ptx was produced by scripts/lower_to_ptx.sh:
# "smem" with the current schedule (A/B tiles staged in shared memory), "global" with the previous one
# (each thread reads A/B from global memory through the caches). Every run checks the output against an
# fp64 reference first and times cuBLAS in the same process, so compare the "% of cuBLAS" column, not raw
# GFLOP/s across sessions (laptop GPUs change clocks with the power mode).
# Needs CuPy (pip install cupy-cuda13x or cupy-cuda12x).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${1:-$ROOT/runs/mlir_sweep}
mkdir -p "$OUT"
# Refuse to time anything that is not what its name says (a stale checkout once produced a "result" for the
# wrong kernel; docs/DEVLOG.md, 2026-11-17).
shopt -s nullglob
files=("$ROOT"/kernels/generated/sweep/*.ptx)
[[ ${#files[@]} -eq 10 ]] || { echo "error: expected 10 kernels in kernels/generated/sweep, found ${#files[@]} (git pull?)" >&2; exit 1; }
for ptx in "${files[@]}"; do
  n=$(grep -c "ld.shared" "$ptx" || true)
  case "$ptx" in
    *_smem.ptx)   [[ $n -gt 0 ]] || { echo "error: $ptx has no shared-memory loads" >&2; exit 1; } ;;
    *_global.ptx) [[ $n -eq 0 ]] || { echo "error: $ptx uses shared memory" >&2; exit 1; } ;;
  esac
done
echo "commit $(git -C "$ROOT" rev-parse --short HEAD); 10 kernels checked (5 staged, 5 unstaged)"

for ptx in "${files[@]}"; do
  name=$(basename "$ptx" .ptx)
  bm=${name%%x*}; rest=${name#*x}; bn=${rest%%x*}
  echo "== $name"
  python3 "$HERE/run_ptx_on_gpu.py" --ptx "$ptx" --bm "$bm" --bn "$bn" --json "$OUT/$name.json" | tail -3
done
python3 - "$OUT" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if f.endswith("summary.json"):
        continue
    r = json.load(open(f))
    name = os.path.basename(f)[:-5]
    tile, kind = name.rsplit("_", 1)
    rows.append(dict(tile=tile, kind=kind, correct=r["correct"], gflops=r["mlir_gflops"],
                     pct=100 * r["mlir_gflops"] / r["cublas_sgemm_gflops"], cublas=r["cublas_sgemm_gflops"]))
print(f"\n{'tile':18s} {'staging':8s} {'correct':8s} {'GFLOP/s':>9s} {'% cuBLAS':>9s}")
for r in sorted(rows, key=lambda r: (r["tile"], r["kind"])):
    print(f"{r['tile']:18s} {r['kind']:8s} {str(r['correct']):8s} {r['gflops']:9.0f} {r['pct']:8.1f}%")
json.dump(rows, open(os.path.join(sys.argv[1], "summary.json"), "w"), indent=1)
print(f"\nwrote {sys.argv[1]}/summary.json")
PY
