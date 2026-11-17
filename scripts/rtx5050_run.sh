#!/usr/bin/env bash
# One-command run for an RTX 5050 Laptop GPU (Blackwell, sm_120) under WSL2 or Linux.
#   bash scripts/rtx5050_run.sh
# Needs: NVIDIA driver, CUDA toolkit 12.8 or newer (older nvcc cannot target sm_120), g++ >= 12, python3.
# Plug the laptop in and select its performance power mode first: laptop GPU clocks follow the
# power limit, and the numbers move with them.
# STATUS: the tuner profile and every CPU-side step are tested; this script has NOT been run on a GPU yet.
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
nvcc --version | tail -n 2

make all bank build/ptxforge-tests
./build/ptxforge-tests
python3 tests/parity_check.py build/ptxforge-tune
make cuda
mkdir -p runs

echo
echo "== 1/3  smoke test: 5 trials at 1024^3 (the correctness gate runs before any timing)"
./build/ptxforge-tune-cuda tune --runner cuda --gpu rtx5050 --M 1024 --N 1024 --K 1024 --budget 5 --out runs/rtx5050_smoke.json

echo
echo "== 2/3  tuning run: 40 trials at 2048^3 with the cuBLAS baseline"
./build/ptxforge-tune-cuda tune --runner cuda --gpu rtx5050 --advisor kg --M 2048 --N 2048 --K 2048 --budget 40 --out runs/rtx5050_2048.json
python3 scripts/make_studio.py runs/rtx5050_2048.json studio/rtx5050.html

echo
echo "== 3/3  the MLIR-generated PTX (compiled for sm_75; the driver JIT-compiles it for this GPU)"
if python3 -c "import cupy" 2>/dev/null; then
    python3 scripts/run_ptx_on_gpu.py --json runs/mlir_rtx5050.json
else
    echo "skipped: CuPy is not installed. Install it (pip install cupy-cuda13x, or cupy-cuda12x for a 12.x toolkit) and run:"
    echo "  python3 scripts/run_ptx_on_gpu.py --json runs/mlir_rtx5050.json"
fi

echo
echo "Done. Commit runs/rtx5050_2048.json (and runs/mlir_rtx5050.json if it was written); the README numbers come from those files."
