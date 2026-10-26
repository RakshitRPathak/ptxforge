#!/usr/bin/env bash
# Colab / Kaggle (T4, P100) setup for milestones M0-M3 — no LLVM build needed.
# In a notebook cell:   !bash scripts/colab_setup.sh
# STATUS: the CUDA build below has never been run by the author (no GPU in the authoring sandbox).
set -euo pipefail

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version | tail -n 2

make all bank build/ptxforge-tests
./build/ptxforge-tests
python3 tests/parity_check.py build/ptxforge-tune

# the real GPU runner
make cuda

echo
echo "Smoke test (real GPU, 5 trials; the correctness gate runs before any timing):"
echo "  ./build/ptxforge-tune-cuda tune --runner cuda --gpu t4 --M 1024 --N 1024 --K 1024 --budget 5 --out runs/smoke.json"
