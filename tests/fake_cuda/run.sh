#!/usr/bin/env bash
# Builds CudaRunner against real CUDA headers and a fake driver (no GPU needed), then checks:
#   1. a driver that rejects nvcc's PTX as too new: the runner switches to cubins once and every trial passes
#   2. a driver that accepts the PTX: the runner stays on PTX and builds no cubins
# Usage: tests/fake_cuda/run.sh <dir containing cuda.h etc.>...   (see make cuda-fallback-test)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
INC=()
for d in "$@"; do INC+=("-I$d"); done

g++ -std=c++20 -O1 -fPIC -shared "${INC[@]}" "$ROOT/tests/fake_cuda/fake_cuda.cpp" -o "$T/libfakecuda.so"
g++ -std=c++20 -O1 -Wall -Wextra -I"$ROOT/core/include" -I"$ROOT/compiler/include" -DPTXFORGE_ROOT_DIR="\"$ROOT\"" \
    -DPTXFORGE_WITH_CUDA -DPTXFORGE_WITH_CUBLAS "${INC[@]}" \
    "$ROOT/core/src/main.cpp" "$ROOT/core/src/cuda_runner.cpp" -o "$T/tune-cuda" -L"$T" -lfakecuda -Wl,-rpath,"$T"

fail=0
check() {  # name, expected "ptx cubin" call counts, env...
    local name="$1" want="$2"; shift 2
    rm -rf "$T/k"; : > "$T/nvcc.log"
    env FAKE_NVCC_LOG="$T/nvcc.log" "$@" "$T/tune-cuda" tune --runner cuda --gpu rtx5050 --M 128 --N 128 --K 128 \
        --budget 6 --nvcc "$ROOT/tests/fake_cuda/fake_nvcc" --workdir "$T/k" --out "$T/out.json" > "$T/log" 2>&1
    local got ok
    got="$(grep -c '^ptx$' "$T/nvcc.log") $(grep -c '^cubin$' "$T/nvcc.log" || true)"
    ok="$(python3 -c "import json; d=json.load(open('$T/out.json')); print(sum(t['ok'] for t in d['trials']), len(d['trials']))")"
    if [ "$got" = "$want" ] && [ "$ok" = "6 6" ]; then echo "[ ok ] $name (nvcc ptx/cubin calls: $got, trials ok: 6/6)"
    else echo "[FAIL] $name: nvcc ptx/cubin calls '$got' (want '$want'), trials ok/total '$ok'"; tail -20 "$T/log"; fail=1; fi
}
check "driver older than the toolkit falls back to cubins" "6 6"
grep -q "loading cubins" "$T/log" && [ "$(grep -c 'cannot JIT' "$T/log")" = 1 ] \
    && echo "[ ok ] the fallback is announced once on stderr" || { echo "[FAIL] fallback notice missing or repeated"; fail=1; }
check "current driver stays on the PTX path" "6 0" FAKE_ACCEPT_PTX=1
exit $fail
