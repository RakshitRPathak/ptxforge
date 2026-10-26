#!/usr/bin/env python3
"""Parity: the C++ cost model (ptxforge-tune dump) must match the Python reference implementation
(reference/py_tuner) on EVERY valid config. Run via `make parity`."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "py_tuner"))
from ptxforge_tuner.cost_model import analyze  # noqa: E402
from ptxforge_tuner.hw import get_gpu  # noqa: E402
from ptxforge_tuner.space import Config, Problem, enumerate_space  # noqa: E402

exe = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "ptxforge-tune")
bad = 0
for gpu in ("t4", "p100", "l4", "a100"):
    for size in (1024, 2048, 4096):
        cpp = json.loads(subprocess.run([exe, "dump", "--gpu", gpu, "--M", str(size), "--N", str(size), "--K", str(size)],
                                        check=True, capture_output=True, text=True).stdout)
        prob, hw = Problem(size, size, size), get_gpu(gpu)
        py = enumerate_space(prob, hw)
        if [Config(**{k: e["cfg"][k] for k in ("BM", "BN", "BK", "TM", "TN", "PAD")}) for e in cpp] != py:
            print(f"FAIL {gpu} {size}: enumerated spaces differ ({len(cpp)} vs {len(py)})")
            bad += 1
            continue
        worst = 0.0
        for e, c in zip(cpp, py):
            a, p = e["analysis"], analyze(c, prob, hw)
            for key in ("ai", "roof_gflops", "occupancy", "waves", "predicted_gflops"):
                ref = getattr(p, key)
                err = abs(a[key] - ref) - 5.1e-5  # the trace JSON rounds these fields to 4 decimals
                worst = max(worst, err / max(1.0, abs(ref)))
            for key in ("blocks_per_sm", "spill_regs", "reg_cap", "bank_degree", "memory_bound"):
                if a[key] != getattr(p, key):
                    print(f"FAIL {gpu} {size} {c}: {key} {a[key]} != {getattr(p, key)}")
                    bad += 1
            if a["bank"] != p.bank:
                print(f"FAIL {gpu} {size} {c}: bank {a['bank']} != {p.bank}")
                bad += 1
        print(f"{gpu:5s} {size}: {len(py):5d} configs, worst error beyond 4-decimal rounding: {max(worst, 0):.1e}")
        if worst > 1e-6:
            bad += 1
print("PARITY OK" if not bad else f"PARITY FAILED ({bad})")
sys.exit(1 if bad else 0)
