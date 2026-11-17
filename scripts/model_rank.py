#!/usr/bin/env python3
"""How well does the cost model rank configs that were actually measured?

    python3 scripts/model_rank.py runs/rtx5050_2048.json [--json out.json]

Re-scores every successful trial in a tuning trace with the *current* cost model (the Python reference,
which tests/parity_check.py keeps identical to the C++ one) and reports the Spearman rank correlation
with the measured GFLOP/s, next to the correlation of the predictions recorded in the trace when it ran.
Standard library only.

A trace the model was tuned against is in-sample: a high number there shows the model can describe that
run, not that it predicts new ones. The test that counts is a trace recorded after the model changed.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference", "py_tuner"))
from ptxforge_tuner.cost_model import analyze  # noqa: E402
from ptxforge_tuner.hw import GPU  # noqa: E402
from ptxforge_tuner.space import Config, Problem  # noqa: E402


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", help="write the per-trial table here")
    a = ap.parse_args()
    t = json.load(open(a.trace))
    g = t["gpu"]
    hw = GPU(**{k: g[k] for k in ("name", "arch", "sms", "peak_gflops", "bw_gbs", "regs_per_sm",
                                              "max_threads_sm", "max_blocks_sm", "smem_per_sm", "smem_per_block",
                                              "sustained")})
    prob = Problem(t["problem"]["M"], t["problem"]["N"], t["problem"]["K"])
    rows = []
    for tr in t["trials"]:
        if not tr.get("ok") or tr.get("gflops") is None:
            continue
        c = Config(**{k: tr["cfg"][k] for k in ("BM", "BN", "BK", "TM", "TN", "PAD")})
        an = analyze(c, prob, hw)
        rows.append({"i": tr["i"], "cfg": tr["cfg"], "measured": tr["gflops"],
                     "recorded": tr["modeled"]["predicted_gflops"], "current": round(an.predicted_gflops, 4),
                     "f_pipe": round(an.f_pipe, 4), "f_lat": round(an.f_lat, 4)})
    meas = [r["measured"] for r in rows]
    rho_rec = spearman([r["recorded"] for r in rows], meas)
    rho_cur = spearman([r["current"] for r in rows], meas)
    best = max(rows, key=lambda r: r["current"])
    print(f"{a.trace}: {len(rows)} measured trials on {g['name']}")
    print(f"  Spearman, model recorded in the trace: {rho_rec:+.3f}")
    print(f"  Spearman, current model:               {rho_cur:+.3f}")
    print(f"  current model's top pick measured {best['measured']:.0f} GFLOP/s (best measured {max(meas):.0f})")
    if a.json:
        json.dump({"trace": a.trace, "spearman_recorded": rho_rec, "spearman_current": rho_cur, "trials": rows},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
