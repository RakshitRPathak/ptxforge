from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .advisors import make_advisor
from .cost_model import analyze
from .hw import GPUS, get_gpu
from .runners import HarnessRunner, SimRunner
from .schedule import render
from .search import tune
from .space import DEFAULT, Config, Problem, enumerate_space


def _problem(a) -> Problem:
    return Problem(a.M, a.N, a.K)


def _runner(a, seed=0):
    if a.runner == "sim":
        return SimRunner(seed=seed)
    return HarnessRunner(bench=a.bench, nvcc=a.nvcc, cublas=not a.no_cublas)


def cmd_tune(a) -> int:
    hw, prob = get_gpu(a.gpu), _problem(a)
    runner = _runner(a, a.seed)
    if a.runner == "sim":
        print("!! SIMULATED RUN — cost model + noise, no GPU. Output is flagged and must not be quoted.\n")
    trace = tune(prob, hw, runner, make_advisor(a.advisor), a.budget, a.seed, log=print)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(trace, indent=1))
    print(f"\nsource={trace['source']}  space={trace['space_size']} configs  trials={len(trace['trials'])}")
    print(f"baseline {trace['baseline_gflops']} -> best {trace['best_gflops']} GFLOP/s "
          f"(x{trace['speedup_vs_default']})")
    if trace["pct_of_cublas"]:
        print(f"best = {trace['pct_of_cublas']}% of cuBLAS SGEMM ({trace['cublas_gflops']} GFLOP/s)")
    print(f"cost-model rank correlation (Spearman): {trace['model_spearman']}")
    print(f"trace -> {a.out}")
    return 0


def cmd_ablate(a) -> int:
    hw, prob = get_gpu(a.gpu), _problem(a)
    if a.runner == "sim":
        print("!! SIMULATED ablation: demonstrates the harness only. The sim world is built from the same\n"
              "   cost model the advisors reason with, so it cannot tell you whether the KG helps on real\n"
              "   hardware. Re-run with --runner harness on a GPU for evidence.\n")
    rows = []
    for name in a.advisors.split(","):
        bests = []
        for s in range(a.seeds):
            tr = tune(prob, hw, _runner(a, s), make_advisor(name), a.budget, s,
                      seed_trials=None if a.warm_start else 1)
            bests.append(tr["best_gflops"])
        rows.append((name, statistics.mean(bests), statistics.pstdev(bests), max(bests)))
    print(f"{'advisor':10s} {'mean best':>10s} {'std':>8s} {'max':>10s}   (GFLOP/s, budget={a.budget}, seeds={a.seeds})")
    for n, m, sd, mx in sorted(rows, key=lambda r: -r[1]):
        print(f"{n:10s} {m:10.1f} {sd:8.1f} {mx:10.1f}")
    return 0


def cmd_schedule(a) -> int:
    cfg = Config(a.BM, a.BN, a.BK, a.TM, a.TN, a.PAD)
    text = render(cfg)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


def cmd_gpus(_a) -> int:
    for k, g in GPUS.items():
        print(f"{k:5s} {g.name:18s} {g.arch}  SMs={g.sms:3d}  peak={g.peak_gflops:7.0f} GFLOP/s  "
              f"bw={g.bw_gbs:6.0f} GB/s  ridge={g.ridge:5.1f} FLOP/B")
    return 0


def cmd_space(a) -> int:
    hw, prob = get_gpu(a.gpu), _problem(a)
    sp = enumerate_space(prob, hw)
    top = sorted(sp, key=lambda c: -analyze(c, prob, hw).predicted_gflops)[:10]
    print(f"{len(sp)} valid configs for {prob.M}x{prob.N}x{prob.K} on {hw.name}\nTop 10 by cost model:")
    for c in top:
        an = analyze(c, prob, hw)
        print(f"  {c.short():22s} pred={an.predicted_gflops:7.0f}  AI={an.ai:5.1f}  occ={an.occupancy:.2f}  "
              f"bank={an.bank_degree}-way  spill={an.spill_regs}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ptxforge_tuner")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--gpu", default="t4", choices=list(GPUS))
        sp.add_argument("--M", type=int, default=2048)
        sp.add_argument("--N", type=int, default=2048)
        sp.add_argument("--K", type=int, default=2048)

    def runner_args(sp):
        sp.add_argument("--runner", default="sim", choices=["sim", "harness"])
        sp.add_argument("--bench", default="build/ptxforge-bench")
        sp.add_argument("--nvcc", default="nvcc")
        sp.add_argument("--no-cublas", action="store_true")

    t = sub.add_parser("tune", help="run the closed-loop autotuner")
    common(t); runner_args(t)
    t.add_argument("--advisor", default="kg", choices=["none", "random", "rules", "kg", "llm"])
    t.add_argument("--budget", type=int, default=30)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs/trace.json")
    t.set_defaults(fn=cmd_tune)

    ab = sub.add_parser("ablate", help="compare advisors across seeds")
    common(ab); runner_args(ab)
    ab.add_argument("--advisors", default="none,random,rules,kg")
    ab.add_argument("--budget", type=int, default=20)
    ab.add_argument("--seeds", type=int, default=5)
    ab.add_argument("--warm-start", action="store_true",
                    help="seed with the cost model's top picks (default: cold start, so advisors are compared on their own)")
    ab.set_defaults(fn=cmd_ablate)

    s = sub.add_parser("schedule", help="render the MLIR transform schedule for a config")
    for k, d in (("BM", DEFAULT.BM), ("BN", DEFAULT.BN), ("BK", DEFAULT.BK),
                 ("TM", DEFAULT.TM), ("TN", DEFAULT.TN), ("PAD", DEFAULT.PAD)):
        s.add_argument(f"--{k}", type=int, default=d)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_schedule)

    g = sub.add_parser("gpus", help="list built-in GPU specs"); g.set_defaults(fn=cmd_gpus)
    sp = sub.add_parser("space", help="show search-space size and cost-model top picks")
    common(sp); sp.set_defaults(fn=cmd_space)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
