"""Closed-loop search: cost-model seeding -> advisor proposals / neighbor hill-climb -> measure."""
from __future__ import annotations

import random
from datetime import datetime, timezone

from .advisors import Advisor, Context
from .cost_model import analyze
from .hw import GPU
from .kg import diagnose
from .runners import Runner
from .space import DEFAULT, Config, Problem, enumerate_space, is_valid, neighbors


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def tune(prob: Problem, hw: GPU, runner: Runner, advisor: Advisor, budget: int = 30,
         seed: int = 0, log=lambda *_: None, seed_trials: int | None = None) -> dict:
    rng = random.Random(seed)
    space = enumerate_space(prob, hw)
    if not space:
        raise SystemExit("empty search space: problem size not divisible by any tile")
    pred = {c: analyze(c, prob, hw) for c in space}
    trials: list[dict] = []
    measured: set[Config] = set()
    history: list[dict] = []

    def run(cfg: Config, origin: str, why: str) -> dict:
        an = pred.get(cfg) or analyze(cfg, prob, hw)
        m = runner.measure(cfg, prob, hw)
        measured.add(cfg)
        symptoms = diagnose(cfg, an, m.profile, hw) if m.ok else []
        t = {
            "i": len(trials), "cfg": cfg.to_json(), "origin": origin, "why": why,
            "ok": m.ok, "error": m.error, "gflops": round(m.gflops, 2) if m.gflops else None,
            "ms": round(m.ms, 4) if m.ms else None, "profile": m.profile, "symptoms": symptoms,
            "modeled": {k: v for k, v in an.to_json().items()
                        if k in ("ai", "ai_ideal", "roof_gflops", "memory_bound", "occupancy", "bank",
                                 "bank_degree", "spill_regs", "predicted_gflops", "waves")},
            "cublas_gflops": m.cublas_gflops,
        }
        trials.append(t)
        history.append({"cfg": cfg, "gflops": t["gflops"]})
        log(f"[{t['i']:3d}] {cfg.short():22s} {t['origin']:14s} "
            f"{(str(t['gflops']) + ' GFLOP/s') if t['ok'] else 'FAILED: ' + str(t['error'])[:60]}")
        return t

    def best_trial() -> dict | None:
        ok = [t for t in trials if t["ok"]]
        return max(ok, key=lambda t: t["gflops"]) if ok else None

    # 0. baseline: the untuned default — all speedups are reported against THIS measured number
    base_cfg = DEFAULT if is_valid(DEFAULT, prob, hw) else max(space, key=lambda c: pred[c].predicted_gflops)
    run(base_cfg, "baseline", "untuned default")

    # 1. seed with the cost model's top picks
    ranked = sorted(space, key=lambda c: -pred[c].predicted_gflops)
    n_seed = seed_trials if seed_trials is not None else max(2, budget // 4)  # 1 = cold start (baseline only)
    for c in ranked:
        if len(trials) >= n_seed:
            break
        if c not in measured:
            run(c, "cost-model", "top predicted")

    # 2. closed loop
    while len(trials) < budget:
        bt = best_trial()
        pick, origin, why = None, "", ""
        if bt is not None:
            best = Config(**{k: bt["cfg"][k] for k in ("BM", "BN", "BK", "TM", "TN", "PAD")})
            ctx = Context(prob, hw, best, pred[best], bt["profile"], bt["symptoms"], history,
                          measured, rng)
            for p in advisor.propose(ctx):
                if p.cfg not in measured and is_valid(p.cfg, prob, hw):
                    pick, origin, why = p.cfg, f"advisor:{advisor.name}", p.why
                    break
            if pick is None:
                cand = [n for n in neighbors(best, prob, hw) if n not in measured]
                if cand:
                    pick = max(cand, key=lambda c: pred[c].predicted_gflops)
                    origin, why = "neighbor", "best-predicted neighbor of incumbent"
        if pick is None:
            rest = [c for c in ranked if c not in measured]
            if not rest:
                break
            pick, origin, why = rest[0], "cost-model", "next best predicted"
        run(pick, origin, why)

    bt = best_trial()
    base = trials[0]
    both = [(t["modeled"]["predicted_gflops"], t["gflops"]) for t in trials if t["ok"]]
    cub = next((t["cublas_gflops"] for t in trials if t.get("cublas_gflops")), None)
    best_cfg = Config(**{k: bt["cfg"][k] for k in ("BM", "BN", "BK", "TM", "TN", "PAD")}) if bt else None
    sim = runner.source == "simulated"
    return {
        "schema": 1,
        "source": runner.source,
        "warning": ("SIMULATED: numbers come from the analytic cost model plus noise, not a GPU. "
                    "Do not quote them as results.") if sim else None,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem": {"M": prob.M, "N": prob.N, "K": prob.K, "flops": prob.flops},
        "gpu": hw.to_json(),
        "advisor": advisor.name,
        "budget": budget,
        "seed": seed,
        "space_size": len(space),
        "baseline_gflops": base["gflops"],
        "best_gflops": bt["gflops"] if bt else None,
        "best_index": bt["i"] if bt else None,
        "speedup_vs_default": round(bt["gflops"] / base["gflops"], 3) if bt and base["gflops"] else None,
        "cublas_gflops": cub,
        "pct_of_cublas": round(100 * bt["gflops"] / cub, 1) if bt and cub else None,
        "model_spearman": spearman(*zip(*both)) if len(both) >= 3 else None,
        "trials": trials,
        "stages": runner.stages(best_cfg, prob, hw) if best_cfg else [],
    }
