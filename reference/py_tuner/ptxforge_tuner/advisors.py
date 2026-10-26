"""Advisors propose configs to measure next. They never bypass validity checks, and every
proposal is measured — an advisor can only be wrong in a way the benchmark will catch.

  none    : no proposals (cost-model + neighbor search only)       -> ablation baseline
  random  : uniformly random valid configs                          -> ablation baseline
  rules   : hard-coded if/else diagnosis -> parameter move          -> "what a human would script"
  kg      : knowledge-graph retrieval with side-effect scoring      -> GraphRAG-lite, deterministic
  llm     : kg retrieval as grounded context + an LLM that proposes -> optional, needs an API key
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from . import kg as kgmod
from .bank import choose_pad
from .cost_model import Analysis
from .hw import GPU
from .space import Config, Problem, enumerate_space, is_valid, step


@dataclass
class Proposal:
    cfg: Config
    why: str


@dataclass
class Context:
    prob: Problem
    hw: GPU
    best: Config
    best_an: Analysis
    best_profile: dict
    symptoms: list[str]
    history: list[dict]            # [{"cfg": Config, "gflops": float|None}, ...]
    measured: set                  # set[Config]
    rng: random.Random


class Advisor:
    name = "base"

    def propose(self, ctx: Context) -> list[Proposal]:  # pragma: no cover
        raise NotImplementedError


class NoneAdvisor(Advisor):
    name = "none"

    def propose(self, ctx):
        return []


class RandomAdvisor(Advisor):
    name = "random"

    def __init__(self):
        self._space = None

    def propose(self, ctx):
        if self._space is None:
            self._space = enumerate_space(ctx.prob, ctx.hw)
        pool = [c for c in self._space if c not in ctx.measured]
        return [Proposal(ctx.rng.choice(pool), "random valid config")] if pool else []


def _apply(ctx: Context, param: str, direction: int, why: str) -> Proposal | None:
    n = step(ctx.best, param, direction)
    if n is not None and is_valid(n, ctx.prob, ctx.hw) and n not in ctx.measured:
        return Proposal(n, why)
    return None


class RuleAdvisor(Advisor):
    name = "rules"

    def propose(self, ctx):
        out: list[Proposal] = []
        s = set(ctx.symptoms)
        if "register_spill" in s:
            for p in ("TM", "TN"):
                out.append(_apply(ctx, p, -1, f"spill -> decrease {p}"))
        if "bank_conflict" in s:
            pad, _ = choose_pad(ctx.best.BM, 4, 4)
            n = Config(**{**ctx.best.__dict__, "PAD": pad})
            if is_valid(n, ctx.prob, ctx.hw) and n not in ctx.measured:
                out.append(Proposal(n, f"bank conflicts -> PAD={pad}"))
        if "memory_bound" in s:
            out.append(_apply(ctx, "BM", +1, "memory-bound -> larger BM"))
            out.append(_apply(ctx, "BN", +1, "memory-bound -> larger BN"))
        if "low_occupancy" in s:
            out.append(_apply(ctx, "TM", -1, "low occupancy -> decrease TM"))
        return [o for o in out if o]


class KGAdvisor(Advisor):
    name = "kg"

    def __init__(self, kg: dict | None = None):
        self.kg = kg or kgmod.load_kg()

    def _materialize(self, ctx: Context, r: kgmod.Remedy) -> Proposal | None:
        why = f"[{r.id}] {r.symptom}: {r.action} {r.param} — {r.why}"
        if r.action == "set_conflict_free":
            pad = ctx.best.PAD
            # smallest PAD in the domain that minimizes the modeled conflict for THIS config
            from .bank import kernel_bank_degrees
            from .space import DOMAINS
            best = min(DOMAINS["PAD"], key=lambda p: (
                sum(kernel_bank_degrees(ctx.best.BM, ctx.best.BN, ctx.best.BK, ctx.best.TM, ctx.best.TN, p).values()), p))
            n = Config(**{**ctx.best.__dict__, "PAD": best})
            return Proposal(n, why) if n != ctx.best and is_valid(n, ctx.prob, ctx.hw) and n not in ctx.measured else None
        direction = +1 if r.action == "increase" else -1
        return _apply(ctx, r.param, direction, why)

    def propose(self, ctx):
        remedies = kgmod.retrieve(ctx.symptoms, self.kg)
        out: list[Proposal] = []
        for r in remedies:
            if r.score <= 0 and out:
                continue
            p = self._materialize(ctx, r)
            if p:
                out.append(p)
        # compound move: apply the top two compatible remedies at once (escapes single-step plateaus)
        if len(out) >= 2:
            a, b = out[0].cfg, out[1].cfg
            merged = {k: getattr(a, k) if getattr(a, k) != getattr(ctx.best, k) else getattr(b, k)
                      for k in a.__dict__}
            m = Config(**merged)
            if is_valid(m, ctx.prob, ctx.hw) and m not in ctx.measured and m not in (a, b):
                out.insert(1, Proposal(m, "compound: " + out[0].why.split(" — ")[0] + " + " + out[1].why.split(" — ")[0]))
        return out


class LLMAdvisor(Advisor):
    """Grounded LLM proposer. The model only *suggests*; validity is enforced in code and every
    suggestion is benchmarked, so hallucinated configs cost one wasted trial at worst."""
    name = "llm"

    def __init__(self, model: str | None = None, kg: dict | None = None):
        self.model = model or os.environ.get("PTXFORGE_LLM_MODEL", "claude-sonnet-5-5")
        self.kg = kg or kgmod.load_kg()
        self.rejected = 0
        self.accepted = 0

    def _prompt(self, ctx: Context) -> str:
        remedies = kgmod.retrieve(ctx.symptoms, self.kg)
        hist = "\n".join(f"  {h['cfg'].short()} -> {h['gflops']} GFLOP/s" for h in ctx.history[-12:])
        return (
            f"You tune a tiled SGEMM+bias+GELU CUDA kernel on {ctx.hw.name}; problem "
            f"{ctx.prob.M}x{ctx.prob.N}x{ctx.prob.K}.\n"
            f"Best so far: {ctx.best.short()} ({ctx.best.to_json()}), profile: {ctx.best_profile}.\n\n"
            f"{kgmod.subgraph_text(ctx.symptoms, remedies)}\n\nRecent trials:\n{hist}\n\n"
            "Allowed values: BM,BN in {32,64,128,256}; BK in {4,8,16,32}; TM,TN in {2,4,8,16}; "
            "PAD in {0,1,2,4,8}. threads=(BM/TM)*(BN/TN) must be a multiple of 32 in [64,1024]; "
            "shared mem 4*BK*(BM+PAD+BN) bytes <= 49152.\n"
            "Propose up to 3 NEW configs (not in the history) most likely to beat the best. "
            'Reply with JSON only: {"proposals":[{"BM":..,"BN":..,"BK":..,"TM":..,"TN":..,"PAD":..,"why":"..."}]}'
        )

    def propose(self, ctx):
        try:
            import anthropic  # lazy: the rest of the package has no API dependency
        except ImportError:
            raise SystemExit("llm advisor needs `pip install anthropic` and ANTHROPIC_API_KEY")
        client = anthropic.Anthropic()
        msg = client.messages.create(model=self.model, max_tokens=800,
                                     messages=[{"role": "user", "content": self._prompt(ctx)}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return self.parse(text, ctx)

    def parse(self, text: str, ctx: Context) -> list[Proposal]:
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            items = json.loads(text[start:end])["proposals"]
        except (ValueError, KeyError, json.JSONDecodeError):
            self.rejected += 1
            return []
        out = []
        for it in items[:3]:
            try:
                c = Config(**{k: int(it[k]) for k in ("BM", "BN", "BK", "TM", "TN", "PAD")})
            except (KeyError, ValueError, TypeError):
                self.rejected += 1
                continue
            if is_valid(c, ctx.prob, ctx.hw) and c not in ctx.measured:
                self.accepted += 1
                out.append(Proposal(c, "LLM: " + str(it.get("why", ""))[:200]))
            else:
                self.rejected += 1  # hallucinated / invalid / duplicate: dropped, never run
        return out


def make_advisor(name: str) -> Advisor:
    table = {"none": NoneAdvisor, "random": RandomAdvisor, "rules": RuleAdvisor,
             "kg": KGAdvisor, "llm": LLMAdvisor}
    if name not in table:
        raise SystemExit(f"unknown advisor '{name}'. choices: {', '.join(table)}")
    return table[name]()
