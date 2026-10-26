"""Symptom diagnosis + knowledge-graph retrieval (the 'GraphRAG-lite' part).

Graph:  symptom --remedy(param, action)--> param move --side_effects--> other symptoms
Retrieval = for each symptom present in the profile, pull its remedy edges, then walk one hop
along `side_effects` to see which *other* symptoms the remedy risks creating, and rank remedies
by (symptom weight) - (penalty for side effects that are already borderline).
The same retrieved subgraph is what the LLM advisor receives as grounded context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .cost_model import Analysis
from .hw import GPU
from .space import Config, Problem

KG_PATH = Path(__file__).resolve().parents[3] / "knowledge" / "pass_graph.json"


def load_kg(path: Path | None = None) -> dict:
    with open(path or KG_PATH) as f:
        return json.load(f)


def diagnose(c: Config, an: Analysis, profile: dict, hw: GPU) -> list[str]:
    """Symptoms, using MEASURED profile fields where available, modeled ones otherwise."""
    s: list[str] = []
    local = profile.get("local_bytes")
    if (local is not None and local > 0) or (local is None and an.spill_regs > 0):
        s.append("register_spill")
    occ = profile.get("occupancy", an.occupancy)
    if occ is not None and occ < 0.25:
        s.append("low_occupancy")
    if an.bank_degree >= 2:
        s.append("bank_conflict")
    if an.memory_bound:
        s.append("memory_bound")
    if c.TM * c.TN < 16:
        s.append("low_ilp")
    if an.waves < 1.0:
        s.append("underfilled_gpu")
    return s


@dataclass
class Remedy:
    id: str
    symptom: str
    param: str
    action: str
    why: str
    side_effects: list[str]
    evidence: str
    score: float


def retrieve(symptoms: list[str], kg: dict) -> list[Remedy]:
    weights = {k: v["weight"] for k, v in kg["symptoms"].items()}
    present = set(symptoms)
    out: list[Remedy] = []
    for r in kg["remedies"]:
        if r["symptom"] not in present:
            continue
        # one-hop walk: a side effect that is ALREADY present is not a new risk (we're already
        # paying for it); a new one is penalized by that symptom's weight.
        penalty = sum(0.5 * weights.get(se, 0.3) for se in r["side_effects"] if se not in present)
        # a remedy that fixes symptom A by worsening another present symptom B is contradictory
        conflict = sum(1.0 for se in r["side_effects"] if se in present and weights.get(se, 0) > weights[r["symptom"]])
        score = weights[r["symptom"]] - penalty - conflict
        out.append(Remedy(r["id"], r["symptom"], r["param"], r["action"], r["why"],
                          r["side_effects"], r.get("evidence", ""), round(score, 3)))
    out.sort(key=lambda x: -x.score)
    return out


def subgraph_text(symptoms: list[str], remedies: list[Remedy]) -> str:
    lines = [f"Observed symptoms: {', '.join(symptoms) or 'none'}", "Candidate remedies (best first):"]
    for r in remedies[:8]:
        lines.append(
            f"- [{r.id}] {r.symptom}: {r.action} {r.param} (score {r.score}). {r.why} "
            f"Risks: {', '.join(r.side_effects) or 'none'}. Source family: {r.evidence}."
        )
    return "\n".join(lines)
