"""Analytic cost model: roofline + occupancy + register-cap + bank-conflict + tail effects.

This is a PRIOR used to rank/prune configs and to place points on the roofline. It is not ground
truth: every number it produces is labelled `modeled` in the trace, and the search always
re-measures the configs it cares about. Calibrating it against measurements (and reporting the
rank correlation) is milestone M5 and a good thing to put in the paper/README.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .bank import kernel_bank_degrees
from .hw import GPU
from .space import Config, Problem


@dataclass
class Analysis:
    flops: float
    traffic_bytes: float       # bytes moved global->SM under the tiling (no L2 reuse credit)
    ideal_bytes: float         # compulsory traffic: each matrix touched exactly once
    ai: float                  # FLOP / traffic_bytes  (x-axis of the roofline)
    ai_ideal: float
    roof_gflops: float         # min(sustained peak, ai * bw)
    memory_bound: bool
    regs_est: int
    reg_cap: int               # registers/thread the compiler may use under __launch_bounds__
    spill_regs: int
    blocks_per_sm: int
    occupancy: float
    bank: dict
    bank_degree: int           # worst of the shared-memory access patterns
    grid_blocks: int
    waves: float
    f_occ: float
    f_ilp: float
    f_bank: float
    f_spill: float
    f_fill: float
    predicted_gflops: float

    def to_json(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def analyze(c: Config, p: Problem, hw: GPU) -> Analysis:
    flops = p.flops
    traffic = 4.0 * (p.M * p.K * (p.N / c.BN) + p.K * p.N * (p.M / c.BM) + p.M * p.N)
    ideal = 4.0 * (p.M * p.K + p.K * p.N + p.M * p.N)
    ai = flops / traffic
    ai_ideal = flops / ideal
    peak = hw.peak_gflops * hw.sustained
    roof = min(peak, ai * hw.bw_gbs)

    threads = c.threads
    reg_cap = min(255, (hw.regs_per_sm // threads) // 8 * 8)
    spill = max(0, c.regs_est - reg_cap)
    regs = min(c.regs_est, reg_cap)
    regs_alloc = _round_up(max(regs, 16), 8)
    regs_block = _round_up(regs_alloc * threads, 256)
    by_regs = hw.regs_per_sm // regs_block
    by_threads = hw.max_threads_sm // threads
    by_smem = hw.smem_per_sm // max(c.smem_bytes, 1)
    bpsm = max(0, min(by_regs, by_threads, by_smem, hw.max_blocks_sm))
    occ = bpsm * threads / hw.max_threads_sm

    bank = kernel_bank_degrees(c.BM, c.BN, c.BK, c.TM, c.TN, c.PAD)
    extra = sum(d - 1 for d in bank.values())

    grid = (p.M // c.BM) * (p.N // c.BN)
    waves = grid / max(1, hw.sms * max(bpsm, 1))
    f_fill = max(0.3, waves / math.ceil(waves)) if waves > 0 else 0.3

    f_occ = min(1.0, occ / 0.5) ** 0.5 if bpsm > 0 else 0.0
    f_ilp = (c.TM * c.TN) / (c.TM * c.TN + 0.5 * (c.TM + c.TN))
    f_bank = 1.0 / (1.0 + 0.1 * extra)
    f_spill = math.exp(-spill / 32.0)

    pred = roof * f_occ * f_ilp * f_bank * f_spill * f_fill
    return Analysis(
        flops=flops, traffic_bytes=traffic, ideal_bytes=ideal, ai=ai, ai_ideal=ai_ideal,
        roof_gflops=roof, memory_bound=ai * hw.bw_gbs < peak, regs_est=c.regs_est,
        reg_cap=reg_cap, spill_regs=spill, blocks_per_sm=bpsm, occupancy=occ, bank=bank,
        bank_degree=max(bank.values()), grid_blocks=grid, waves=waves, f_occ=f_occ,
        f_ilp=f_ilp, f_bank=f_bank, f_spill=f_spill, f_fill=f_fill, predicted_gflops=pred,
    )
