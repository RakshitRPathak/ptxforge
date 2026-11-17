"""Analytic cost model: roofline + occupancy + register-cap + issue rate + load latency + tail effects.

This is a PRIOR used to rank/prune configs and to place points on the roofline. It is not ground
truth: every number it produces is labelled `modeled` in the trace, and the search always
re-measures the configs it cares about. Calibrating it against measurements (and reporting the
rank correlation) is milestone M5 and a good thing to put in the paper/README.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .bank import WARP, conflict_degree, kernel_bank_degrees, transposed_store_degree
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
    f_pipe: float
    f_lat: float
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


# Same model as core/include/ptxforge/cost_model.hpp (tests/parity_check.py compares them on every config).
LOAD_LATENCY_CYCLES = 600.0
_LANES_128 = {"sm_86", "sm_87", "sm_89", "sm_90", "sm_100", "sm_120"}


def fp32_lanes_per_sm(arch: str) -> int:
    return 128 if arch in _LANES_128 else 64


def vector_width(count: int, row: int) -> int:
    """Widest vector (4, 2 or 1 floats) ptxas can use for `count` consecutive floats in rows of `row`."""
    for w in (4, 2):
        if count % w == 0 and row % w == 0:
            return w
    return 1


def warp_load_cycles(byte_addrs: list[int], vec: int) -> int:
    nbytes = 4 * vec
    phases = max(1, WARP * nbytes // 128)
    return phases * conflict_degree(byte_addrs, nbytes)


def smem_pipe_factor(c: Config, lanes: int) -> float:
    """FMA pipe vs shared-memory pipe for one K step of kernels/gemm_bias_gelu.cu, per warp."""
    tx_n, threads = c.BN // c.TN, c.threads
    va, vb = vector_width(c.TM, c.BM + c.PAD), vector_width(c.TN, c.BN)
    cyc_a = sum(warp_load_cycles([((lane // tx_n) * c.TM + i) * 4 for lane in range(WARP)], va)
                for i in range(0, c.TM, va))
    cyc_b = sum(warp_load_cycles([((lane % tx_n) * c.TN + j) * 4 for lane in range(WARP)], vb)
                for j in range(0, c.TN, vb))
    lds = c.BK * (cyc_a + cyc_b)
    stores_a = -(-(c.BM * c.BK) // threads)
    stores_b = -(-(c.BK * c.BN) // threads)
    sts = stores_a * transposed_store_degree(c.BK, c.BM, c.PAD, 4) + stores_b
    fma = c.BK * c.TM * c.TN / (lanes / 32.0)
    return fma / max(fma, lds + sts)


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

    grid = (p.M // c.BM) * (p.N // c.BN)
    waves = grid / max(1, hw.sms * max(bpsm, 1))
    f_fill = max(0.3, waves / math.ceil(waves)) if waves > 0 else 0.3

    f_occ = min(1.0, occ / 0.5) ** 0.5 if bpsm > 0 else 0.0
    lanes = fp32_lanes_per_sm(hw.arch)
    f_pipe = smem_pipe_factor(c, lanes)
    work = threads * c.BK * c.TM * c.TN / lanes
    resident = max(bpsm, 1)
    f_lat = resident * work / (resident * work + LOAD_LATENCY_CYCLES)
    f_spill = math.exp(-spill / 32.0)

    pred = roof * f_occ * f_pipe * f_lat * f_spill * f_fill
    return Analysis(
        flops=flops, traffic_bytes=traffic, ideal_bytes=ideal, ai=ai, ai_ideal=ai_ideal,
        roof_gflops=roof, memory_bound=ai * hw.bw_gbs < peak, regs_est=c.regs_est,
        reg_cap=reg_cap, spill_regs=spill, blocks_per_sm=bpsm, occupancy=occ, bank=bank,
        bank_degree=max(bank.values()), grid_blocks=grid, waves=waves, f_occ=f_occ,
        f_pipe=f_pipe, f_lat=f_lat, f_spill=f_spill, f_fill=f_fill, predicted_gflops=pred,
    )
