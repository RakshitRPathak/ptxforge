"""GPU specs used by the cost model and the roofline.

Numbers are public datasheet / CUDA-occupancy-calculator values, rounded. CONFIRM them for your
card (nvidia-smi, deviceQuery) before quoting any "% of peak" figure. `sustained` is the fraction
of datasheet FP32 peak a tuned kernel can realistically hold (clock/power throttling): it is a
prior, not a measurement; the T4 in particular rarely holds its boost clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    arch: str                 # nvcc -arch value
    sms: int
    peak_gflops: float        # FP32 FMA datasheet peak
    bw_gbs: float             # DRAM bandwidth
    regs_per_sm: int
    max_threads_sm: int
    max_blocks_sm: int
    smem_per_sm: int          # bytes
    smem_per_block: int       # bytes usable without opt-in
    sustained: float = 0.8

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (FLOP/byte) where the roofline turns compute-bound."""
        return self.peak_gflops / self.bw_gbs

    def to_json(self) -> dict:
        d = asdict(self)
        d["ridge"] = round(self.ridge, 3)
        return d


GPUS: dict[str, GPU] = {
    "t4": GPU("NVIDIA T4", "sm_75", 40, 8141, 320, 65536, 1024, 16, 65536, 49152, sustained=0.62),
    "p100": GPU("NVIDIA P100", "sm_60", 56, 9300, 732, 65536, 2048, 32, 65536, 49152, sustained=0.85),
    "l4": GPU("NVIDIA L4", "sm_89", 58, 30300, 300, 65536, 1536, 24, 102400, 49152, sustained=0.7),
    "a100": GPU("NVIDIA A100 40GB", "sm_80", 108, 19500, 1555, 65536, 2048, 32, 167936, 49152, sustained=0.85),
}


def get_gpu(key: str) -> GPU:
    try:
        return GPUS[key.lower()]
    except KeyError:
        raise SystemExit(f"unknown gpu '{key}'. choices: {', '.join(GPUS)}")
