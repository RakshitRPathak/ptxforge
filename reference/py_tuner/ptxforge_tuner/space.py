"""Search space for the parametric GEMM+bias+GELU kernel (and the MLIR schedule template)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .hw import GPU

DOMAINS: dict[str, tuple[int, ...]] = {
    "BM": (32, 64, 128, 256),
    "BN": (32, 64, 128, 256),
    "BK": (4, 8, 16, 32),
    "TM": (2, 4, 8, 16),
    "TN": (2, 4, 8, 16),
    "PAD": (0, 1, 2, 4, 8),
}
PARAMS = tuple(DOMAINS)

DEFAULT = None  # set below (the "untuned" starting point reported as the baseline)


@dataclass(frozen=True)
class Problem:
    M: int = 2048
    N: int = 2048
    K: int = 2048

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K  # epilogue FLOPs are negligible (O(MN))


@dataclass(frozen=True, order=True)
class Config:
    BM: int
    BN: int
    BK: int
    TM: int
    TN: int
    PAD: int

    @property
    def threads(self) -> int:
        return (self.BM // self.TM) * (self.BN // self.TN)

    @property
    def smem_bytes(self) -> int:
        return 4 * (self.BK * (self.BM + self.PAD) + self.BK * self.BN)

    @property
    def regs_est(self) -> int:
        """Rough registers/thread: accumulators + A/B fragments + addressing/loop overhead."""
        return self.TM * self.TN + self.TM + self.TN + 26

    def grid(self, p: Problem) -> tuple[int, int]:
        return (p.N // self.BN, p.M // self.BM)

    def to_json(self) -> dict:
        d = asdict(self)
        d["threads"] = self.threads
        d["smem_bytes"] = self.smem_bytes
        return d

    def short(self) -> str:
        return f"{self.BM}x{self.BN}x{self.BK} t{self.TM}x{self.TN} p{self.PAD}"

    def defines(self) -> list[str]:
        return [f"-D{k}={getattr(self, k)}" for k in PARAMS]


DEFAULT = Config(BM=64, BN=64, BK=8, TM=4, TN=4, PAD=0)


def validity_error(c: Config, p: Problem, hw: GPU) -> str | None:
    if p.M % c.BM or p.N % c.BN or p.K % c.BK:
        return "problem size not divisible by tile (kernel has no bounds checks)"
    if c.BM % c.TM or c.BN % c.TN:
        return "tile not divisible by micro-tile"
    t = c.threads
    if t < 64 or t > 1024 or t % 32:
        return f"threads/block {t} outside [64,1024] or not a multiple of 32"
    if c.smem_bytes > hw.smem_per_block:
        return f"static shared memory {c.smem_bytes} B exceeds {hw.smem_per_block} B"
    return None


def is_valid(c: Config, p: Problem, hw: GPU) -> bool:
    return validity_error(c, p, hw) is None


def enumerate_space(p: Problem, hw: GPU) -> list[Config]:
    out: list[Config] = []
    for bm in DOMAINS["BM"]:
        for bn in DOMAINS["BN"]:
            for bk in DOMAINS["BK"]:
                for tm in DOMAINS["TM"]:
                    for tn in DOMAINS["TN"]:
                        for pad in DOMAINS["PAD"]:
                            c = Config(bm, bn, bk, tm, tn, pad)
                            if is_valid(c, p, hw):
                                out.append(c)
    return out


def step(c: Config, param: str, direction: int) -> Config | None:
    """Move one notch along a parameter's domain (direction = +1 / -1); None at the edge."""
    dom = DOMAINS[param]
    i = dom.index(getattr(c, param)) + direction
    if i < 0 or i >= len(dom):
        return None
    return Config(**{**asdict(c), param: dom[i]})


def neighbors(c: Config, p: Problem, hw: GPU) -> list[Config]:
    out = []
    for param in PARAMS:
        for d in (-1, +1):
            n = step(c, param, d)
            if n is not None and is_valid(n, p, hw):
                out.append(n)
    return out
