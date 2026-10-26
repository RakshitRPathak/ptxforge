"""Runners turn a Config into a Measurement.

  SimRunner      : NO GPU. Uses the cost model plus a hidden per-config 'unmodeled' factor and noise so
                   the search/UI/CI can run anywhere. Everything it emits is flagged source="simulated"
                   and the studio shows a banner. NEVER quote simulated numbers as results.
  HarnessRunner  : real. nvcc -> PTX, then runtime/ptxforge_bench (CUDA Driver API) on a GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .cost_model import analyze
from .hw import GPU
from .space import Config, Problem

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Measurement:
    ok: bool
    gflops: float | None = None
    ms: float | None = None
    profile: dict = field(default_factory=dict)
    error: str | None = None
    cublas_gflops: float | None = None


class Runner:
    source = "base"

    def measure(self, cfg: Config, prob: Problem, hw: GPU) -> Measurement:  # pragma: no cover
        raise NotImplementedError

    def stages(self, cfg: Config, prob: Problem, hw: GPU) -> list[dict]:
        return []


def _hash01(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64  # [0,1)


class SimRunner(Runner):
    source = "simulated"

    def __init__(self, seed: int = 0, world: int = 0):
        self.rng = random.Random(seed)
        self.world = world

    def measure(self, cfg, prob, hw):
        an = analyze(cfg, prob, hw)
        if an.blocks_per_sm == 0:
            return Measurement(False, error="kernel cannot launch (0 blocks/SM)")
        hidden = (_hash01(self.world, cfg.short()) - 0.5) * 0.30      # +-15% unmodeled effects
        noise = self.rng.gauss(0, 0.015)
        g = min(an.predicted_gflops * (1 + hidden) * (1 + noise), hw.peak_gflops * hw.sustained)
        regs = min(cfg.regs_est, an.reg_cap)
        profile = {
            "regs": regs,
            "local_bytes": an.spill_regs * 4 * 4 if an.spill_regs else 0,
            "static_smem": cfg.smem_bytes,
            "blocks_per_sm": an.blocks_per_sm,
            "occupancy": round(an.occupancy, 4),
            "max_err": 0.0,
        }
        return Measurement(True, gflops=g, ms=prob.flops / (g * 1e6), profile=profile)

    def stages(self, cfg, prob, hw):
        src = (REPO / "kernels" / "gemm_bias_gelu.cu").read_text()
        note = "(simulated run: no compiler or GPU was used, so there is no real IR here)"
        return [
            {"name": "CUDA C++ source", "lang": "cuda", "text": f"// build flags: {' '.join(cfg.defines())}\n{src}"},
            {"name": "LLVM IR", "lang": "llvm", "text": note},
            {"name": "PTX", "lang": "ptx", "text": note},
        ]


_PTXAS_REGS = re.compile(r"Used (\d+) registers(?:, (\d+) bytes smem)?")
_PTXAS_SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")


def parse_ptxas(text: str) -> dict:
    """Parse `ptxas -v` output: works without a GPU (the no-GPU path from the original plan)."""
    out: dict = {}
    m = _PTXAS_REGS.search(text)
    if m:
        out["regs"] = int(m.group(1))
        if m.group(2):
            out["static_smem"] = int(m.group(2))
    s = _PTXAS_SPILL.search(text)
    if s:
        out["spill_store_bytes"], out["spill_load_bytes"] = int(s.group(1)), int(s.group(2))
    return out


class HarnessRunner(Runner):
    source = "harness"

    def __init__(self, bench: str = "build/ptxforge-bench", nvcc: str = "nvcc",
                 workdir: str = "build/kernels", cublas: bool = True, iters: int = 20,
                 timeout: int = 180):
        self.bench, self.nvcc = Path(bench), nvcc
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cublas, self.iters, self.timeout = cublas, iters, timeout
        self._cublas_cache: float | None = None

    def _ptx_path(self, cfg: Config) -> Path:
        return self.workdir / f"k_{cfg.BM}_{cfg.BN}_{cfg.BK}_{cfg.TM}_{cfg.TN}_{cfg.PAD}.ptx"

    def compile(self, cfg: Config, hw: GPU) -> tuple[Path | None, str]:
        ptx = self._ptx_path(cfg)
        if ptx.exists():
            return ptx, ""
        cmd = [self.nvcc, "-ptx", f"-arch={hw.arch}", "-O3", "-lineinfo", *cfg.defines(),
               str(REPO / "kernels" / "gemm_bias_gelu.cu"), "-o", str(ptx)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            return None, r.stderr[-800:]
        return ptx, r.stderr

    def measure(self, cfg, prob, hw):
        ptx, err = self.compile(cfg, hw)
        if ptx is None:
            return Measurement(False, error=f"nvcc failed: {err}")
        gx, gy = cfg.grid(prob)
        cmd = [str(self.bench), "--ptx", str(ptx), "--M", str(prob.M), "--N", str(prob.N),
               "--K", str(prob.K), "--grid", f"{gx},{gy}", "--block", str(cfg.threads),
               "--iters", str(self.iters)]
        if self.cublas and self._cublas_cache is None:
            cmd.append("--cublas")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        line = (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1]
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            return Measurement(False, error=f"bench output unparsable: {line[:200]}")
        if not j.get("ok"):
            return Measurement(False, error=j.get("error") or f"incorrect result (max_err={j.get('max_err')})",
                               profile={k: j[k] for k in ("regs", "max_err") if k in j})
        cb = j.get("cublas_gflops", -1)
        if cb and cb > 0:
            self._cublas_cache = cb
        profile = {k: j[k] for k in ("regs", "local_bytes", "static_smem", "blocks_per_sm",
                                     "occupancy", "max_err") if k in j}
        return Measurement(True, gflops=j["gflops"], ms=j["ms"], profile=profile,
                           cublas_gflops=self._cublas_cache)

    def stages(self, cfg, prob, hw):
        src = (REPO / "kernels" / "gemm_bias_gelu.cu").read_text()
        ptx = self._ptx_path(cfg)
        return [
            {"name": "CUDA C++ source", "lang": "cuda", "text": f"// build flags: {' '.join(cfg.defines())}\n{src}"},
            {"name": "LLVM IR", "lang": "llvm", "text": "// nvcc backend does not expose LLVM IR; the MLIR backend (milestone M1) will."},
            {"name": "PTX", "lang": "ptx", "text": ptx.read_text()[:60000] if ptx.exists() else "// missing"},
        ]
