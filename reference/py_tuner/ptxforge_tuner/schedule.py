"""Render scripts/schedule_template.mlir for a Config (no C++ rebuild needed per trial)."""
from __future__ import annotations

from pathlib import Path

from .space import Config

TEMPLATE = Path(__file__).resolve().parents[3] / "scripts" / "schedule_template.mlir"


def copy_piece(rows: int, cols: int, threads: int) -> tuple[int, int]:
    """Piece of a rows x cols tile each thread copies to shared memory per K step (same rule as main.cpp)."""
    if cols % 4:
        raise ValueError("shared-memory tile width must be a multiple of 4 floats")
    per = max(1, -(-(rows * cols // 4) // threads))
    r = min(per, rows)
    c = 4 * -(-per // r)
    if rows % r or cols % c:
        raise ValueError(f"cannot split a {rows}x{cols} tile evenly over {threads} threads")
    return r, c


def render(cfg: Config, template: Path | None = None) -> str:
    text = (template or TEMPLATE).read_text()
    values = dict(cfg.to_json())
    # Thread-block shape for map_nested_forall_to_threads (same as `ptxforge-tune schedule`).
    values.update(TX=cfg.BN // cfg.TN, TY=cfg.BM // cfg.TM)
    threads = (cfg.BN // cfg.TN) * (cfg.BM // cfg.TM)
    (values["CA_R"], values["CA_C"]) = copy_piece(cfg.BM, cfg.BK, threads)
    (values["CB_R"], values["CB_C"]) = copy_piece(cfg.BK, cfg.BN, threads)
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        leftover = text[text.index("{{"): text.index("{{") + 12]
        raise ValueError(f"unrendered placeholder near: {leftover!r}")
    return text
