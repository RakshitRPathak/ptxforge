"""Render scripts/schedule_template.mlir for a Config (no C++ rebuild needed per trial)."""
from __future__ import annotations

from pathlib import Path

from .space import Config

TEMPLATE = Path(__file__).resolve().parents[3] / "scripts" / "schedule_template.mlir"


def render(cfg: Config, template: Path | None = None) -> str:
    text = (template or TEMPLATE).read_text()
    for k, v in cfg.to_json().items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        leftover = text[text.index("{{"): text.index("{{") + 12]
        raise ValueError(f"unrendered placeholder near: {leftover!r}")
    return text
