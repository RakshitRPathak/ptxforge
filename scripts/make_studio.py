#!/usr/bin/env python3
"""Embed a trace into studio/index.html -> one self-contained HTML file (double-click to open).

    python scripts/make_studio.py runs/trace.json studio/report.html
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = '<script id="embedded" type="application/json">null</script>'


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 64
    trace = json.loads(Path(sys.argv[1]).read_text())
    html = (ROOT / "studio" / "index.html").read_text()
    if MARK not in html:
        raise SystemExit("placeholder not found in studio/index.html")
    blob = json.dumps(trace).replace("</", "<\\/")  # keep </script> out of the payload
    Path(sys.argv[2]).write_text(html.replace(MARK, f'<script id="embedded" type="application/json">{blob}</script>'))
    print(f"wrote {sys.argv[2]}  (source={trace.get('source')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
