#!/usr/bin/env python3
"""Bridge for `ptxforge-tune tune --advisor llm --llm-cmd "python3 scripts/llm_propose.py"`.

Protocol: the C++ tuner writes a context JSON file and runs `<cmd> <file>`. This script prints JSON
    {"proposals":[{"BM":..,"BN":..,"BK":..,"TM":..,"TN":..,"PAD":..,"why":".."}, ...]}
to stdout. The C++ side validates EVERYTHING (domain, divisibility, smem, duplicates), so a
hallucinated config is rejected and counted, never run.

Needs ANTHROPIC_API_KEY and `pip install anthropic`. STATUS: written but not run by the author
(no network/API key in the authoring sandbox); the C++ side of the protocol is tested with
core/tests/fake_advisor.sh. Set PTXFORGE_LLM_MODEL to choose the model.
"""
import json
import os
import sys

PROMPT = """You are a GPU performance engineer tuning a register-blocked SGEMM+bias+GELU kernel.
Parameters and allowed values: BM,BN in {32,64,128,256}; BK in {4,8,16,32}; TM,TN in {2,4,8,16}; PAD in {0,1,2,4,8}.
Given the context JSON (problem, GPU limits, best config so far with its modeled/measured counters,
detected symptoms, and history), propose up to 3 NEW configs likely to beat the best.
Reply with ONLY JSON: {{"proposals":[{{"BM":..,"BN":..,"BK":..,"TM":..,"TN":..,"PAD":..,"why":"one sentence"}}]}}

Context:
{ctx}
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 64
    ctx = open(sys.argv[1], encoding="utf-8").read()
    try:
        import anthropic
    except ImportError:
        print("pip install anthropic", file=sys.stderr)
        return 1
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("PTXFORGE_LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=800,
        messages=[{"role": "user", "content": PROMPT.format(ctx=ctx)}],
    )
    print("".join(b.text for b in msg.content if getattr(b, "type", "") == "text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
