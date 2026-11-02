#!/usr/bin/env python3
"""Pull the PTX text out of a `gpu.binary` op produced by `gpu-module-to-binary{format=isa}`.

    python3 scripts/extract_ptx.py lowered.mlir out.ptx

MLIR prints the PTX as a string attribute with \\XX hex escapes (\\0A for newline, \\22 for a quote).
"""
import re
import sys


def mlir_string_literals(text):
    """Yield every double-quoted MLIR string literal in `text`, unescaped."""
    i = 0
    while True:
        i = text.find('"', i)
        if i < 0:
            return
        out, j = bytearray(), i + 1
        while text[j] != '"':
            c = text[j]
            if c == "\\":
                nxt = text[j + 1]
                if nxt in '"\\':
                    out += nxt.encode()
                    j += 2
                elif nxt == "n":
                    out += b"\n"
                    j += 2
                elif nxt == "t":
                    out += b"\t"
                    j += 2
                else:
                    out.append(int(text[j + 1:j + 3], 16))
                    j += 3
            else:
                out += c.encode()
                j += 1
        yield out.decode("utf-8", errors="replace")
        i = j + 1


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()
    start = text.find("gpu.binary")
    if start < 0:
        sys.exit(f"{src}: no gpu.binary op (run gpu-module-to-binary with format=isa first)")
    ptx = [s for s in mlir_string_literals(text[start:]) if ".entry" in s and ".target" in s]
    if not ptx:
        sys.exit(f"{src}: gpu.binary has no PTX text (was the format isa?)")
    open(dst, "w", encoding="utf-8").write(ptx[0])
    entries = re.findall(r"\.entry\s+(\w+)", ptx[0])
    print(f"wrote {dst}: {len(ptx[0].splitlines())} lines, kernels: {', '.join(entries)}")


if __name__ == "__main__":
    main()
