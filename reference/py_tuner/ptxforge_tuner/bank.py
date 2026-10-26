"""Python port of compiler/include/PtxForge/BankModel.h.

tests/test_bank_crosscheck.py compiles tools/bank_sim.cpp and compares this module against it
over ~1000 cases, so the C++ pass and the tuner's cost model cannot silently disagree.
"""
from __future__ import annotations

NUM_BANKS = 32
BANK_BYTES = 4
WARP = 32


def conflict_degree(byte_addrs: list[int], access_bytes: int) -> int:
    nbytes = max(access_bytes, 1)
    phase = WARP if nbytes <= BANK_BYTES else max(1, (NUM_BANKS * BANK_BYTES) // nbytes)
    worst = 1
    for p in range(0, len(byte_addrs), phase):
        per_bank: list[set[int]] = [set() for _ in range(NUM_BANKS)]
        for a in byte_addrs[p : p + phase]:
            for w in range(a // BANK_BYTES, (a + nbytes - 1) // BANK_BYTES + 1):
                per_bank[w % NUM_BANKS].add(w)
        worst = max(worst, max(len(s) for s in per_bank))
    return worst


def column_access_degree(row_stride_bytes: int, access_bytes: int) -> int:
    return conflict_degree([t * row_stride_bytes for t in range(WARP)], access_bytes)


def transposed_store_degree(bk: int, bm: int, pad: int, elem_bytes: int = 4) -> int:
    """Cooperative store As[c][r] = A[r][c] with lane -> (c = lane % bk, r = lane // bk)."""
    addrs = [((lane % bk) * (bm + pad) + lane // bk) * elem_bytes for lane in range(WARP)]
    return conflict_degree(addrs, elem_bytes)


def kernel_bank_degrees(bm: int, bn: int, bk: int, tm: int, tn: int, pad: int) -> dict[str, int]:
    """Bank-conflict degree of the three shared-memory access patterns in gemm_bias_gelu.cu."""
    tx_n = bn // tn
    store = transposed_store_degree(bk, bm, pad)
    a_read = conflict_degree([((lane // tx_n) * tm) * 4 for lane in range(WARP)], 4)
    b_read = conflict_degree([((lane % tx_n) * tn) * 4 for lane in range(WARP)], 4)
    return {"store_As": store, "read_As": a_read, "read_Bs": b_read}


def choose_pad(row_elems: int, elem_bytes: int, access_bytes: int, align_bytes: int = 0,
               max_pad: int = 32) -> tuple[int, int]:
    best = (0, 1 << 30)
    for pad in range(max_pad + 1):
        stride = (row_elems + pad) * elem_bytes
        if align_bytes > 0 and stride % align_bytes:
            continue
        d = column_access_degree(stride, access_bytes)
        if d < best[1]:
            best = (pad, d)
            if d == 1:
                break
    if best[1] == 1 << 30:
        best = (0, column_access_degree(row_elems * elem_bytes, access_bytes))
    return best
