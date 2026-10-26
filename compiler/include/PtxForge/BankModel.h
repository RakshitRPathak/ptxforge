// BankModel.h — header-only shared-memory bank-conflict model.
//
// No MLIR/LLVM dependency on purpose: the same logic is used by
//   (a) the ptxforge-pad-workgroup-memory MLIR pass,
//   (b) tools/bank_sim (standalone, unit-tested, runs anywhere),
//   (c) core/include/ptxforge/cost_model.hpp (the tuner, which includes this header directly); reference/py_tuner/ptxforge_tuner/bank.py is the Python mirror used for parity tests.
//
// Model: 32 banks x 4 bytes. A warp issues one shared-memory request.
// Accesses wider than 4 B are split into phases of (128 B / accessBytes) threads
// (8 threads for 16 B, 16 threads for 8 B, 32 threads for <=4 B).
// Within a phase, threads touching the *same 4-byte word* broadcast (no conflict);
// threads touching *different words in the same bank* serialize.
// conflictDegree == max number of distinct words that land in one bank in a phase.
// 1 means conflict-free.
#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

namespace ptxforge {

inline constexpr int kNumBanks = 32;
inline constexpr int kBankBytes = 4;
inline constexpr int kWarpSize = 32;

// Conflict degree for an arbitrary list of per-thread byte addresses (size <= 32).
inline int conflictDegree(const std::vector<int64_t> &byteAddrs, int accessBytes) {
  const int bytes = std::max(accessBytes, 1);
  const int phaseThreads =
      bytes <= kBankBytes ? kWarpSize : std::max(1, (kNumBanks * kBankBytes) / bytes);
  int worst = 1;
  for (size_t p = 0; p < byteAddrs.size(); p += phaseThreads) {
    std::vector<std::set<int64_t>> wordsPerBank(kNumBanks);
    size_t end = std::min(byteAddrs.size(), p + phaseThreads);
    for (size_t t = p; t < end; ++t) {
      int64_t first = byteAddrs[t] / kBankBytes;
      int64_t last = (byteAddrs[t] + bytes - 1) / kBankBytes;
      for (int64_t w = first; w <= last; ++w)
        wordsPerBank[w % kNumBanks].insert(w);
    }
    for (auto &s : wordsPerBank)
      worst = std::max<int>(worst, static_cast<int>(s.size()));
  }
  return worst;
}

// Column access: thread t reads row t, same column  -> address = t * rowStrideBytes.
// (e.g. every thread of a warp reading element k of its own row of a tile).
inline int columnAccessDegree(int64_t rowStrideBytes, int accessBytes) {
  std::vector<int64_t> a(kWarpSize);
  for (int t = 0; t < kWarpSize; ++t)
    a[t] = t * rowStrideBytes;
  return conflictDegree(a, accessBytes);
}

// Cooperative-store pattern used by the kernel in kernels/gemm_bias_gelu.cu:
// linear index i = lane -> (c = i % bk, r = i / bk), stored transposed at As[c][r]
// with row length (bm + pad) elements.
inline int transposedStoreDegree(int bk, int bm, int pad, int elemBytes) {
  std::vector<int64_t> a(kWarpSize);
  for (int lane = 0; lane < kWarpSize; ++lane) {
    int c = lane % bk, r = lane / bk;
    a[lane] = (static_cast<int64_t>(c) * (bm + pad) + r) * elemBytes;
  }
  return conflictDegree(a, elemBytes);
}

struct PadChoice {
  int pad = 0;      // extra elements appended to each row
  int degree = 1;   // resulting conflict degree for the column pattern
};

// Smallest padding (in elements) that minimizes the column-access conflict degree.
// alignBytes > 0 forces the padded row stride to be a multiple of alignBytes
// (needed to keep 16-byte vector loads legal).
inline PadChoice choosePad(int rowElems, int elemBytes, int accessBytes,
                           int alignBytes = 0, int maxPad = 32) {
  PadChoice best{0, 1 << 30};
  for (int pad = 0; pad <= maxPad; ++pad) {
    int64_t strideBytes = static_cast<int64_t>(rowElems + pad) * elemBytes;
    if (alignBytes > 0 && strideBytes % alignBytes != 0)
      continue;
    int d = columnAccessDegree(strideBytes, accessBytes);
    if (d < best.degree) {
      best = {pad, d};
      if (d == 1)
        break;
    }
  }
  if (best.degree == (1 << 30))
    best = {0, columnAccessDegree(static_cast<int64_t>(rowElems) * elemBytes, accessBytes)};
  return best;
}

} // namespace ptxforge
