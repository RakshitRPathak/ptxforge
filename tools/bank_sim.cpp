// bank_sim — standalone, dependency-free check of the bank-conflict model.
//   g++ -std=c++20 -O2 -I compiler/include tools/bank_sim.cpp -o bank_sim && ./bank_sim
// Exit code != 0 if any known-answer test fails (used by CI).
#include "PtxForge/BankModel.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ptxforge;

static int failures = 0;
#define EXPECT_EQ(a, b)                                                                  \
  do {                                                                                   \
    long long _a = (a), _b = (b);                                                        \
    if (_a != _b) {                                                                      \
      std::printf("FAIL %s:%d  %s == %lld, expected %lld\n", __FILE__, __LINE__, #a, _a, \
                  _b);                                                                   \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

// `bank_sim --dump` prints a table that reference/py_tuner/tests/test_bank_crosscheck.py and core/tests/test_core.cpp compares against
// the Python port, so the two implementations cannot silently drift apart.
static int dump() {
  for (int access : {2, 4, 8, 16})
    for (int stride = 4; stride <= 1024; stride += 4)
      std::printf("col %d %d %d\n", stride, access, columnAccessDegree(stride, access));
  for (int bk : {4, 8, 16, 32})
    for (int bm : {32, 64, 128})
      for (int pad : {0, 1, 2, 4, 8})
        std::printf("store %d %d %d %d\n", bk, bm, pad, transposedStoreDegree(bk, bm, pad, 4));
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--dump")
    return dump();
  // --- known answers: 4-byte scalar column access --------------------------------
  EXPECT_EQ(columnAccessDegree(32 * 4, 4), 32); // stride 32 words: everyone in bank 0
  EXPECT_EQ(columnAccessDegree(33 * 4, 4), 1);  // classic +1 padding
  EXPECT_EQ(columnAccessDegree(34 * 4, 4), 2);  // gcd(34,32)=2
  EXPECT_EQ(columnAccessDegree(36 * 4, 4), 4);  // gcd(36,32)=4
  EXPECT_EQ(columnAccessDegree(1 * 4, 4), 1);   // unit stride

  // --- broadcast: all threads read the same word ---------------------------------
  EXPECT_EQ(conflictDegree(std::vector<int64_t>(32, 128), 4), 1);

  // --- fp16 pairs share a word: stride of 2 halves (=1 word) is conflict-free ---
  EXPECT_EQ(columnAccessDegree(2 * 2, 2), 1);

  // --- 16-byte vector access: quarter-warp phases ---------------------------------
  EXPECT_EQ(columnAccessDegree(32 * 4, 16), 8);  // 8 threads/phase, all start at bank 0
  EXPECT_EQ(columnAccessDegree(36 * 4, 16), 1);  // 36 words = 16B-aligned AND conflict-free

  // --- padding chooser ---------------------------------------------------------------
  auto p1 = choosePad(32, 4, 4);                  // scalar fp32, 32-wide rows
  EXPECT_EQ(p1.pad, 1);
  EXPECT_EQ(p1.degree, 1);
  auto p2 = choosePad(32, 4, 16, /*align=*/16);   // float4 loads need 16B-aligned rows
  EXPECT_EQ(p2.pad, 4);
  EXPECT_EQ(p2.degree, 1);
  auto p3 = choosePad(64, 2, 2);                  // fp16 scalar, 64-wide rows
  EXPECT_EQ(p3.degree, 1);

  // --- kernel store pattern (BK=8, BM=64) ----------------------------------------
  int d0 = transposedStoreDegree(8, 64, 0, 4);
  int d1 = transposedStoreDegree(8, 64, 1, 4);
  std::printf("As[c][r] store, BK=8 BM=64: pad0 -> %d-way, pad1 -> %d-way\n", d0, d1);
  EXPECT_EQ(d0 > d1, 1); // padding must not make things worse
  EXPECT_EQ(transposedStoreDegree(8, 64, 4, 4), 1); // pad 4 makes the kernel's store conflict-free

  // --- human-readable table -------------------------------------------------------
  std::printf("\nrowElems=32 fp32, scalar access (column pattern)\n  pad  degree\n");
  for (int pad : {0, 1, 2, 3, 4, 8})
    std::printf("  %3d  %2d-way\n", pad, columnAccessDegree((32 + pad) * 4, 4));

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all bank-model checks passed",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
