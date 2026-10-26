"""The Python bank model must agree with the C++ header the MLIR pass uses (BankModel.h)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ptxforge_tuner import bank

ROOT = Path(__file__).resolve().parents[3]


class BankKnownAnswers(unittest.TestCase):
    def test_classic_cases(self):
        self.assertEqual(bank.column_access_degree(32 * 4, 4), 32)
        self.assertEqual(bank.column_access_degree(33 * 4, 4), 1)
        self.assertEqual(bank.column_access_degree(36 * 4, 4), 4)
        self.assertEqual(bank.column_access_degree(36 * 4, 16), 1)   # 16B access: quarter-warp phases
        self.assertEqual(bank.column_access_degree(32 * 4, 16), 8)

    def test_broadcast_is_free(self):
        self.assertEqual(bank.conflict_degree([128] * 32, 4), 1)

    def test_choose_pad(self):
        self.assertEqual(bank.choose_pad(32, 4, 4), (1, 1))
        self.assertEqual(bank.choose_pad(32, 4, 16, align_bytes=16), (4, 1))

    def test_kernel_store_pad4_conflict_free(self):
        self.assertEqual(bank.transposed_store_degree(8, 64, 4), 1)
        self.assertGreater(bank.transposed_store_degree(8, 64, 0), 1)


@unittest.skipUnless(shutil.which("g++"), "g++ not available")
class CppCrossCheck(unittest.TestCase):
    def test_python_matches_cpp(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "bank_sim"
            subprocess.run(["g++", "-std=c++20", "-O1", "-I", str(ROOT / "compiler/include"),
                            str(ROOT / "tools/bank_sim.cpp"), "-o", str(exe)], check=True)
            out = subprocess.run([str(exe), "--dump"], check=True, capture_output=True, text=True).stdout
        n = 0
        for line in out.splitlines():
            f = line.split()
            if f[0] == "col":
                stride, access, deg = map(int, f[1:])
                self.assertEqual(bank.column_access_degree(stride, access), deg, line)
            else:
                bk, bm, pad, _elem, deg = None, None, None, None, None
                bk, bm, pad, deg = map(int, f[1:])
                self.assertEqual(bank.transposed_store_degree(bk, bm, pad, 4), deg, line)
            n += 1
        self.assertGreater(n, 1000)


if __name__ == "__main__":
    unittest.main()
