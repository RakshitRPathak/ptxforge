import unittest

from ptxforge_tuner.cost_model import analyze
from ptxforge_tuner.hw import GPUS, get_gpu
from ptxforge_tuner.space import DEFAULT, Config, Problem, enumerate_space, is_valid, neighbors, validity_error

P = Problem(2048, 2048, 2048)
T4 = get_gpu("t4")


class Space(unittest.TestCase):
    def test_default_valid_and_space_nonempty(self):
        self.assertTrue(is_valid(DEFAULT, P, T4))
        self.assertGreater(len(enumerate_space(P, T4)), 200)

    def test_invalid_examples(self):
        self.assertIn("shared", validity_error(Config(256, 256, 32, 8, 8, 0), P, T4))
        self.assertIn("threads", validity_error(Config(32, 32, 8, 8, 8, 0), P, T4))
        self.assertIn("divisible", validity_error(Config(64, 64, 8, 4, 4, 0), Problem(2000, 2048, 2048), T4))

    def test_neighbors_valid(self):
        for n in neighbors(DEFAULT, P, T4):
            self.assertTrue(is_valid(n, P, T4))


class Cost(unittest.TestCase):
    def test_all_predictions_finite_positive(self):
        for c in enumerate_space(P, T4):
            a = analyze(c, P, T4)
            self.assertGreater(a.predicted_gflops, 0, c)
            self.assertLessEqual(a.predicted_gflops, T4.peak_gflops, c)

    def test_bigger_tiles_raise_arithmetic_intensity(self):
        a = analyze(Config(32, 32, 8, 4, 4, 0), P, T4)
        b = analyze(Config(128, 128, 8, 8, 8, 0), P, T4)
        self.assertGreater(b.ai, a.ai)

    def test_roofline_ridge_flips_memory_bound(self):
        small = analyze(Config(32, 32, 8, 4, 4, 0), P, T4)
        big = analyze(Config(128, 128, 8, 8, 8, 0), P, T4)
        self.assertTrue(small.memory_bound)
        self.assertFalse(big.memory_bound)

    def test_spill_when_microtile_too_big(self):
        a = analyze(Config(128, 128, 8, 16, 16, 0), P, T4)
        self.assertGreater(a.spill_regs, 0)
        self.assertLess(a.f_spill, 0.5)

    def test_padding_removes_modeled_conflict(self):
        self.assertGreater(analyze(Config(64, 64, 8, 4, 4, 0), P, T4).bank["store_As"], 1)
        self.assertEqual(analyze(Config(64, 64, 8, 4, 4, 4), P, T4).bank["store_As"], 1)

    def test_gpu_table_sane(self):
        for g in GPUS.values():
            self.assertGreater(g.ridge, 1)


if __name__ == "__main__":
    unittest.main()
