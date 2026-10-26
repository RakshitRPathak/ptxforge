import json
import random
import tempfile
import unittest
from pathlib import Path

from ptxforge_tuner import kg as kgmod
from ptxforge_tuner.advisors import Context, LLMAdvisor, make_advisor
from ptxforge_tuner.cost_model import analyze
from ptxforge_tuner.hw import get_gpu
from ptxforge_tuner.runners import SimRunner, parse_ptxas
from ptxforge_tuner.schedule import render
from ptxforge_tuner.search import spearman, tune
from ptxforge_tuner.space import DEFAULT, Config, Problem

P = Problem(2048, 2048, 2048)
T4 = get_gpu("t4")


def ctx_for(cfg):
    an = analyze(cfg, P, T4)
    return Context(P, T4, cfg, an, {}, kgmod.diagnose(cfg, an, {}, T4), [], set(), random.Random(0))


class Tune(unittest.TestCase):
    def test_sim_trace_is_flagged_and_improves(self):
        tr = tune(P, T4, SimRunner(seed=1), make_advisor("kg"), budget=24, seed=1)
        self.assertEqual(tr["source"], "simulated")
        self.assertIn("SIMULATED", tr["warning"])
        self.assertEqual(tr["trials"][0]["origin"], "baseline")
        self.assertGreaterEqual(tr["best_gflops"], tr["baseline_gflops"])
        self.assertEqual(len(tr["trials"]), 24)
        self.assertEqual(len({json.dumps(t["cfg"], sort_keys=True) for t in tr["trials"]}), 24)  # no repeats
        self.assertIsNone(tr["cublas_gflops"])  # never invent a cuBLAS number
        json.dumps(tr)

    def test_all_advisors_run(self):
        for name in ("none", "random", "rules", "kg"):
            tr = tune(P, T4, SimRunner(seed=2), make_advisor(name), budget=12, seed=2)
            self.assertEqual(len(tr["trials"]), 12, name)

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


class KG(unittest.TestCase):
    def test_spill_retrieves_register_remedies_first(self):
        kg = kgmod.load_kg()
        rem = kgmod.retrieve(["register_spill"], kg)
        self.assertEqual(rem[0].symptom, "register_spill")
        self.assertIn(rem[0].param, ("TM", "TN"))
        self.assertEqual(rem[0].action, "decrease")

    def test_all_remedy_symptoms_exist(self):
        kg = kgmod.load_kg()
        for r in kg["remedies"]:
            self.assertIn(r["symptom"], kg["symptoms"])
            for se in r["side_effects"]:
                self.assertIn(se, kg["symptoms"])

    def test_kg_advisor_fixes_bank_conflict(self):
        c = Config(64, 64, 8, 4, 4, 0)
        props = make_advisor("kg").propose(ctx_for(c))
        self.assertTrue(any(p.cfg.PAD != 0 for p in props))

    def test_kg_advisor_reacts_to_spill(self):
        c = Config(128, 128, 8, 16, 16, 0)
        props = make_advisor("kg").propose(ctx_for(c))
        self.assertTrue(props)
        self.assertTrue(any(p.cfg.TM < 16 or p.cfg.TN < 16 for p in props))


class LLMGuard(unittest.TestCase):
    def test_rejects_invalid_and_duplicate_suggestions(self):
        adv = LLMAdvisor.__new__(LLMAdvisor)
        adv.rejected = adv.accepted = 0
        ctx = ctx_for(DEFAULT)
        text = ('here you go {"proposals":['
                '{"BM":128,"BN":128,"BK":8,"TM":8,"TN":8,"PAD":4,"why":"ok"},'
                '{"BM":512,"BN":64,"BK":8,"TM":4,"TN":4,"PAD":0,"why":"not even in the domain"},'
                '{"BM":256,"BN":256,"BK":32,"TM":8,"TN":8,"PAD":0,"why":"too much smem"},'
                '{"BM":64}]}')
        out = adv.parse(text, ctx)
        self.assertEqual(len(out), 1)
        # only the first 3 suggestions are considered: 1 accepted; out-of-domain/over-limit and malformed rejected
        self.assertEqual(adv.accepted, 1)
        self.assertEqual(adv.rejected, 2)
        too_much_smem = '{"proposals":[{"BM":256,"BN":256,"BK":32,"TM":8,"TN":8,"PAD":0}]}'
        self.assertEqual(adv.parse(too_much_smem, ctx), [])
        self.assertEqual(adv.rejected, 3)
        self.assertEqual(adv.parse("no json at all", ctx), [])


class Misc(unittest.TestCase):
    def test_ptxas_parse(self):
        txt = ("ptxas info    : Function properties for gemm_bias_gelu\n"
               "    8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\n"
               "ptxas info    : Used 64 registers, 4352 bytes smem, 388 bytes cmem[0]\n")
        self.assertEqual(parse_ptxas(txt), {"regs": 64, "static_smem": 4352,
                                            "spill_store_bytes": 4, "spill_load_bytes": 4})

    def test_schedule_renders_without_placeholders(self):
        text = render(Config(128, 64, 16, 8, 4, 4))
        self.assertIn("tile_sizes [128, 64]", text)
        self.assertIn("tile_sizes [0, 0, 16]", text)
        self.assertNotIn("{{", text)


if __name__ == "__main__":
    unittest.main()
