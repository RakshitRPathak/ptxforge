// Unit tests for the C++ core. No framework dependency.
//   g++ -std=c++20 -O1 -Icore/include -Icompiler/include -DPTXFORGE_ROOT_DIR='"."' core/tests/test_core.cpp -o build/ptxforge-tests
#include <cstdio>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "ptxforge/search.hpp"

#ifndef PTXFORGE_ROOT_DIR
#define PTXFORGE_ROOT_DIR "."
#endif

using namespace ptxforge;

namespace {
struct TestCase { const char *name; std::function<void()> fn; };
std::vector<TestCase> &registry() { static std::vector<TestCase> r; return r; }
struct Reg { Reg(const char *n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
int g_failures = 0;
} // namespace

#define TEST(name) static void name(); static Reg reg_##name(#name, name); static void name()
#define CHECK(cond) do { if (!(cond)) { std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)
#define CHECK_EQ(a, b) do { auto _a = (a); auto _b = (b); if (!(_a == _b)) { std::printf("    FAIL %s:%d  %s == %s\n", __FILE__, __LINE__, #a, #b); ++g_failures; } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = (a), _b = (b); if (std::fabs(_a - _b) > (tol)) { std::printf("    FAIL %s:%d  %s ~ %s (%g vs %g)\n", __FILE__, __LINE__, #a, #b, _a, _b); ++g_failures; } } while (0)

static const Problem P{2048, 2048, 2048};
static const GPU &T4() { return get_gpu("t4"); }
static const std::string kRoot = PTXFORGE_ROOT_DIR;

// ---- json ----------------------------------------------------------------------------
TEST(json_roundtrip) {
  auto v = json::Value::parse(R"({"a":[1,2.5,-3e2,true,null,"x\n\"y\" é"],"b":{"c":{}} ,"d":[]})");
  CHECK_EQ(v.at("a").items().size(), size_t(6));
  CHECK_NEAR(v.at("a").items()[2].as_number(), -300.0, 1e-12);
  CHECK(v.at("a").items()[4].is_null());
  CHECK_EQ(v.at("a").items()[5].as_string(), std::string("x\n\"y\" \xC3\xA9"));
  auto again = json::Value::parse(v.dump(2));
  CHECK_EQ(again.dump(), v.dump());
}
TEST(json_rejects_garbage) {
  for (const char *bad : {"", "{", "[1,]x", "{\"a\" 1}", "tru", "\"unterminated", "[1] 2"}) {
    bool threw = false;
    try { json::Value::parse(bad); } catch (const std::exception &) { threw = true; }
    CHECK(threw);
  }
}
TEST(json_numbers_print_cleanly) {
  CHECK_EQ(json::Value(3).dump(), std::string("3"));
  CHECK_EQ(json::Value(2048.0).dump(), std::string("2048"));
  CHECK_EQ(json::Value(0.5).dump(), std::string("0.5"));
  CHECK_EQ(json::Value(std::nan("")).dump(), std::string("null"));
}

// ---- bank model (same expectations as tools/bank_sim.cpp) ------------------------------
TEST(bank_known_answers) {
  CHECK_EQ(columnAccessDegree(32 * 4, 4), 32);
  CHECK_EQ(columnAccessDegree(33 * 4, 4), 1);
  CHECK_EQ(columnAccessDegree(36 * 4, 4), 4);
  CHECK_EQ(columnAccessDegree(36 * 4, 16), 1);
  CHECK_EQ(columnAccessDegree(32 * 4, 16), 8);
  CHECK_EQ(transposedStoreDegree(8, 64, 4, 4), 1);
  CHECK(transposedStoreDegree(8, 64, 0, 4) > 1);
}

// ---- space / cost model ---------------------------------------------------------------
TEST(space_size_and_validity) {
  CHECK_EQ(enumerate_space(P, T4()).size(), size_t(3610));
  CHECK(is_valid(kDefaultConfig, P, T4()));
  CHECK(validity_error(Config{256, 256, 32, 8, 8, 0}, P, T4())->find("shared") != std::string::npos);
  CHECK(validity_error(Config{32, 32, 8, 8, 8, 0}, P, T4())->find("threads") != std::string::npos);
  CHECK(validity_error(kDefaultConfig, Problem{2000, 2048, 2048}, T4())->find("divisible") != std::string::npos);
  CHECK(validity_error(Config{512, 64, 8, 4, 4, 0}, P, T4())->find("domain") != std::string::npos);
  for (const Config &n : neighbors(kDefaultConfig, P, T4())) CHECK(is_valid(n, P, T4()));
}
TEST(cost_model_properties) {
  for (const Config &c : enumerate_space(P, T4())) {
    Analysis a = analyze(c, P, T4());
    CHECK(a.predicted_gflops > 0);
    CHECK(a.predicted_gflops <= T4().peak_gflops);
  }
  CHECK(analyze(Config{128, 128, 8, 8, 8, 0}, P, T4()).ai > analyze(Config{32, 32, 8, 4, 4, 0}, P, T4()).ai);
  CHECK(analyze(Config{32, 32, 8, 4, 4, 0}, P, T4()).memory_bound);
  CHECK(!analyze(Config{128, 128, 8, 8, 8, 0}, P, T4()).memory_bound);
  Analysis spill = analyze(Config{128, 128, 8, 16, 16, 0}, P, T4());
  CHECK(spill.spill_regs > 0);
  CHECK(spill.f_spill < 0.5);
  CHECK(analyze(Config{64, 64, 8, 4, 4, 0}, P, T4()).bank.store_As > 1);
  CHECK_EQ(analyze(Config{64, 64, 8, 4, 4, 4}, P, T4()).bank.store_As, 1);
}

// ---- knowledge graph / advisors ---------------------------------------------------------
static KnowledgeGraph load_kg() { return KnowledgeGraph::load(kRoot + "/knowledge/pass_graph.json"); }

struct CtxHolder {
  Config best;
  Analysis an;
  std::vector<HistoryEntry> hist;
  std::set<Config> measured;
  std::mt19937_64 rng{0};
  std::vector<std::string> symptoms;
  explicit CtxHolder(const Config &c) : best(c), an(analyze(c, P, T4())), symptoms(diagnose(c, an, Profile{})) {}
  Context ctx() { return Context{P, T4(), best, an, Profile{}, symptoms, hist, measured, rng}; }
};

TEST(kg_integrity) {
  KnowledgeGraph kg = load_kg();
  CHECK(!kg.edges.empty());
  for (const Remedy &r : kg.edges) {
    CHECK(kg.weights.count(r.symptom) == 1);
    for (const auto &se : r.side_effects) CHECK(kg.weights.count(se) == 1);
    if (r.action != "set_conflict_free") (void)param_from_name(r.param);
  }
}
TEST(kg_spill_retrieves_register_remedies_first) {
  auto rem = retrieve({"register_spill"}, load_kg());
  CHECK(!rem.empty());
  CHECK_EQ(rem[0].symptom, std::string("register_spill"));
  CHECK(rem[0].param == "TM" || rem[0].param == "TN");
  CHECK_EQ(rem[0].action, std::string("decrease"));
}
TEST(kg_advisor_fixes_bank_conflict) {
  CtxHolder h(Config{64, 64, 8, 4, 4, 0});
  KGAdvisor adv(load_kg());
  bool any_pad = false;
  for (const Proposal &p : adv.propose(h.ctx())) any_pad |= p.cfg.PAD != 0;
  CHECK(any_pad);
}
TEST(kg_advisor_reacts_to_spill) {
  CtxHolder h(Config{128, 128, 8, 16, 16, 0});
  KGAdvisor adv(load_kg());
  auto props = adv.propose(h.ctx());
  CHECK(!props.empty());
  bool lowered = false;
  for (const Proposal &p : props) lowered |= p.cfg.TM < 16 || p.cfg.TN < 16;
  CHECK(lowered);
}
TEST(external_advisor_guard_rejects_invalid) {
  CtxHolder h(kDefaultConfig);
  ExternalAdvisor adv("sh " + kRoot + "/core/tests/fake_advisor.sh", load_kg());
  auto props = adv.propose(h.ctx());
  CHECK_EQ(props.size(), size_t(1));
  CHECK_EQ(adv.accepted, 1);
  CHECK_EQ(adv.rejected, 2); // out-of-domain + malformed
  CHECK(props[0].why.rfind("LLM:", 0) == 0);
  CHECK(adv.parse_proposals("no json at all", h.ctx()).empty());
  CHECK_EQ(adv.rejected, 3);
  CHECK(adv.parse_proposals(R"({"proposals":[{"BM":256,"BN":256,"BK":32,"TM":8,"TN":8,"PAD":0}]})", h.ctx()).empty()); // smem
  CHECK_EQ(adv.rejected, 4);
}

// ---- search ---------------------------------------------------------------------------------
TEST(spearman_known) {
  CHECK_NEAR(*spearman({1, 2, 3, 4}, {10, 20, 30, 40}), 1.0, 1e-12);
  CHECK_NEAR(*spearman({1, 2, 3, 4}, {40, 30, 20, 10}), -1.0, 1e-12);
  CHECK(!spearman({1, 2}, {1, 2}).has_value());
}
TEST(tune_trace_contract) {
  SimRunner runner(1, kRoot);
  KGAdvisor adv(load_kg());
  TuneOptions o; o.budget = 24; o.seed = 1;
  json::Value tr = tune(P, T4(), runner, adv, o);
  CHECK_EQ(tr.at("source").as_string(), std::string("simulated"));
  CHECK(tr.at("warning").as_string().find("SIMULATED") != std::string::npos);
  CHECK_EQ(tr.at("trials").size(), size_t(24));
  CHECK_EQ(tr.at("trials").items()[0].at("origin").as_string(), std::string("baseline"));
  CHECK(tr.at("best_gflops").as_number() >= tr.at("baseline_gflops").as_number());
  CHECK(tr.at("cublas_gflops").is_null()); // never invent a cuBLAS number
  std::set<std::string> seen;
  for (const auto &t : tr.at("trials").items()) seen.insert(t.at("cfg").dump());
  CHECK_EQ(seen.size(), size_t(24)); // no config measured twice
  CHECK_EQ(tr.at("stages").size(), size_t(3));
  json::Value::parse(tr.dump(1)); // serializes to valid JSON
}
TEST(tune_is_deterministic) {
  auto run = [] {
    SimRunner r(7, kRoot); KGAdvisor a(load_kg());
    TuneOptions o; o.budget = 16; o.seed = 7;
    return tune(P, T4(), r, a, o).at("best_gflops").as_number();
  };
  CHECK_EQ(run(), run());
}
TEST(all_advisors_fill_the_budget) {
  for (const char *n : {"none", "random", "rules", "kg"}) {
    SimRunner r(2, kRoot);
    auto adv = make_advisor(n, kRoot + "/knowledge/pass_graph.json", "");
    TuneOptions o; o.budget = 12; o.seed = 2;
    CHECK_EQ(tune(P, T4(), r, *adv, o).at("trials").size(), size_t(12));
  }
}
TEST(cold_start_skips_cost_model_seeding) {
  SimRunner r(3, kRoot); RandomAdvisor a;
  TuneOptions o; o.budget = 6; o.seed = 3; o.seed_trials = 1;
  json::Value tr = tune(P, T4(), r, a, o);
  for (size_t i = 1; i < tr.at("trials").size(); ++i)
    CHECK(tr.at("trials").items()[i].at("origin").as_string() != "cost-model");
}

int main() {
  int failed_tests = 0;
  for (const TestCase &t : registry()) {
    int before = g_failures;
    try { t.fn(); } catch (const std::exception &e) { std::printf("    EXCEPTION: %s\n", e.what()); ++g_failures; }
    const bool ok = g_failures == before;
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", t.name);
    failed_tests += !ok;
  }
  std::printf("\n%zu tests, %d failed\n", registry().size(), failed_tests);
  return failed_tests ? 1 : 0;
}
