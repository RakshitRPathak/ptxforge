// search.hpp — closed-loop search: cost-model seeding -> advisor proposals / neighbor hill-climb -> measure.
#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>

#include "ptxforge/advisors.hpp"
#include "ptxforge/runner.hpp"

namespace ptxforge {

// Spearman rank correlation with average ranks for ties; nullopt when undefined.
inline std::optional<double> spearman(const std::vector<double> &xs, const std::vector<double> &ys) {
  const size_t n = xs.size();
  if (n < 3 || ys.size() != n) return std::nullopt;
  auto ranks = [n](const std::vector<double> &v) {
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> r(n);
    for (size_t i = 0; i < n;) {
      size_t j = i;
      while (j + 1 < n && v[order[j + 1]] == v[order[i]]) ++j;
      for (size_t k = i; k <= j; ++k) r[order[k]] = (static_cast<double>(i) + static_cast<double>(j)) / 2 + 1;
      i = j + 1;
    }
    return r;
  };
  auto rx = ranks(xs), ry = ranks(ys);
  double mx = std::accumulate(rx.begin(), rx.end(), 0.0) / n, my = std::accumulate(ry.begin(), ry.end(), 0.0) / n;
  double num = 0, dx = 0, dy = 0;
  for (size_t i = 0; i < n; ++i) {
    num += (rx[i] - mx) * (ry[i] - my);
    dx += (rx[i] - mx) * (rx[i] - mx);
    dy += (ry[i] - my) * (ry[i] - my);
  }
  const double den = std::sqrt(dx * dy);
  if (den == 0) return std::nullopt;
  return num / den;
}

struct TuneOptions {
  int budget = 30;
  uint64_t seed = 0;
  std::optional<int> seed_trials;               // 1 = cold start (baseline only)
  std::function<void(const std::string &)> log; // optional progress sink
};

inline std::string iso_now() {
  std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[40];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S+00:00", &tm);
  return buf;
}

inline json::Value tune(const Problem &prob, const GPU &hw, Runner &runner, Advisor &advisor, const TuneOptions &opt) {
  std::mt19937_64 rng(opt.seed);
  const std::vector<Config> space = enumerate_space(prob, hw);
  if (space.empty()) throw std::runtime_error("empty search space: problem size not divisible by any tile");
  std::map<Config, Analysis> pred;
  for (const Config &c : space) pred.emplace(c, analyze(c, prob, hw));
  auto predicted = [&](const Config &c) { return pred.at(c).predicted_gflops; };

  struct Trial {
    Config cfg;
    bool ok;
    std::optional<double> gflops;
    Profile profile;
    std::vector<std::string> symptoms;
    json::Value json;
    std::optional<double> predicted, cublas;
  };
  std::vector<Trial> trials;
  std::set<Config> measured;
  std::vector<HistoryEntry> history;
  auto r2 = [](double x) { return std::round(x * 100.0) / 100.0; };

  auto run = [&](const Config &cfg, const std::string &origin, const std::string &why) {
    auto it = pred.find(cfg);
    const Analysis an = it != pred.end() ? it->second : analyze(cfg, prob, hw);
    Measurement m = runner.measure(cfg, prob, hw);
    measured.insert(cfg);
    Trial t{cfg, m.ok, m.gflops, m.profile, {}, {}, an.predicted_gflops, m.cublas_gflops};
    if (m.ok) t.symptoms = diagnose(cfg, an, m.profile);

    json::Value j = json::Value::object();
    j.set("i", trials.size()).set("cfg", cfg.to_json()).set("origin", origin).set("why", why).set("ok", m.ok);
    j.set("error", m.ok ? json::Value() : json::Value(m.error));
    j.set("gflops", m.gflops ? json::Value(r2(*m.gflops)) : json::Value());
    j.set("ms", m.ms ? json::Value(std::round(*m.ms * 1e4) / 1e4) : json::Value());
    j.set("profile", m.profile.to_json());
    json::Value sym = json::Value::array();
    for (const auto &s : t.symptoms) sym.push_back(s);
    j.set("symptoms", sym);
    json::Value mod = json::Value::object();
    const json::Value full = an.to_json();
    for (const char *k : {"ai", "ai_ideal", "roof_gflops", "memory_bound", "occupancy", "bank", "bank_degree",
                          "spill_regs", "predicted_gflops", "waves"})
      mod.set(k, full.at(k));
    j.set("modeled", mod);
    j.set("cublas_gflops", m.cublas_gflops ? json::Value(*m.cublas_gflops) : json::Value());
    t.json = j;
    history.push_back({cfg, m.gflops ? std::optional<double>(r2(*m.gflops)) : std::nullopt});
    if (opt.log) {
      char line[200];
      std::snprintf(line, sizeof line, "[%3zu] %-22s %-14s ", trials.size(), cfg.short_name().c_str(), origin.c_str());
      std::string s = line;
      if (m.ok) { char g[40]; std::snprintf(g, sizeof g, "%g GFLOP/s", r2(*m.gflops)); s += g; }
      else s += "FAILED: " + m.error.substr(0, 60);
      opt.log(s);
    }
    trials.push_back(std::move(t));
  };

  auto best_index = [&]() -> std::optional<size_t> {
    std::optional<size_t> b;
    for (size_t i = 0; i < trials.size(); ++i)
      if (trials[i].ok && (!b || *trials[i].gflops > *trials[*b].gflops)) b = i;
    return b;
  };

  // 0. baseline: the untuned default — every speedup is reported against THIS measured number
  Config base_cfg = kDefaultConfig;
  if (!is_valid(base_cfg, prob, hw))
    base_cfg = *std::max_element(space.begin(), space.end(), [&](const Config &a, const Config &b) { return predicted(a) < predicted(b); });
  run(base_cfg, "baseline", "untuned default");

  // 1. seed with the cost model's top picks
  std::vector<Config> ranked = space;
  std::stable_sort(ranked.begin(), ranked.end(), [&](const Config &a, const Config &b) { return predicted(a) > predicted(b); });
  const size_t n_seed = static_cast<size_t>(opt.seed_trials ? *opt.seed_trials : std::max(2, opt.budget / 4));
  for (const Config &c : ranked) {
    if (trials.size() >= n_seed) break;
    if (!measured.count(c)) run(c, "cost-model", "top predicted");
  }

  // 2. closed loop
  while (trials.size() < static_cast<size_t>(opt.budget)) {
    std::optional<Config> pick;
    std::string origin, why;
    if (auto bi = best_index()) {
      const Trial &bt = trials[*bi];
      Context ctx{prob, hw, bt.cfg, pred.at(bt.cfg), bt.profile, bt.symptoms, history, measured, rng};
      for (Proposal &p : advisor.propose(ctx)) {
        if (!measured.count(p.cfg) && is_valid(p.cfg, prob, hw)) {
          pick = p.cfg;
          origin = "advisor:" + advisor.name();
          why = p.why;
          break;
        }
      }
      if (!pick) {
        std::optional<Config> bestn;
        for (const Config &n : neighbors(bt.cfg, prob, hw))
          if (!measured.count(n) && (!bestn || predicted(n) > predicted(*bestn))) bestn = n;
        if (bestn) { pick = bestn; origin = "neighbor"; why = "best-predicted neighbor of incumbent"; }
      }
    }
    if (!pick) {
      auto it = std::find_if(ranked.begin(), ranked.end(), [&](const Config &c) { return !measured.count(c); });
      if (it == ranked.end()) break;
      pick = *it; origin = "cost-model"; why = "next best predicted";
    }
    run(*pick, origin, why);
  }

  // ---- assemble the trace -------------------------------------------------------------
  const auto bi = best_index();
  const Trial &base = trials.front();
  std::vector<double> xs, ys;
  std::optional<double> cublas;
  for (const Trial &t : trials) {
    if (t.ok) { xs.push_back(*t.predicted); ys.push_back(*t.gflops); }
    if (!cublas && t.cublas) cublas = t.cublas;
  }
  const bool sim = runner.source() == "simulated";

  json::Value out = json::Value::object();
  out.set("schema", 1).set("source", runner.source());
  out.set("warning", sim ? json::Value("SIMULATED: numbers come from the analytic cost model plus noise, not a GPU. Do not quote them as results.") : json::Value());
  out.set("created", iso_now());
  json::Value pj = json::Value::object();
  pj.set("M", prob.M).set("N", prob.N).set("K", prob.K).set("flops", prob.flops());
  out.set("problem", pj).set("gpu", hw.to_json()).set("advisor", advisor.name()).set("budget", opt.budget)
      .set("seed", opt.seed).set("space_size", space.size());
  out.set("baseline_gflops", base.gflops ? json::Value(r2(*base.gflops)) : json::Value());
  out.set("best_gflops", bi ? json::Value(r2(*trials[*bi].gflops)) : json::Value());
  out.set("best_index", bi ? json::Value(*bi) : json::Value());
  out.set("speedup_vs_default", bi && base.gflops ? json::Value(std::round(*trials[*bi].gflops / *base.gflops * 1000.0) / 1000.0) : json::Value());
  out.set("cublas_gflops", cublas ? json::Value(*cublas) : json::Value());
  out.set("pct_of_cublas", bi && cublas ? json::Value(std::round(100.0 * *trials[*bi].gflops / *cublas * 10.0) / 10.0) : json::Value());
  auto rho = spearman(xs, ys);
  out.set("model_spearman", rho ? json::Value(*rho) : json::Value());
  json::Value tj = json::Value::array();
  for (const Trial &t : trials) tj.push_back(t.json);
  out.set("trials", tj);
  json::Value st = json::Value::array();
  if (bi)
    for (const Stage &s : runner.stages(trials[*bi].cfg, prob, hw)) {
      json::Value sj = json::Value::object();
      sj.set("name", s.name).set("lang", s.lang).set("text", s.text);
      st.push_back(sj);
    }
  out.set("stages", st);
  return out;
}

} // namespace ptxforge
