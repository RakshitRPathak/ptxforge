// advisors.hpp — propose configs to measure next.
//
// An advisor never bypasses validity checks and every proposal is MEASURED, so an advisor can only
// be wrong in a way the benchmark catches.
//   none     no proposals (cost-model + neighbor search only)   ablation baseline
//   random   uniformly random valid configs                      ablation baseline
//   rules    hard-coded diagnosis -> parameter move              "what a human would script"
//   kg       knowledge-graph retrieval with side-effect scoring  deterministic GraphRAG-lite
//   external knowledge-graph context -> an external process      e.g. scripts/llm_propose.py
//            (typically an LLM); the C++ side validates every suggestion
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ptxforge/cost_model.hpp"
#include "ptxforge/kg.hpp"
#include "ptxforge/space.hpp"

namespace ptxforge {

struct Proposal {
  Config cfg;
  std::string why;
};

struct HistoryEntry {
  Config cfg;
  std::optional<double> gflops;
};

struct Context {
  const Problem &prob;
  const GPU &hw;
  Config best;
  Analysis best_an;
  Profile best_profile;
  std::vector<std::string> symptoms;
  const std::vector<HistoryEntry> &history;
  const std::set<Config> &measured;
  std::mt19937_64 &rng;
};

class Advisor {
public:
  virtual ~Advisor() = default;
  virtual std::string name() const = 0;
  virtual std::vector<Proposal> propose(const Context &ctx) = 0;
};

class NoneAdvisor : public Advisor {
public:
  std::string name() const override { return "none"; }
  std::vector<Proposal> propose(const Context &) override { return {}; }
};

class RandomAdvisor : public Advisor {
public:
  std::string name() const override { return "random"; }
  std::vector<Proposal> propose(const Context &ctx) override {
    if (space_.empty()) space_ = enumerate_space(ctx.prob, ctx.hw);
    std::vector<const Config *> pool;
    for (const Config &c : space_)
      if (!ctx.measured.count(c)) pool.push_back(&c);
    if (pool.empty()) return {};
    std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
    return {{*pool[pick(ctx.rng)], "random valid config"}};
  }

private:
  std::vector<Config> space_;
};

// One notch along `param` from the incumbent, if valid and not yet measured.
inline std::optional<Proposal> apply_step(const Context &ctx, Param param, int dir, std::string why) {
  auto n = step(ctx.best, param, dir);
  if (n && is_valid(*n, ctx.prob, ctx.hw) && !ctx.measured.count(*n)) return Proposal{*n, std::move(why)};
  return std::nullopt;
}

class RuleAdvisor : public Advisor {
public:
  std::string name() const override { return "rules"; }
  std::vector<Proposal> propose(const Context &ctx) override {
    std::vector<Proposal> out;
    auto add = [&](std::optional<Proposal> p) { if (p) out.push_back(std::move(*p)); };
    auto has = [&](const char *s) {
      return std::find(ctx.symptoms.begin(), ctx.symptoms.end(), s) != ctx.symptoms.end();
    };
    if (has("register_spill")) {
      add(apply_step(ctx, Param::TM, -1, "spill -> decrease TM"));
      add(apply_step(ctx, Param::TN, -1, "spill -> decrease TN"));
    }
    if (has("bank_conflict")) {
      PadChoice pc = choosePad(ctx.best.BM, 4, 4);
      Config n = ctx.best.with(Param::PAD, pc.pad);
      if (is_valid(n, ctx.prob, ctx.hw) && !ctx.measured.count(n))
        out.push_back({n, "bank conflicts -> PAD=" + std::to_string(pc.pad)});
    }
    if (has("memory_bound")) {
      add(apply_step(ctx, Param::BM, +1, "memory-bound -> larger BM"));
      add(apply_step(ctx, Param::BN, +1, "memory-bound -> larger BN"));
    }
    if (has("low_occupancy")) add(apply_step(ctx, Param::TM, -1, "low occupancy -> decrease TM"));
    return out;
  }
};

class KGAdvisor : public Advisor {
public:
  explicit KGAdvisor(KnowledgeGraph kg) : kg_(std::move(kg)) {}
  std::string name() const override { return "kg"; }

  std::vector<Proposal> propose(const Context &ctx) override {
    std::vector<Remedy> remedies = retrieve(ctx.symptoms, kg_);
    std::vector<Proposal> out;
    for (const Remedy &r : remedies) {
      if (r.score <= 0 && !out.empty()) continue;
      if (auto p = materialize(ctx, r)) out.push_back(std::move(*p));
    }
    // compound move: take the first two remedies' changes together (escapes single-step plateaus)
    if (out.size() >= 2) {
      const Config &a = out[0].cfg, &b = out[1].cfg;
      Config m = ctx.best;
      for (Param p : kParams) m = m.with(p, a.get(p) != ctx.best.get(p) ? a.get(p) : b.get(p));
      if (is_valid(m, ctx.prob, ctx.hw) && !ctx.measured.count(m) && m != a && m != b) {
        auto head = [](const std::string &w) { return w.substr(0, w.find(" \xE2\x80\x94 ")); }; // " — "
        out.insert(out.begin() + 1, {m, "compound: " + head(out[0].why) + " + " + head(out[1].why)});
      }
    }
    return out;
  }

private:
  std::optional<Proposal> materialize(const Context &ctx, const Remedy &r) const {
    std::string why = "[" + r.id + "] " + r.symptom + ": " + r.action + " " + r.param + " \xE2\x80\x94 " + r.why;
    if (r.action == "set_conflict_free") {
      // smallest PAD in the domain that minimizes the modeled conflicts for THIS config
      int best_pad = domain(Param::PAD).front(), best_sum = 1 << 30;
      for (int pad : domain(Param::PAD)) {
        Config t = ctx.best.with(Param::PAD, pad);
        KernelBank kb = kernel_bank_degrees(t);
        int sum = kb.store_As + kb.read_As + kb.read_Bs;
        if (sum < best_sum) { best_sum = sum; best_pad = pad; }
      }
      Config n = ctx.best.with(Param::PAD, best_pad);
      if (n != ctx.best && is_valid(n, ctx.prob, ctx.hw) && !ctx.measured.count(n)) return Proposal{n, why};
      return std::nullopt;
    }
    return apply_step(ctx, param_from_name(r.param), r.action == "increase" ? +1 : -1, why);
  }

  KnowledgeGraph kg_;
};

// Bridge to an external proposer (an LLM script). Protocol: the advisor writes a context JSON file
// and runs `<command> <context-file>`; the process prints {"proposals":[{BM,BN,BK,TM,TN,PAD,why}]}
// on stdout. Everything it returns is validated here — hallucinated, out-of-domain, over-limit and
// duplicate configs are dropped and counted, never executed.
class ExternalAdvisor : public Advisor {
public:
  ExternalAdvisor(std::string command, KnowledgeGraph kg) : cmd_(std::move(command)), kg_(std::move(kg)) {}
  std::string name() const override { return "llm"; }
  int accepted = 0, rejected = 0;

  std::vector<Proposal> propose(const Context &ctx) override {
    std::string path = "/tmp/ptxforge_ctx_" + std::to_string(std::rand()) + ".json";
    {
      std::ofstream f(path);
      f << build_context(ctx).dump(1);
    }
    std::string out;
    if (FILE *p = popen((cmd_ + " '" + path + "'").c_str(), "r")) {
      char buf[4096];
      size_t n;
      while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
      pclose(p);
    }
    std::remove(path.c_str());
    return parse_proposals(out, ctx);
  }

  std::vector<Proposal> parse_proposals(const std::string &text, const Context &ctx) {
    std::vector<Proposal> out;
    json::Value root;
    try {
      auto a = text.find('{'), b = text.rfind('}');
      if (a == std::string::npos || b == std::string::npos || b < a) throw std::runtime_error("no json");
      root = json::Value::parse(text.substr(a, b - a + 1));
      (void)root.at("proposals").items();
    } catch (const std::exception &) {
      ++rejected;
      return out;
    }
    const auto &items = root.at("proposals").items();
    for (size_t i = 0; i < items.size() && i < 3; ++i) { // at most 3 suggestions are considered
      Config c;
      try {
        for (Param p : kParams) c = c.with(p, items[i].at(param_name(p)).as_int());
      } catch (const std::exception &) {
        ++rejected;
        continue;
      }
      if (is_valid(c, ctx.prob, ctx.hw) && !ctx.measured.count(c)) {
        ++accepted;
        std::string why = items[i].contains("why") && items[i].at("why").is_string() ? items[i].at("why").as_string() : "";
        out.push_back({c, "LLM: " + why.substr(0, 200)});
      } else {
        ++rejected;
      }
    }
    return out;
  }

private:
  json::Value build_context(const Context &ctx) const {
    json::Value o = json::Value::object();
    json::Value prob = json::Value::object();
    prob.set("M", ctx.prob.M).set("N", ctx.prob.N).set("K", ctx.prob.K);
    o.set("problem", prob).set("gpu", ctx.hw.to_json()).set("best", ctx.best.to_json())
        .set("profile", ctx.best_profile.to_json());
    json::Value sym = json::Value::array();
    for (const auto &s : ctx.symptoms) sym.push_back(s);
    o.set("symptoms", sym).set("knowledge", subgraph_text(ctx.symptoms, retrieve(ctx.symptoms, kg_)));
    json::Value hist = json::Value::array();
    size_t start = ctx.history.size() > 12 ? ctx.history.size() - 12 : 0;
    for (size_t i = start; i < ctx.history.size(); ++i) {
      json::Value h = json::Value::object();
      h.set("cfg", ctx.history[i].cfg.to_json());
      h.set("gflops", ctx.history[i].gflops ? json::Value(*ctx.history[i].gflops) : json::Value());
      hist.push_back(h);
    }
    o.set("history", hist);
    json::Value dom = json::Value::object();
    for (Param p : kParams) {
      json::Value arr = json::Value::array();
      for (int v : domain(p)) arr.push_back(v);
      dom.set(param_name(p), arr);
    }
    o.set("domains", dom);
    return o;
  }

  std::string cmd_;
  KnowledgeGraph kg_;
};

inline std::unique_ptr<Advisor> make_advisor(const std::string &name, const std::string &kg_path,
                                             const std::string &llm_cmd) {
  if (name == "none") return std::make_unique<NoneAdvisor>();
  if (name == "random") return std::make_unique<RandomAdvisor>();
  if (name == "rules") return std::make_unique<RuleAdvisor>();
  if (name == "kg") return std::make_unique<KGAdvisor>(KnowledgeGraph::load(kg_path));
  if (name == "llm") {
    if (llm_cmd.empty()) throw std::runtime_error("--advisor llm needs --llm-cmd (e.g. 'python3 scripts/llm_propose.py')");
    return std::make_unique<ExternalAdvisor>(llm_cmd, KnowledgeGraph::load(kg_path));
  }
  throw std::runtime_error("unknown advisor '" + name + "' (none|random|rules|kg|llm)");
}

} // namespace ptxforge
