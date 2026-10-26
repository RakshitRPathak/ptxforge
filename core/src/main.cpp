// ptxforge-tune — the autotuner CLI.
//
//   ptxforge-tune tune     --runner sim|cuda --gpu t4 --advisor kg --budget 30 --out runs/trace.json
//   ptxforge-tune ablate   --runner sim --advisors none,random,rules,kg --budget 16 --seeds 10
//   ptxforge-tune space    --gpu t4            (size + cost-model top picks)
//   ptxforge-tune dump     --gpu t4            (JSON: every config with modeled numbers; used by the parity test)
//   ptxforge-tune gpus
//   ptxforge-tune schedule --BM 128 --BN 64 --BK 16 --TM 8 --TN 4 --PAD 4 --out build/ir/schedule.mlir
//   ptxforge-tune bench    --ptx k.ptx --grid 32,32 --block 256 --M 2048 --N 2048 --K 2048   (cuda build only)
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include "ptxforge/search.hpp"

#ifdef PTXFORGE_WITH_CUDA
#include "ptxforge/cuda_runner.hpp"
#endif

#ifndef PTXFORGE_ROOT_DIR
#define PTXFORGE_ROOT_DIR "."
#endif

using namespace ptxforge;

namespace {

struct Args {
  std::map<std::string, std::string> kv;
  std::set<std::string> flags;

  static Args parse(int argc, char **argv, int from) {
    static const std::set<std::string> kFlags{"warm-start", "no-cublas", "cublas", "help"};
    Args a;
    for (int i = from; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument '" + k + "'");
      k = k.substr(2);
      if (kFlags.count(k)) { a.flags.insert(k); continue; }
      if (i + 1 >= argc) throw std::runtime_error("missing value for --" + k);
      a.kv[k] = argv[++i];
    }
    return a;
  }
  std::string get(const std::string &k, const std::string &d) const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
  int geti(const std::string &k, int d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stoi(it->second); }
  bool has(const std::string &k) const { return flags.count(k) > 0; }
};

std::string root_dir(const Args &a) {
  if (a.kv.count("root")) return a.kv.at("root");
  if (const char *e = std::getenv("PTXFORGE_ROOT")) return e;
  return PTXFORGE_ROOT_DIR;
}

Problem problem_of(const Args &a) { return Problem{a.geti("M", 2048), a.geti("N", 2048), a.geti("K", 2048)}; }

std::unique_ptr<Runner> make_runner(const Args &a, uint64_t seed) {
  const std::string kind = a.get("runner", "sim");
  if (kind == "sim") return std::make_unique<SimRunner>(seed, root_dir(a));
#ifdef PTXFORGE_WITH_CUDA
  if (kind == "cuda") {
    CudaRunner::Options o;
    o.root = root_dir(a);
    o.nvcc = a.get("nvcc", "nvcc");
    o.workdir = a.get("workdir", "build/kernels");
    o.cublas = !a.has("no-cublas");
    return std::make_unique<CudaRunner>(o);
  }
#endif
  throw std::runtime_error("runner '" + kind + "' unavailable (sim" +
#ifdef PTXFORGE_WITH_CUDA
                           std::string("|cuda") +
#endif
                           "); build with -DPTXFORGE_WITH_CUDA for the real runner");
}

std::string kg_path(const Args &a) { return a.get("kg", root_dir(a) + "/knowledge/pass_graph.json"); }

void write_file(const std::string &path, const std::string &text) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << text;
}

int cmd_tune(const Args &a) {
  const GPU hw = get_gpu(a.get("gpu", "t4"));
  const Problem prob = problem_of(a);
  const uint64_t seed = static_cast<uint64_t>(a.geti("seed", 0));
  auto runner = make_runner(a, seed);
  auto advisor = make_advisor(a.get("advisor", "kg"), kg_path(a), a.get("llm-cmd", ""));
  if (runner->source() == "simulated")
    std::puts("!! SIMULATED RUN — cost model + noise, no GPU. Output is flagged and must not be quoted.\n");
  TuneOptions o;
  o.budget = a.geti("budget", 30);
  o.seed = seed;
  o.log = [](const std::string &s) { std::puts(s.c_str()); };
  json::Value tr = tune(prob, hw, *runner, *advisor, o);
  const std::string out = a.get("out", "runs/trace.json");
  write_file(out, tr.dump(1));
  auto num = [&](const char *k) { return tr.at(k).is_null() ? std::string("n/a") : tr.at(k).dump(); };
  std::printf("\nsource=%s  space=%s configs  trials=%zu\n", tr.at("source").as_string().c_str(), num("space_size").c_str(), tr.at("trials").size());
  std::printf("baseline %s -> best %s GFLOP/s (x%s)\n", num("baseline_gflops").c_str(), num("best_gflops").c_str(), num("speedup_vs_default").c_str());
  if (!tr.at("pct_of_cublas").is_null())
    std::printf("best = %s%% of cuBLAS SGEMM (%s GFLOP/s)\n", num("pct_of_cublas").c_str(), num("cublas_gflops").c_str());
  std::printf("cost-model rank correlation (Spearman): %s\ntrace -> %s\n", num("model_spearman").c_str(), out.c_str());
  return 0;
}

int cmd_ablate(const Args &a) {
  const GPU hw = get_gpu(a.get("gpu", "t4"));
  const Problem prob = problem_of(a);
  const int seeds = a.geti("seeds", 5), budget = a.geti("budget", 20);
  if (a.get("runner", "sim") == "sim")
    std::puts("!! SIMULATED ablation: demonstrates the harness only. The sim world is built from the same\n"
              "   cost model the advisors reason with, so it cannot tell you whether the KG helps on real\n"
              "   hardware. Re-run with --runner cuda on a GPU for evidence.\n");
  struct Row { std::string name; double mean, sd, max; };
  std::vector<Row> rows;
  std::stringstream names(a.get("advisors", "none,random,rules,kg"));
  for (std::string name; std::getline(names, name, ',');) {
    std::vector<double> best;
    for (int s = 0; s < seeds; ++s) {
      auto runner = make_runner(a, static_cast<uint64_t>(s));
      auto adv = make_advisor(name, kg_path(a), a.get("llm-cmd", ""));
      TuneOptions o;
      o.budget = budget;
      o.seed = static_cast<uint64_t>(s);
      if (!a.has("warm-start")) o.seed_trials = 1; // cold start: compare advisors on their own
      best.push_back(tune(prob, hw, *runner, *adv, o).at("best_gflops").as_number());
    }
    double mean = 0, mx = 0;
    for (double b : best) { mean += b / best.size(); mx = std::max(mx, b); }
    double var = 0;
    for (double b : best) var += (b - mean) * (b - mean) / best.size();
    rows.push_back({name, mean, std::sqrt(var), mx});
  }
  std::sort(rows.begin(), rows.end(), [](const Row &x, const Row &y) { return x.mean > y.mean; });
  std::printf("%-10s %10s %8s %10s   (GFLOP/s, budget=%d, seeds=%d)\n", "advisor", "mean best", "std", "max", budget, seeds);
  for (const Row &r : rows) std::printf("%-10s %10.1f %8.1f %10.1f\n", r.name.c_str(), r.mean, r.sd, r.max);
  return 0;
}

int cmd_space(const Args &a) {
  const GPU hw = get_gpu(a.get("gpu", "t4"));
  const Problem prob = problem_of(a);
  std::vector<Config> sp = enumerate_space(prob, hw);
  std::stable_sort(sp.begin(), sp.end(), [&](const Config &x, const Config &y) {
    return analyze(x, prob, hw).predicted_gflops > analyze(y, prob, hw).predicted_gflops;
  });
  std::printf("%zu valid configs for %dx%dx%d on %s\nTop 10 by cost model:\n", sp.size(), prob.M, prob.N, prob.K, hw.name.c_str());
  for (size_t i = 0; i < sp.size() && i < 10; ++i) {
    Analysis an = analyze(sp[i], prob, hw);
    std::printf("  %-22s pred=%7.0f  AI=%5.1f  occ=%.2f  bank=%d-way  spill=%d\n", sp[i].short_name().c_str(),
                an.predicted_gflops, an.ai, an.occupancy, an.bank_degree, an.spill_regs);
  }
  return 0;
}

int cmd_dump(const Args &a) {
  const GPU hw = get_gpu(a.get("gpu", "t4"));
  const Problem prob = problem_of(a);
  json::Value arr = json::Value::array();
  for (const Config &c : enumerate_space(prob, hw)) {
    json::Value e = json::Value::object();
    e.set("cfg", c.to_json()).set("analysis", analyze(c, prob, hw).to_json());
    arr.push_back(e);
  }
  std::puts(arr.dump().c_str());
  return 0;
}

int cmd_gpus(const Args &) {
  for (const GPU &g : all_gpus())
    std::printf("%-5s %-18s %s  SMs=%3d  peak=%7.0f GFLOP/s  bw=%6.0f GB/s  ridge=%5.1f FLOP/B\n", g.key.c_str(),
                g.name.c_str(), g.arch.c_str(), g.sms, g.peak_gflops, g.bw_gbs, g.ridge());
  return 0;
}

int cmd_schedule(const Args &a) {
  Config c = kDefaultConfig;
  for (Param p : kParams) c = c.with(p, a.geti(param_name(p), c.get(p)));
  std::string text = read_file(root_dir(a) + "/scripts/schedule_template.mlir");
  const json::Value cj = c.to_json();
  for (size_t i = 0; i < cj.keys().size(); ++i) {
    const std::string needle = "{{" + cj.keys()[i] + "}}", val = cj.values()[i].dump();
    for (size_t pos = 0; (pos = text.find(needle, pos)) != std::string::npos; pos += val.size())
      text.replace(pos, needle.size(), val);
  }
  if (auto p = text.find("{{"); p != std::string::npos)
    throw std::runtime_error("unrendered placeholder near: " + text.substr(p, 12));
  if (a.kv.count("out")) { write_file(a.kv.at("out"), text); std::printf("wrote %s\n", a.kv.at("out").c_str()); }
  else std::fputs(text.c_str(), stdout);
  return 0;
}

#ifdef PTXFORGE_WITH_CUDA
int cmd_bench(const Args &a) {
  CudaRunner::Options o;
  o.root = root_dir(a);
  o.cublas = a.has("cublas");
  CudaRunner r(o);
  int gx = 0, gy = 0;
  if (std::sscanf(a.get("grid", "").c_str(), "%d,%d", &gx, &gy) != 2) throw std::runtime_error("--grid gx,gy required");
  Measurement m = r.run_ptx(read_file(a.get("ptx", "")), a.get("kernel", "gemm_bias_gelu"), problem_of(a), gx, gy, a.geti("block", 256));
  json::Value j = json::Value::object();
  j.set("ok", m.ok).set("error", m.ok ? json::Value() : json::Value(m.error));
  j.set("gflops", m.gflops ? json::Value(*m.gflops) : json::Value()).set("ms", m.ms ? json::Value(*m.ms) : json::Value());
  j.set("profile", m.profile.to_json()).set("cublas_gflops", m.cublas_gflops ? json::Value(*m.cublas_gflops) : json::Value());
  std::puts(j.dump().c_str());
  return m.ok ? 0 : 4;
}
#endif

void usage() {
  std::puts("usage: ptxforge-tune <tune|ablate|space|dump|gpus|schedule|bench> [--key value ...]\n"
            "see the header comment of core/src/main.cpp for options");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) { usage(); return 64; }
  const std::string cmd = argv[1];
  try {
    Args a = Args::parse(argc, argv, 2);
    if (cmd == "tune") return cmd_tune(a);
    if (cmd == "ablate") return cmd_ablate(a);
    if (cmd == "space") return cmd_space(a);
    if (cmd == "dump") return cmd_dump(a);
    if (cmd == "gpus") return cmd_gpus(a);
    if (cmd == "schedule") return cmd_schedule(a);
#ifdef PTXFORGE_WITH_CUDA
    if (cmd == "bench") return cmd_bench(a);
#endif
    usage();
    return 64;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
