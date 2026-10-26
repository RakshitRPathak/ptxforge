// runner.hpp — turns a Config into a Measurement.
//
//   SimRunner   NO GPU. Cost model + a hidden per-config "unmodeled" factor + noise, so search/UI/CI run
//               anywhere. Everything it emits is flagged source="simulated"; never quote it as a result.
//   CudaRunner  real (cuda_runner.cpp, built with -DPTXFORGE_WITH_CUDA): nvcc -> PTX, then runs it
//               IN-PROCESS via the CUDA Driver API (cuModuleLoadData/cuLaunchKernel).
#pragma once

#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ptxforge/cost_model.hpp"

namespace ptxforge {

struct Measurement {
  bool ok = false;
  std::optional<double> gflops, ms, cublas_gflops;
  Profile profile;
  std::string error;
};

struct Stage {
  std::string name, lang, text;
};

inline std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

class Runner {
public:
  virtual ~Runner() = default;
  virtual std::string source() const = 0;
  virtual Measurement measure(const Config &, const Problem &, const GPU &) = 0;
  virtual std::vector<Stage> stages(const Config &, const Problem &, const GPU &) { return {}; }
};

inline double hash01(uint64_t a, const std::string &s) {
  uint64_t h = 1469598103934665603ULL ^ (a * 0x9E3779B97F4A7C15ULL); // FNV-1a, then a splitmix finalizer
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
  h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 27; h *= 0x94D049BB133111EBULL; h ^= h >> 31;
  return static_cast<double>(h >> 11) / 9007199254740992.0; // [0,1)
}

class SimRunner : public Runner {
public:
  SimRunner(uint64_t seed, std::string root, uint64_t world = 0) : rng_(seed), root_(std::move(root)), world_(world) {}
  std::string source() const override { return "simulated"; }

  Measurement measure(const Config &c, const Problem &p, const GPU &hw) override {
    Analysis an = analyze(c, p, hw);
    Measurement m;
    if (an.blocks_per_sm == 0) { m.error = "kernel cannot launch (0 blocks/SM)"; return m; }
    const double hidden = (hash01(world_, c.short_name()) - 0.5) * 0.30; // +-15% unmodeled effects
    std::normal_distribution<double> noise(0.0, 0.015);
    const double g = std::min(an.predicted_gflops * (1 + hidden) * (1 + noise(rng_)), hw.peak_gflops * hw.sustained);
    m.ok = true;
    m.gflops = g;
    m.ms = p.flops() / (g * 1e6);
    m.profile.regs = std::min(c.regs_est(), an.reg_cap);
    m.profile.local_bytes = an.spill_regs ? an.spill_regs * 16 : 0;
    m.profile.static_smem = c.smem_bytes();
    m.profile.blocks_per_sm = an.blocks_per_sm;
    m.profile.occupancy = std::round(an.occupancy * 1e4) / 1e4;
    m.profile.max_err = 0.0;
    return m;
  }

  std::vector<Stage> stages(const Config &c, const Problem &, const GPU &) override {
    std::string flags;
    for (const auto &d : c.defines()) flags += (flags.empty() ? "" : " ") + d;
    const std::string note = "(simulated run: no compiler or GPU was used, so there is no real IR here)";
    return {{"CUDA C++ source", "cuda", "// build flags: " + flags + "\n" + read_file(root_ + "/kernels/gemm_bias_gelu.cu")},
            {"LLVM IR", "llvm", note},
            {"PTX", "ptx", note}};
  }

private:
  std::mt19937_64 rng_;
  std::string root_;
  uint64_t world_;
};

} // namespace ptxforge
