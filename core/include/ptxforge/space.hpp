// space.hpp — search space for the parametric GEMM+bias+GELU kernel / MLIR schedule.
#pragma once

#include <array>
#include <compare>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ptxforge/hw.hpp"
#include "ptxforge/json.hpp"

namespace ptxforge {

enum class Param { BM, BN, BK, TM, TN, PAD };
inline constexpr std::array<Param, 6> kParams{Param::BM, Param::BN, Param::BK,
                                              Param::TM, Param::TN, Param::PAD};

inline const char *param_name(Param p) {
  switch (p) {
  case Param::BM: return "BM";
  case Param::BN: return "BN";
  case Param::BK: return "BK";
  case Param::TM: return "TM";
  case Param::TN: return "TN";
  case Param::PAD: return "PAD";
  }
  return "?";
}

inline Param param_from_name(std::string_view n) {
  for (Param p : kParams)
    if (n == param_name(p)) return p;
  throw std::runtime_error("unknown parameter '" + std::string(n) + "'");
}

inline const std::vector<int> &domain(Param p) {
  static const std::vector<int> tile{32, 64, 128, 256}, bk{4, 8, 16, 32}, micro{2, 4, 8, 16}, pad{0, 1, 2, 4, 8};
  switch (p) {
  case Param::BM:
  case Param::BN: return tile;
  case Param::BK: return bk;
  case Param::TM:
  case Param::TN: return micro;
  case Param::PAD: return pad;
  }
  return tile;
}

struct Problem {
  int M = 2048, N = 2048, K = 2048;
  double flops() const { return 2.0 * M * N * static_cast<double>(K); } // epilogue FLOPs are O(MN): negligible
};

struct Config {
  int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4, PAD = 0;

  auto operator<=>(const Config &) const = default;

  int get(Param p) const {
    switch (p) {
    case Param::BM: return BM;
    case Param::BN: return BN;
    case Param::BK: return BK;
    case Param::TM: return TM;
    case Param::TN: return TN;
    case Param::PAD: return PAD;
    }
    return 0;
  }
  Config with(Param p, int v) const {
    Config c = *this;
    switch (p) {
    case Param::BM: c.BM = v; break;
    case Param::BN: c.BN = v; break;
    case Param::BK: c.BK = v; break;
    case Param::TM: c.TM = v; break;
    case Param::TN: c.TN = v; break;
    case Param::PAD: c.PAD = v; break;
    }
    return c;
  }

  int threads() const { return (BM / TM) * (BN / TN); }
  int smem_bytes() const { return 4 * (BK * (BM + PAD) + BK * BN); }
  // Rough registers/thread: accumulators + A/B fragments + addressing/loop overhead.
  int regs_est() const { return TM * TN + TM + TN + 26; }
  std::pair<int, int> grid(const Problem &p) const { return {p.N / BN, p.M / BM}; }

  std::string short_name() const {
    return std::to_string(BM) + "x" + std::to_string(BN) + "x" + std::to_string(BK) + " t" +
           std::to_string(TM) + "x" + std::to_string(TN) + " p" + std::to_string(PAD);
  }
  std::vector<std::string> defines() const {
    std::vector<std::string> d;
    for (Param p : kParams) d.push_back(std::string("-D") + param_name(p) + "=" + std::to_string(get(p)));
    return d;
  }
  json::Value to_json() const {
    json::Value o = json::Value::object();
    for (Param p : kParams) o.set(param_name(p), get(p));
    o.set("threads", threads()).set("smem_bytes", smem_bytes());
    return o;
  }
  static Config from_json(const json::Value &v) {
    Config c;
    for (Param p : kParams) c = c.with(p, v.at(param_name(p)).as_int());
    return c;
  }
};

inline constexpr Config kDefaultConfig{64, 64, 8, 4, 4, 0}; // the "untuned" baseline all speedups are against

inline std::optional<std::string> validity_error(const Config &c, const Problem &p, const GPU &hw) {
  for (Param q : kParams) {
    bool in = false;
    for (int v : domain(q)) in |= (v == c.get(q));
    if (!in) return std::string(param_name(q)) + "=" + std::to_string(c.get(q)) + " outside search domain";
  }
  if (p.M % c.BM || p.N % c.BN || p.K % c.BK)
    return "problem size not divisible by tile (kernel has no bounds checks)";
  if (c.BM % c.TM || c.BN % c.TN) return "tile not divisible by micro-tile";
  int t = c.threads();
  if (t < 64 || t > 1024 || t % 32)
    return "threads/block " + std::to_string(t) + " outside [64,1024] or not a multiple of 32";
  if (c.smem_bytes() > hw.smem_per_block)
    return "static shared memory " + std::to_string(c.smem_bytes()) + " B exceeds " +
           std::to_string(hw.smem_per_block) + " B";
  return std::nullopt;
}

inline bool is_valid(const Config &c, const Problem &p, const GPU &hw) {
  return !validity_error(c, p, hw).has_value();
}

inline std::vector<Config> enumerate_space(const Problem &p, const GPU &hw) {
  std::vector<Config> out;
  for (int bm : domain(Param::BM))
    for (int bn : domain(Param::BN))
      for (int bk : domain(Param::BK))
        for (int tm : domain(Param::TM))
          for (int tn : domain(Param::TN))
            for (int pad : domain(Param::PAD)) {
              Config c{bm, bn, bk, tm, tn, pad};
              if (is_valid(c, p, hw)) out.push_back(c);
            }
  return out;
}

// One notch along a parameter's domain (+1 / -1); nullopt at the edge or off-domain.
inline std::optional<Config> step(const Config &c, Param param, int direction) {
  const auto &dom = domain(param);
  for (size_t i = 0; i < dom.size(); ++i) {
    if (dom[i] != c.get(param)) continue;
    long j = static_cast<long>(i) + direction;
    if (j < 0 || j >= static_cast<long>(dom.size())) return std::nullopt;
    return c.with(param, dom[static_cast<size_t>(j)]);
  }
  return std::nullopt;
}

inline std::vector<Config> neighbors(const Config &c, const Problem &p, const GPU &hw) {
  std::vector<Config> out;
  for (Param q : kParams)
    for (int d : {-1, +1})
      if (auto n = step(c, q, d); n && is_valid(*n, p, hw)) out.push_back(*n);
  return out;
}

} // namespace ptxforge
