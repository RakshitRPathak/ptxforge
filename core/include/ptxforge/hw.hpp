// hw.hpp — GPU specs for the cost model and the roofline.
// Datasheet values, rounded: CONFIRM for your card before quoting any "% of peak" figure.
// `sustained` is a prior for the fraction of FP32 peak a tuned kernel can hold (clock/power
// throttling), not a measurement.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptxforge/json.hpp"

namespace ptxforge {

struct GPU {
  std::string key, name, arch;
  int sms;
  double peak_gflops, bw_gbs;
  int regs_per_sm, max_threads_sm, max_blocks_sm, smem_per_sm, smem_per_block;
  double sustained;

  double ridge() const { return peak_gflops / bw_gbs; } // FLOP/B where the roofline turns compute-bound

  json::Value to_json() const {
    json::Value o = json::Value::object();
    o.set("name", name).set("arch", arch).set("sms", sms).set("peak_gflops", peak_gflops)
        .set("bw_gbs", bw_gbs).set("regs_per_sm", regs_per_sm).set("max_threads_sm", max_threads_sm)
        .set("max_blocks_sm", max_blocks_sm).set("smem_per_sm", smem_per_sm)
        .set("smem_per_block", smem_per_block).set("sustained", sustained)
        .set("ridge", std::round(ridge() * 1000.0) / 1000.0);
    return o;
  }
};

inline const std::vector<GPU> &all_gpus() {
  static const std::vector<GPU> g = {
      {"t4", "NVIDIA T4", "sm_75", 40, 8141, 320, 65536, 1024, 16, 65536, 49152, 0.62},
      {"p100", "NVIDIA P100", "sm_60", 56, 9300, 732, 65536, 2048, 32, 65536, 49152, 0.85},
      {"l4", "NVIDIA L4", "sm_89", 58, 30300, 300, 65536, 1536, 24, 102400, 49152, 0.70},
      {"a100", "NVIDIA A100 40GB", "sm_80", 108, 19500, 1555, 65536, 2048, 32, 167936, 49152, 0.85},
  };
  return g;
}

inline const GPU &get_gpu(const std::string &key) {
  for (const GPU &g : all_gpus())
    if (g.key == key) return g;
  std::string choices;
  for (const GPU &g : all_gpus()) choices += (choices.empty() ? "" : ", ") + g.key;
  throw std::runtime_error("unknown gpu '" + key + "'. choices: " + choices);
}

} // namespace ptxforge
