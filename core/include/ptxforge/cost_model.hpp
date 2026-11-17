// cost_model.hpp — analytic roofline + occupancy + register-cap + issue-rate + latency + tail model.
//
// This is a PRIOR used to rank/prune configs and place points on the roofline. It is not
// ground truth; the trace labels every value from it as `modeled`, and the search re-measures
// whatever it cares about. Calibrating it (report the rank correlation) is milestone M5.
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "PtxForge/BankModel.h"
#include "ptxforge/hw.hpp"
#include "ptxforge/json.hpp"
#include "ptxforge/space.hpp"

namespace ptxforge {

// Shared-memory conflict degree of the three access patterns in kernels/gemm_bias_gelu.cu.
struct KernelBank {
  int store_As = 1, read_As = 1, read_Bs = 1;
  int worst() const { return std::max({store_As, read_As, read_Bs}); }
  int extra() const { return (store_As - 1) + (read_As - 1) + (read_Bs - 1); }
};

inline KernelBank kernel_bank_degrees(const Config &c) {
  const int tx_n = c.BN / c.TN;
  KernelBank k;
  k.store_As = transposedStoreDegree(c.BK, c.BM, c.PAD, 4);
  std::vector<int64_t> a(kWarpSize), b(kWarpSize);
  for (int lane = 0; lane < kWarpSize; ++lane) {
    a[static_cast<size_t>(lane)] = static_cast<int64_t>((lane / tx_n) * c.TM) * 4;
    b[static_cast<size_t>(lane)] = static_cast<int64_t>((lane % tx_n) * c.TN) * 4;
  }
  k.read_As = conflictDegree(a, 4);
  k.read_Bs = conflictDegree(b, 4);
  return k;
}

// Fixed exposed latency of one K step's global loads, in cycles. The kernel has no double buffering, so
// each block waits about one DRAM round trip per K step; other resident blocks hide it if they have
// enough FMAs queued. Set from typical global-memory latency on recent GPUs, not fitted to a trace.
inline constexpr double kLoadLatencyCycles = 600.0;

// FP32 lanes per SM: 64 on Pascal, Volta, Turing and A100; 128 on GA10x, Ada, Hopper and Blackwell.
inline int fp32_lanes_per_sm(const std::string &arch) {
  return (arch == "sm_86" || arch == "sm_87" || arch == "sm_89" || arch == "sm_90" || arch == "sm_100" ||
          arch == "sm_120")
             ? 128
             : 64;
}

// Issue-rate model of one K step, per warp, for kernels/gemm_bias_gelu.cu. Each k does TM x TN FMAs and
// reads TM floats of As and TN floats of Bs. The SM issues lanes/32 warp-FMAs per cycle; the shared pipe
// serves one 128-byte phase per cycle, times the bank-conflict degree inside the phase. The reads are
// consecutive floats, so ptxas issues them as the widest vector their alignment allows (float4 when the
// count and the row length are multiples of 4, float2 for multiples of 2): a float4 warp load takes
// 4 phases of 8 lanes. Costing every load instruction this way, instead of taking the worst scalar
// pattern, is what separates micro-tile shapes that the old model scored identically.
struct SmemPipe {
  int vec_A = 1, vec_B = 1;      // floats per shared load instruction
  int cyc_A = 0, cyc_B = 0;      // shared-pipe cycles for one k's A / B loads
  double lds_cycles = 0, sts_cycles = 0, fma_cycles = 0;
  double factor() const { return fma_cycles / std::max(fma_cycles, lds_cycles + sts_cycles); }
};

// Widest vector (4, 2 or 1 floats) for `count` consecutive floats at a multiple of `count` in rows of
// `row` floats.
inline int vector_width(int count, int row) {
  for (int w : {4, 2})
    if (count % w == 0 && row % w == 0) return w;
  return 1;
}

// Shared-pipe cycles for one warp load: phases x worst conflict degree within a phase.
inline int warp_load_cycles(const std::vector<int64_t> &byteAddrs, int vec) {
  const int bytes = 4 * vec, phases = std::max(1, kWarpSize * bytes / (kNumBanks * kBankBytes));
  return phases * conflictDegree(byteAddrs, bytes);
}

inline SmemPipe smem_pipe(const Config &c, int lanes) {
  const int tx_n = c.BN / c.TN, threads = c.threads();
  SmemPipe s;
  s.vec_A = vector_width(c.TM, c.BM + c.PAD);
  s.vec_B = vector_width(c.TN, c.BN);
  std::vector<int64_t> addr(kWarpSize);
  for (int i = 0; i < c.TM; i += s.vec_A) {
    for (int lane = 0; lane < kWarpSize; ++lane) addr[static_cast<size_t>(lane)] = ((lane / tx_n) * c.TM + i) * 4;
    s.cyc_A += warp_load_cycles(addr, s.vec_A);
  }
  for (int j = 0; j < c.TN; j += s.vec_B) {
    for (int lane = 0; lane < kWarpSize; ++lane) addr[static_cast<size_t>(lane)] = ((lane % tx_n) * c.TN + j) * 4;
    s.cyc_B += warp_load_cycles(addr, s.vec_B);
  }
  s.lds_cycles = static_cast<double>(c.BK) * (s.cyc_A + s.cyc_B);
  const int stores_A = (c.BM * c.BK + threads - 1) / threads, stores_B = (c.BK * c.BN + threads - 1) / threads;
  s.sts_cycles = static_cast<double>(stores_A) * transposedStoreDegree(c.BK, c.BM, c.PAD, 4) + stores_B;
  s.fma_cycles = static_cast<double>(c.BK) * c.TM * c.TN / (lanes / 32.0);
  return s;
}

// Counters from a (simulated or real) run. Fields are optional: a real run without CUPTI has no
// bank-conflict counter, and that must not silently become "0".
struct Profile {
  std::optional<double> regs, local_bytes, static_smem, blocks_per_sm, occupancy, max_err;

  json::Value to_json() const {
    json::Value o = json::Value::object();
    if (regs) o.set("regs", *regs);
    if (local_bytes) o.set("local_bytes", *local_bytes);
    if (static_smem) o.set("static_smem", *static_smem);
    if (blocks_per_sm) o.set("blocks_per_sm", *blocks_per_sm);
    if (occupancy) o.set("occupancy", *occupancy);
    if (max_err) o.set("max_err", *max_err);
    return o;
  }
};

struct Analysis {
  double flops, traffic_bytes, ideal_bytes, ai, ai_ideal, roof_gflops;
  bool memory_bound;
  int regs_est, reg_cap, spill_regs, blocks_per_sm;
  double occupancy;
  KernelBank bank;
  int bank_degree;
  long grid_blocks;
  SmemPipe pipe;
  double waves, f_occ, f_pipe, f_lat, f_spill, f_fill, predicted_gflops;

  json::Value to_json() const {
    auto r4 = [](double x) { return std::round(x * 1e4) / 1e4; };
    json::Value b = json::Value::object();
    b.set("store_As", bank.store_As).set("read_As", bank.read_As).set("read_Bs", bank.read_Bs);
    json::Value o = json::Value::object();
    o.set("flops", flops).set("traffic_bytes", r4(traffic_bytes)).set("ideal_bytes", r4(ideal_bytes))
        .set("ai", r4(ai)).set("ai_ideal", r4(ai_ideal)).set("roof_gflops", r4(roof_gflops))
        .set("memory_bound", memory_bound).set("regs_est", regs_est).set("reg_cap", reg_cap)
        .set("spill_regs", spill_regs).set("blocks_per_sm", blocks_per_sm).set("occupancy", r4(occupancy))
        .set("bank", b).set("bank_degree", bank_degree).set("grid_blocks", grid_blocks)
        .set("waves", r4(waves)).set("f_occ", r4(f_occ)).set("f_pipe", r4(f_pipe)).set("f_lat", r4(f_lat))
        .set("f_spill", r4(f_spill)).set("f_fill", r4(f_fill)).set("predicted_gflops", r4(predicted_gflops));
    return o;
  }
};

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

inline Analysis analyze(const Config &c, const Problem &p, const GPU &hw) {
  Analysis a{};
  a.flops = p.flops();
  a.traffic_bytes = 4.0 * (static_cast<double>(p.M) * p.K * (static_cast<double>(p.N) / c.BN) +
                           static_cast<double>(p.K) * p.N * (static_cast<double>(p.M) / c.BM) +
                           static_cast<double>(p.M) * p.N);
  a.ideal_bytes = 4.0 * (static_cast<double>(p.M) * p.K + static_cast<double>(p.K) * p.N +
                         static_cast<double>(p.M) * p.N);
  a.ai = a.flops / a.traffic_bytes;
  a.ai_ideal = a.flops / a.ideal_bytes;
  const double peak = hw.peak_gflops * hw.sustained;
  a.roof_gflops = std::min(peak, a.ai * hw.bw_gbs);
  a.memory_bound = a.ai * hw.bw_gbs < peak;

  const int threads = c.threads();
  a.regs_est = c.regs_est();
  a.reg_cap = std::min(255, (hw.regs_per_sm / threads) / 8 * 8); // __launch_bounds__ caps registers
  a.spill_regs = std::max(0, a.regs_est - a.reg_cap);
  const int regs = std::min(a.regs_est, a.reg_cap);
  const int regs_alloc = round_up(std::max(regs, 16), 8);
  const int regs_block = round_up(regs_alloc * threads, 256);
  const int by_regs = hw.regs_per_sm / regs_block;
  const int by_threads = hw.max_threads_sm / threads;
  const int by_smem = hw.smem_per_sm / std::max(c.smem_bytes(), 1);
  a.blocks_per_sm = std::max(0, std::min({by_regs, by_threads, by_smem, hw.max_blocks_sm}));
  a.occupancy = static_cast<double>(a.blocks_per_sm) * threads / hw.max_threads_sm;

  a.bank = kernel_bank_degrees(c);
  a.bank_degree = a.bank.worst();

  a.grid_blocks = static_cast<long>(p.M / c.BM) * (p.N / c.BN);
  a.waves = static_cast<double>(a.grid_blocks) / std::max(1, hw.sms * std::max(a.blocks_per_sm, 1));
  a.f_fill = a.waves > 0 ? std::max(0.3, a.waves / std::ceil(a.waves)) : 0.3;

  a.f_occ = a.blocks_per_sm > 0 ? std::pow(std::min(1.0, a.occupancy / 0.5), 0.5) : 0.0;
  // Issue rate: FMA pipe vs shared-memory pipe (see SmemPipe).
  const int lanes = fp32_lanes_per_sm(hw.arch);
  a.pipe = smem_pipe(c, lanes);
  a.f_pipe = a.pipe.factor();
  // Latency: B resident blocks, each with C cycles of FMA work per K step, overlap one exposed load
  // latency L: busy fraction B*C / (B*C + L). Deeper BK means more work per wait.
  const double work = static_cast<double>(threads) * c.BK * c.TM * c.TN / lanes;
  const double resident = std::max(a.blocks_per_sm, 1);
  a.f_lat = resident * work / (resident * work + kLoadLatencyCycles);
  a.f_spill = std::exp(-a.spill_regs / 32.0);

  a.predicted_gflops = a.roof_gflops * a.f_occ * a.f_pipe * a.f_lat * a.f_spill * a.f_fill;
  return a;
}

} // namespace ptxforge
