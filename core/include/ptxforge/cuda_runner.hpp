// cuda_runner.hpp — real GPU runner. Define PTXFORGE_WITH_CUDA (and link -lcuda [-lcublas -lcudart]).
//
// Pipeline per config:  nvcc -ptx (subprocess, cached)  ->  cuModuleLoadData IN-PROCESS (the driver
// JIT-compiles PTX to SASS)  ->  correctness gate vs an fp64 CPU reference  ->  event timing.
// The MLIR backend (milestone M2) reuses run_ptx() directly: any PTX with the same kernel signature
//   (const float* A, const float* B, const float* bias, float* C, int M, int N, int K)
// can be benchmarked, so the oracle kernel and the compiler's output go through the same gate.
//
// Counters available WITHOUT CUPTI: registers/thread, local (spill) bytes, static shared memory,
// theoretical occupancy, blocks/SM, GFLOP/s, max error, optional cuBLAS SGEMM baseline.
// Bank conflicts / DRAM bytes / L2 hit rate need CUPTI or ncu — see runtime/cupti_hook.md.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ptxforge/runner.hpp"

namespace ptxforge {

class CudaRunner : public Runner {
public:
  struct Options {
    std::string root = ".", nvcc = "nvcc", workdir = "build/kernels";
    bool cublas = true;
    int warmup = 5, iters = 20, reps = 5, samples = 512;
  };

  explicit CudaRunner(Options o);
  ~CudaRunner() override;
  CudaRunner(const CudaRunner &) = delete;
  CudaRunner &operator=(const CudaRunner &) = delete;

  std::string source() const override { return "harness"; }
  Measurement measure(const Config &, const Problem &, const GPU &) override;
  std::vector<Stage> stages(const Config &, const Problem &, const GPU &) override;

  // Benchmark arbitrary PTX text (e.g. from the MLIR pipeline). `block` = threads per block.
  Measurement run_ptx(const std::string &ptx, const std::string &kernel, const Problem &, int grid_x, int grid_y, int block);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ptxforge
