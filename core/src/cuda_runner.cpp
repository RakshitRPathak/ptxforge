// cuda_runner.cpp — see cuda_runner.hpp. Built only with -DPTXFORGE_WITH_CUDA.
#ifdef PTXFORGE_WITH_CUDA

#include "ptxforge/cuda_runner.hpp"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>

#ifdef PTXFORGE_WITH_CUBLAS
#include <cublas_v2.h>
#endif

namespace ptxforge {

#define CK(call)                                                                          \
  do {                                                                                    \
    CUresult _r = (call);                                                                 \
    if (_r != CUDA_SUCCESS) {                                                             \
      const char *_s = nullptr;                                                           \
      cuGetErrorString(_r, &_s);                                                          \
      throw std::runtime_error(std::string(#call) + " -> " + (_s ? _s : "unknown CUDA error")); \
    }                                                                                     \
  } while (0)

namespace {
double gelu_ref(double x) {
  return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

// Run a shell command, capture combined stdout+stderr; returns exit status.
int run_capture(const std::string &cmd, std::string &out) {
  FILE *p = popen((cmd + " 2>&1").c_str(), "r");
  if (!p) return -1;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
  return pclose(p);
}
} // namespace

struct CudaRunner::Impl {
  Options opt;
  CUdevice dev{};
  CUcontext ctx{};
  int cc_major = 0, cc_minor = 0, max_threads_sm = 0, num_sm = 0;
  std::string dev_name;

  // problem-sized buffers, (re)allocated when the problem changes
  Problem prob{0, 0, 0};
  std::vector<float> hA, hB, hBias, hC;
  CUdeviceptr dA = 0, dB = 0, dBias = 0, dC = 0;
  std::optional<double> cublas_gflops;

  explicit Impl(Options o) : opt(std::move(o)) {
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuDevicePrimaryCtxRetain(&ctx, dev)); // primary context so cuBLAS can share it
    CK(cuCtxSetCurrent(ctx));
    char name[256];
    CK(cuDeviceGetName(name, sizeof name, dev));
    dev_name = name;
    CK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CK(cuDeviceGetAttribute(&max_threads_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev));
    CK(cuDeviceGetAttribute(&num_sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    std::filesystem::create_directories(opt.workdir);
  }

  ~Impl() {
    release();
    cuDevicePrimaryCtxRelease(dev);
  }

  void release() {
    for (CUdeviceptr *p : {&dA, &dB, &dBias, &dC})
      if (*p) { cuMemFree(*p); *p = 0; }
  }

  void ensure(const Problem &p) {
    if (p.M == prob.M && p.N == prob.N && p.K == prob.K && dA) return;
    release();
    prob = p;
    cublas_gflops.reset();
    const size_t nA = static_cast<size_t>(p.M) * p.K, nB = static_cast<size_t>(p.K) * p.N, nC = static_cast<size_t>(p.M) * p.N;
    hA.resize(nA); hB.resize(nB); hBias.resize(static_cast<size_t>(p.N)); hC.resize(nC);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    for (auto &v : hA) v = U(rng);
    for (auto &v : hB) v = U(rng);
    for (auto &v : hBias) v = U(rng);
    CK(cuMemAlloc(&dA, nA * 4)); CK(cuMemAlloc(&dB, nB * 4));
    CK(cuMemAlloc(&dBias, hBias.size() * 4)); CK(cuMemAlloc(&dC, nC * 4));
    CK(cuMemcpyHtoD(dA, hA.data(), nA * 4)); CK(cuMemcpyHtoD(dB, hB.data(), nB * 4));
    CK(cuMemcpyHtoD(dBias, hBias.data(), hBias.size() * 4));
    CK(cuMemsetD32(dC, 0, nC));
  }

  template <class Launch> double median_ms(Launch &&launch) {
    CUevent s, e;
    CK(cuEventCreate(&s, CU_EVENT_DEFAULT));
    CK(cuEventCreate(&e, CU_EVENT_DEFAULT));
    for (int i = 0; i < opt.warmup; ++i) launch();
    CK(cuCtxSynchronize());
    std::vector<double> per;
    for (int r = 0; r < opt.reps; ++r) {
      CK(cuEventRecord(s, nullptr));
      for (int i = 0; i < opt.iters; ++i) launch();
      CK(cuEventRecord(e, nullptr));
      CK(cuEventSynchronize(e));
      float ms = 0;
      CK(cuEventElapsedTime(&ms, s, e));
      per.push_back(ms / opt.iters);
    }
    std::sort(per.begin(), per.end());
    CK(cuEventDestroy(s));
    CK(cuEventDestroy(e));
    return per[per.size() / 2];
  }

  void measure_cublas() {
#ifdef PTXFORGE_WITH_CUBLAS
    if (!opt.cublas || cublas_gflops) return;
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return;
    CUdeviceptr dC2;
    CK(cuMemAlloc(&dC2, static_cast<size_t>(prob.M) * prob.N * 4));
    const float one = 1.f, zero = 0.f;
    // row-major C = A*B  <=>  column-major C^T = B^T * A^T
    auto cb = [&] {
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, prob.N, prob.M, prob.K, &one, reinterpret_cast<const float *>(dB), prob.N,
                  reinterpret_cast<const float *>(dA), prob.K, &zero, reinterpret_cast<float *>(dC2), prob.N);
    };
    cublas_gflops = prob.flops() / (median_ms(cb) * 1e6);
    cublasDestroy(h);
    CK(cuMemFree(dC2));
#endif
  }

  Measurement run(const std::string &ptx, const std::string &kernel, const Problem &p, int gx, int gy, int block) {
    Measurement m;
    CUmodule mod = nullptr;
    try {
      ensure(p);
      std::string image = ptx;
      image.push_back('\0'); // PTX must be NUL-terminated
      CK(cuModuleLoadData(&mod, image.data()));
      CUfunction fn;
      CK(cuModuleGetFunction(&fn, mod, kernel.c_str()));

      int regs = 0, local = 0, smem = 0, max_block = 0;
      CK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
      CK(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
      CK(cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
      CK(cuFuncGetAttribute(&max_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
      m.profile.regs = regs;
      m.profile.local_bytes = local;
      m.profile.static_smem = smem;
      if (block > max_block) {
        m.error = "block " + std::to_string(block) + " exceeds kernel max " + std::to_string(max_block) + " (register pressure)";
        cuModuleUnload(mod);
        return m;
      }
      int bpsm = 0;
      CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, fn, block, 0));
      m.profile.blocks_per_sm = bpsm;
      m.profile.occupancy = static_cast<double>(bpsm) * block / max_threads_sm;

      int M = p.M, N = p.N, K = p.K;
      void *params[] = {&dA, &dB, &dBias, &dC, &M, &N, &K};
      auto launch = [&] { CK(cuLaunchKernel(fn, static_cast<unsigned>(gx), static_cast<unsigned>(gy), 1, static_cast<unsigned>(block), 1, 1, 0, nullptr, params, nullptr)); };

      // correctness gate FIRST: never report speed for a wrong kernel
      launch();
      CK(cuCtxSynchronize());
      CK(cuMemcpyDtoH(hC.data(), dC, hC.size() * 4));
      double max_err = 0;
      std::mt19937 pick(99);
      for (int s = 0; s < opt.samples; ++s) {
        int r = static_cast<int>(pick() % static_cast<unsigned>(p.M)), c = static_cast<int>(pick() % static_cast<unsigned>(p.N));
        double acc = 0;
        for (int k = 0; k < p.K; ++k) acc += static_cast<double>(hA[static_cast<size_t>(r) * p.K + k]) * hB[static_cast<size_t>(k) * p.N + c];
        const double ref = gelu_ref(acc + hBias[static_cast<size_t>(c)]);
        max_err = std::max(max_err, std::fabs(ref - hC[static_cast<size_t>(r) * p.N + c]) / (1.0 + std::fabs(ref)));
      }
      m.profile.max_err = max_err;
      if (!(max_err < 5e-3)) {
        m.error = "incorrect result (max_err=" + std::to_string(max_err) + ")";
        cuModuleUnload(mod);
        return m;
      }

      const double ms = median_ms(launch);
      m.ok = true;
      m.ms = ms;
      m.gflops = p.flops() / (ms * 1e6);
      measure_cublas();
      m.cublas_gflops = cublas_gflops;
    } catch (const std::exception &e) {
      m.ok = false;
      m.error = e.what();
    }
    if (mod) cuModuleUnload(mod);
    return m;
  }

  std::string ptx_path(const Config &c) const {
    return opt.workdir + "/k_" + std::to_string(c.BM) + "_" + std::to_string(c.BN) + "_" + std::to_string(c.BK) + "_" +
           std::to_string(c.TM) + "_" + std::to_string(c.TN) + "_" + std::to_string(c.PAD) + ".ptx";
  }
};

CudaRunner::CudaRunner(Options o) : impl_(std::make_unique<Impl>(std::move(o))) {}
CudaRunner::~CudaRunner() = default;

Measurement CudaRunner::run_ptx(const std::string &ptx, const std::string &kernel, const Problem &p, int gx, int gy, int block) {
  return impl_->run(ptx, kernel, p, gx, gy, block);
}

Measurement CudaRunner::measure(const Config &c, const Problem &p, const GPU &hw) {
  const std::string path = impl_->ptx_path(c);
  if (!std::filesystem::exists(path)) {
    std::string cmd = impl_->opt.nvcc + " -ptx -arch=" + hw.arch + " -O3 -lineinfo";
    for (const auto &d : c.defines()) cmd += " " + d;
    cmd += " '" + impl_->opt.root + "/kernels/gemm_bias_gelu.cu' -o '" + path + "'";
    std::string log;
    if (run_capture(cmd, log) != 0) {
      Measurement m;
      m.error = "nvcc failed: " + log.substr(log.size() > 800 ? log.size() - 800 : 0);
      return m;
    }
  }
  const auto [gx, gy] = c.grid(p);
  return impl_->run(read_file(path), "gemm_bias_gelu", p, gx, gy, c.threads());
}

std::vector<Stage> CudaRunner::stages(const Config &c, const Problem &, const GPU &) {
  std::string flags;
  for (const auto &d : c.defines()) flags += (flags.empty() ? "" : " ") + d;
  std::string ptx = "// missing";
  try { ptx = read_file(impl_->ptx_path(c)).substr(0, 60000); } catch (const std::exception &) {}
  return {{"CUDA C++ source", "cuda", "// build flags: " + flags + "\n" + read_file(impl_->opt.root + "/kernels/gemm_bias_gelu.cu")},
          {"LLVM IR", "llvm", "// nvcc backend does not expose LLVM IR; the MLIR backend (milestone M2) will."},
          {"PTX", "ptx", ptx}};
}

} // namespace ptxforge

#endif // PTXFORGE_WITH_CUDA
