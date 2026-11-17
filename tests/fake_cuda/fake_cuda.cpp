// fake_cuda.cpp: a stand-in for libcuda + libcublas so CudaRunner can be tested without a GPU.
//
// Device memory is host memory, and a "launch" computes C = gelu_tanh(A @ B + bias) on the CPU, so
// the runner's correctness gate sees a correct kernel. cuModuleLoadData behaves like a driver that is
// older than the toolkit: it rejects PTX text with CUDA_ERROR_UNSUPPORTED_PTX_VERSION unless
// FAKE_ACCEPT_PTX is set, and accepts anything that starts with "FAKECUBIN" (what fake_nvcc writes
// for -cubin). Timings are meaningless. Used by tests/fake_cuda/run.sh (make cuda-fallback-test).
#include <cuda.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
extern "C" {
static int g_ptx_loads = 0, g_cubin_loads = 0;
CUresult cuInit(unsigned) { return CUDA_SUCCESS; }
CUresult cuDeviceGet(CUdevice *d, int) { *d = 0; return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRetain(CUcontext *c, CUdevice) { *c = (CUcontext)1; return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice) { std::fprintf(stderr, "[fake] ptx loads=%d cubin loads=%d\n", g_ptx_loads, g_cubin_loads); return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext) { return CUDA_SUCCESS; }
CUresult cuDeviceGetName(char *n, int len, CUdevice) { std::strncpy(n, "Fake RTX 5050", len); return CUDA_SUCCESS; }
CUresult cuDeviceGetAttribute(int *v, CUdevice_attribute a, CUdevice) {
  switch (a) { case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *v = 12; break; case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *v = 0; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: *v = 1536; break; default: *v = 20; }
  return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int *v) { *v = 13010; return CUDA_SUCCESS; }
CUresult cuGetErrorString(CUresult r, const char **s) {
  *s = r == CUDA_ERROR_UNSUPPORTED_PTX_VERSION ? "the provided PTX was compiled with an unsupported toolchain." : "fake error"; return CUDA_SUCCESS; }
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) { *p = (CUdeviceptr)std::malloc(n); return CUDA_SUCCESS; }
CUresult cuMemFree_v2(CUdeviceptr p) { std::free((void *)p); return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) { std::memcpy((void *)d, s, n); return CUDA_SUCCESS; }
CUresult cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) { std::memcpy(d, (void *)s, n); return CUDA_SUCCESS; }
CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned v, size_t n) { for (size_t i = 0; i < n; ++i) ((unsigned *)d)[i] = v; return CUDA_SUCCESS; }
CUresult cuModuleLoadData(CUmodule *m, const void *img) {
  const char *s = (const char *)img;
  if (std::strncmp(s, "FAKECUBIN", 9) == 0) { ++g_cubin_loads; *m = (CUmodule)2; return CUDA_SUCCESS; }
  ++g_ptx_loads;
  if (std::getenv("FAKE_ACCEPT_PTX")) { *m = (CUmodule)1; return CUDA_SUCCESS; }
  return CUDA_ERROR_UNSUPPORTED_PTX_VERSION; }
CUresult cuModuleUnload(CUmodule) { return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction *f, CUmodule, const char *) { *f = (CUfunction)1; return CUDA_SUCCESS; }
CUresult cuFuncGetAttribute(int *v, CUfunction_attribute a, CUfunction) { *v = a == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK ? 1024 : 32; return CUDA_SUCCESS; }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, CUfunction, int, size_t) { *n = 4; return CUDA_SUCCESS; }
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **p, void **) {
  const float *A = *(const float **)p[0], *B = *(const float **)p[1], *bias = *(const float **)p[2]; float *C = *(float **)p[3];
  int M = *(int *)p[4], N = *(int *)p[5], K = *(int *)p[6];
  for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) { double a = 0; for (int k = 0; k < K; ++k) a += (double)A[i*K+k]*B[k*N+j];
    double x = a + bias[j]; C[i*N+j] = (float)(0.5*x*(1.0+std::tanh(0.7978845608028654*(x+0.044715*x*x*x)))); }
  return CUDA_SUCCESS; }
CUresult cuCtxSynchronize() { return CUDA_SUCCESS; }
CUresult cuEventCreate(CUevent *e, unsigned) { *e = (CUevent)new double(0); return CUDA_SUCCESS; }
CUresult cuEventRecord(CUevent e, CUstream) { *(double *)e = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent) { return CUDA_SUCCESS; }
CUresult cuEventElapsedTime(float *ms, CUevent s, CUevent e) { *ms = (float)(*(double *)e - *(double *)s) + 0.001f; return CUDA_SUCCESS; }
CUresult cuEventDestroy_v2(CUevent e) { delete (double *)e; return CUDA_SUCCESS; }
cublasStatus_t cublasCreate_v2(cublasHandle_t *h) { *h = (cublasHandle_t)1; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDestroy_v2(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int) { return CUBLAS_STATUS_SUCCESS; }
}
