// gemm_bias_gelu.cu — parametric register-blocked SGEMM with fused bias + GELU epilogue.
//
// Role in the project:
//   * Day-1 backend ("nvcc-ptx"): lets the autotuner, harness, and studio run end-to-end
//     before your MLIR pipeline can lower anything.
//   * Oracle/baseline: your MLIR-generated PTX must beat or approach this, and be compared
//     against cuBLAS. Never compare only against a deliberately naive kernel.
//
// Tunables (pass with -D): BM BN BK TM TN PAD
//   block tile BM x BN, K-step BK, per-thread micro-tile TM x TN,
//   PAD = extra floats per row of the transposed A tile in shared memory (bank conflicts).
// Constraints (enforced by core/include/ptxforge/space.hpp):  M%BM==0, N%BN==0, K%BK==0,
//   threads = (BM/TM)*(BN/TN) in [64, 1024].
//
// Launch:  grid = (N/BN, M/BM, 1), block = (BM/TM)*(BN/TN)
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 8
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif
#ifndef PAD
#define PAD 0
#endif

#define THREADS ((BM / TM) * (BN / TN))

__device__ __forceinline__ float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

extern "C" __global__ void __launch_bounds__(THREADS)
gemm_bias_gelu(const float *__restrict__ A, const float *__restrict__ B,
               const float *__restrict__ bias, float *__restrict__ C, int M, int N, int K) {
  __shared__ float As[BK][BM + PAD]; // A tile stored transposed: As[k][m]
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN);
  const int ty = tid / (BN / TN);
  const int rowBase = blockIdx.y * BM;
  const int colBase = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  float regA[TM], regB[TN];

  for (int bk = 0; bk < K; bk += BK) {
    // cooperative loads (strided so any THREADS works)
    for (int i = tid; i < BM * BK; i += THREADS) {
      int r = i / BK, c = i % BK;
      As[c][r] = A[(size_t)(rowBase + r) * K + bk + c];
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      int r = i / BN, c = i % BN;
      Bs[r][c] = B[(size_t)(bk + r) * N + colBase + c];
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regA[i] = As[k][ty * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) regB[j] = Bs[k][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(regA[i], regB[j], acc[i][j]);
    }
    __syncthreads();
  }

  // fused epilogue: bias + GELU stay in registers — no intermediate tensor in HBM
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int r = rowBase + ty * TM + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int c = colBase + tx * TN + j;
      C[(size_t)r * N + c] = gelu_tanh(acc[i][j] + bias[c]);
    }
  }
}
