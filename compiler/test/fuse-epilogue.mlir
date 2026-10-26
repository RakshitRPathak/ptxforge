// RUN: ptxforge-opt --ptxforge-fuse-epilogue %s | FileCheck %s
//
// STATUS: written against the pass source, NOT yet run (no LLVM/MLIR in the authoring sandbox).
// Contract: bias-add + GELU elementwise generics fuse into one generic tagged `ptxforge.epilogue`;
// the matmul is tagged `ptxforge.fuse_into_consumer`.

#id = affine_map<(d0, d1) -> (d0, d1)>
#bias_map = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @gemm_bias_gelu
// CHECK:       linalg.matmul {{.*}}ptxforge.fuse_into_consumer
// CHECK:       linalg.generic {{.*}}ptxforge.epilogue
// CHECK:       math.tanh
// CHECK-NOT:   linalg.generic
func.func @gemm_bias_gelu(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>, %bias: tensor<64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %biased = linalg.generic {indexing_maps = [#id, #bias_map, #id], iterator_types = ["parallel", "parallel"]}
      ins(%mm, %bias : tensor<64x64xf32>, tensor<64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%x: f32, %b: f32, %o: f32):
      %s = arith.addf %x, %b : f32
      linalg.yield %s : f32
  } -> tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%biased : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%x: f32, %o: f32):
      %th = math.tanh %x : f32
      linalg.yield %th : f32
  } -> tensor<64x64xf32>
  return %out : tensor<64x64xf32>
}
