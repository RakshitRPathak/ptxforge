// RUN: ptxforge-opt --ptxforge-fuse-epilogue --split-input-file %s | FileCheck %s
//
// Contract: a single-use chain of elementwise generics after a contraction is merged into one
// generic tagged `ptxforge.epilogue`, and the contraction is tagged `ptxforge.fuse_into_consumer`.
// Multi-use values are never fused or tagged.

#id = affine_map<(d0, d1) -> (d0, d1)>
#bias_map = affine_map<(d0, d1) -> (d1)>

// bias-add + tanh fuse into one tagged generic; the matmul is tagged.
// CHECK-LABEL: func.func @gemm_bias_gelu
// CHECK:       linalg.matmul {ptxforge.fuse_into_consumer}
// CHECK:       linalg.generic
// CHECK-SAME:  ptxforge.epilogue
// CHECK:       arith.addf
// CHECK-NEXT:  math.tanh
// CHECK-NOT:   linalg.generic
// CHECK:       return
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

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>

// The matmul result is also returned, so it cannot be sunk into the epilogue: nothing is tagged.
// CHECK-LABEL: func.func @matmul_result_used_twice
// CHECK-NOT:   ptxforge.
// CHECK:       return
func.func @matmul_result_used_twice(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %e = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<16x16xf32>) outs(%e : tensor<16x16xf32>) {
    ^bb0(%x: f32, %o: f32):
      %t = math.tanh %x : f32
      linalg.yield %t : f32
  } -> tensor<16x16xf32>
  return %r, %mm : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>

// The intermediate elementwise value is returned too, so fusing would duplicate work: two generics remain.
// CHECK-LABEL: func.func @elementwise_used_twice
// CHECK:       linalg.generic
// CHECK:       arith.addf
// CHECK:       linalg.generic
// CHECK:       math.tanh
func.func @elementwise_used_twice(%X: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %e = tensor.empty() : tensor<16x16xf32>
  %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%X : tensor<16x16xf32>) outs(%e : tensor<16x16xf32>) {
    ^bb0(%x: f32, %o: f32):
      %s = arith.addf %x, %x : f32
      linalg.yield %s : f32
  } -> tensor<16x16xf32>
  %b = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<16x16xf32>) outs(%e : tensor<16x16xf32>) {
    ^bb0(%x: f32, %o: f32):
      %t = math.tanh %x : f32
      linalg.yield %t : f32
  } -> tensor<16x16xf32>
  return %a, %b : tensor<16x16xf32>, tensor<16x16xf32>
}
