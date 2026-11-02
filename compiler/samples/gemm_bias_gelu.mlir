// GEMM + bias + GELU(tanh), 2048^3 fp32: C = gelu(A @ B + bias).
// Other sizes:  sed 's/2048/1024/g' compiler/samples/gemm_bias_gelu.mlir > gbg_1024.mlir
//
// The function takes buffers (memrefs) and writes C in place, which is what a GPU kernel needs:
// the inputs are viewed as tensors with bufferization.to_tensor, and the result is written into C
// with bufferization.materialize_in_destination. scripts/lower_to_ptx.sh lowers this file to PTX.
//
//   ptxforge-opt --ptxforge-fuse-epilogue compiler/samples/gemm_bias_gelu.mlir
// merges the bias-add and GELU generics into ONE linalg.generic tagged `ptxforge.epilogue`, and tags
// the matmul `ptxforge.fuse_into_consumer`.

#id = affine_map<(d0, d1) -> (d0, d1)>
#bias_map = affine_map<(d0, d1) -> (d1)>

func.func @gemm_bias_gelu(%Am: memref<2048x2048xf32>, %Bm: memref<2048x2048xf32>,
                          %biasm: memref<2048xf32>, %Cm: memref<2048x2048xf32>) {
  %A = bufferization.to_tensor %Am restrict : memref<2048x2048xf32>
  %B = bufferization.to_tensor %Bm restrict : memref<2048x2048xf32>
  %bias = bufferization.to_tensor %biasm restrict : memref<2048xf32>

  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2048x2048xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

  %mm = linalg.matmul ins(%A, %B : tensor<2048x2048xf32>, tensor<2048x2048xf32>)
                      outs(%init : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

  %biased = linalg.generic {indexing_maps = [#id, #bias_map, #id],
                            iterator_types = ["parallel", "parallel"]}
      ins(%mm, %bias : tensor<2048x2048xf32>, tensor<2048xf32>)
      outs(%empty : tensor<2048x2048xf32>) {
    ^bb0(%x: f32, %b: f32, %o: f32):
      %s = arith.addf %x, %b : f32
      linalg.yield %s : f32
  } -> tensor<2048x2048xf32>

  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
      ins(%biased : tensor<2048x2048xf32>)
      outs(%empty : tensor<2048x2048xf32>) {
    ^bb0(%x: f32, %o: f32):
      %half = arith.constant 0.5 : f32
      %one = arith.constant 1.0 : f32
      %k0 = arith.constant 0.7978845608 : f32
      %k1 = arith.constant 0.044715 : f32
      %x2 = arith.mulf %x, %x : f32
      %x3 = arith.mulf %x2, %x : f32
      %t1 = arith.mulf %k1, %x3 : f32
      %t2 = arith.addf %x, %t1 : f32
      %t3 = arith.mulf %k0, %t2 : f32
      %th = math.tanh %t3 : f32
      %p = arith.addf %one, %th : f32
      %hx = arith.mulf %half, %x : f32
      %r = arith.mulf %hx, %p : f32
      linalg.yield %r : f32
  } -> tensor<2048x2048xf32>

  bufferization.materialize_in_destination %out in restrict writable %Cm
      : (tensor<2048x2048xf32>, memref<2048x2048xf32>) -> ()
  return
}
