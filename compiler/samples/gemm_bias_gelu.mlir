// GEMM + bias + GELU(tanh) on tensors. 2048^3 fp32.
// Edit the shapes with:  sed 's/2048/1024/g' gemm_bias_gelu.mlir
//
//   ptxforge-opt --ptxforge-fuse-epilogue samples/gemm_bias_gelu.mlir
// Expected: the bias-add and GELU generics merge into ONE linalg.generic, tagged
// `ptxforge.epilogue`, and the matmul is tagged `ptxforge.fuse_into_consumer`.

#id = affine_map<(d0, d1) -> (d0, d1)>
#bias_map = affine_map<(d0, d1) -> (d1)>

func.func @gemm_bias_gelu(%A: tensor<2048x2048xf32>, %B: tensor<2048x2048xf32>,
                          %bias: tensor<2048xf32>) -> tensor<2048x2048xf32> {
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

  return %out : tensor<2048x2048xf32>
}
