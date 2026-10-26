// Lit-style test for ptxforge-pad-workgroup-memory.
//   ptxforge-opt --ptxforge-pad-workgroup-memory samples/pad_workgroup.mlir
// Expected: the 8x64 f32 workgroup buffer becomes memref<8x68xf32, workgroup> (pad 4: the
// transposed-store pattern in kernels/gemm_bias_gelu.cu becomes conflict-free) viewed
// through a memref.subview. The global-memory buffer is untouched.

func.func @k() {
  %shared = memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
  %glob = memref.alloc() : memref<8x64xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %shared[%c0, %c0] : memref<8x64xf32, #gpu.address_space<workgroup>>
  memref.store %v, %glob[%c0, %c0] : memref<8x64xf32>
  return
}
