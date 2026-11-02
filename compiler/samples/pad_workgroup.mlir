// Lit-style test for ptxforge-pad-workgroup-memory.
//   ptxforge-opt --ptxforge-pad-workgroup-memory samples/pad_workgroup.mlir
// Expected: the 8x64 f32 workgroup buffer becomes memref<8x65xf32, workgroup> (pad 1 makes a
// column-wise scalar access conflict-free) viewed through a memref.subview; with
// --ptxforge-pad-workgroup-memory="access-bytes=16 align-bytes=16" it becomes 8x68 (pad 4).
// The global-memory buffer is untouched. compiler/test/pad-workgroup.mlir checks all of this.

func.func @k() {
  %shared = memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
  %glob = memref.alloc() : memref<8x64xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %shared[%c0, %c0] : memref<8x64xf32, #gpu.address_space<workgroup>>
  memref.store %v, %glob[%c0, %c0] : memref<8x64xf32>
  return
}
