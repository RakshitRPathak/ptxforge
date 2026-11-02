// RUN: ptxforge-opt --ptxforge-pad-workgroup-memory --split-input-file %s | FileCheck %s
// RUN: ptxforge-opt --ptxforge-pad-workgroup-memory="access-bytes=16 align-bytes=16" --split-input-file %s | FileCheck %s --check-prefix=VEC
// RUN: ptxforge-opt --ptxforge-pad-workgroup-memory="max-pad=0" --split-input-file %s | FileCheck %s --check-prefix=NOPAD
//
// Expected pads come from the bank model in PtxForge/BankModel.h (also checked by tools/bank_sim):
//   64 x f32 row = 256 B stride -> a warp reading one column hits one bank 32 times.
//   scalar f32 loads: pad 1 (stride 65 floats) is the smallest conflict-free choice.
//   float4 loads with 16 B row alignment: pad 4 (stride 68 floats).

// CHECK-LABEL: func.func @pads_shared_only
// CHECK:       %[[BIG:.*]] = memref.alloc() {ptxforge.padded} : memref<8x65xf32, #gpu.address_space<workgroup>>
// CHECK:       %[[SUB:.*]] = memref.subview %[[BIG]][0, 0] [8, 64] [1, 1]
// CHECK-SAME:  to memref<8x64xf32, strided<[65, 1]>, #gpu.address_space<workgroup>>
// The global buffer is left alone.
// CHECK:       memref.alloc() : memref<8x64xf32>
// CHECK:       memref.load %[[SUB]]
// VEC-LABEL:   func.func @pads_shared_only
// VEC:         memref.alloc() {ptxforge.padded} : memref<8x68xf32, #gpu.address_space<workgroup>>
// NOPAD-LABEL: func.func @pads_shared_only
// NOPAD-NOT:   memref.subview
func.func @pads_shared_only() {
  %shared = memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
  %glob = memref.alloc() : memref<8x64xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %shared[%c0, %c0] : memref<8x64xf32, #gpu.address_space<workgroup>>
  memref.store %v, %glob[%c0, %c0] : memref<8x64xf32>
  return
}

// -----

// A 63-float row (odd stride) is already conflict-free: it is marked but not padded.
// CHECK-LABEL: func.func @already_conflict_free
// CHECK:       memref.alloc() {ptxforge.padded} : memref<8x63xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:   memref.subview
func.func @already_conflict_free(%out: memref<f32>) {
  %shared = memref.alloc() : memref<8x63xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %v = memref.load %shared[%c0, %c0] : memref<8x63xf32, #gpu.address_space<workgroup>>
  memref.store %v, %out[] : memref<f32>
  return
}

// -----

// Skipped: a dealloc of the original value would not type-check against a subview.
// CHECK-LABEL: func.func @skips_dealloc
// CHECK:       memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:   memref.subview
func.func @skips_dealloc() {
  %shared = memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
  memref.dealloc %shared : memref<8x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Skipped: only static 2-D buffers are padded.
// CHECK-LABEL: func.func @skips_rank3
// CHECK:       memref.alloc() : memref<2x8x64xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:   memref.subview
func.func @skips_rank3(%out: memref<f32>) {
  %shared = memref.alloc() : memref<2x8x64xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %v = memref.load %shared[%c0, %c0, %c0] : memref<2x8x64xf32, #gpu.address_space<workgroup>>
  memref.store %v, %out[] : memref<f32>
  return
}
