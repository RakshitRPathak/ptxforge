// RUN: ptxforge-opt --ptxforge-pad-workgroup-memory %s | FileCheck %s
// RUN: ptxforge-opt --ptxforge-pad-workgroup-memory="max-pad=0" %s | FileCheck %s --check-prefix=NOPAD
//
// STATUS: written against the pass source, NOT yet run (no LLVM/MLIR in the authoring sandbox).
// Expected values come from the verified C++ bank model (tools/bank_sim.cpp): for an 8-element-wide
// f32 row the padding search picks the smallest conflict-free pad within the options.

// CHECK-LABEL: func.func @k
// CHECK:       %[[BIG:.*]] = memref.alloc() {ptxforge.padded} : memref<8x{{[0-9]+}}xf32, #gpu.address_space<workgroup>>
// CHECK:       memref.subview %[[BIG]][0, 0] [8, 64] [1, 1]
// The global buffer must be left alone.
// CHECK:       memref.alloc() : memref<8x64xf32>
// NOPAD-LABEL: func.func @k
// NOPAD-NOT:   memref.subview

func.func @k() {
  %shared = memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
  %glob = memref.alloc() : memref<8x64xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %shared[%c0, %c0] : memref<8x64xf32, #gpu.address_space<workgroup>>
  memref.store %v, %glob[%c0, %c0] : memref<8x64xf32>
  return
}
