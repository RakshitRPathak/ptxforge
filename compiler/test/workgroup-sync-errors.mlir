// RUN: not ptxforge-opt --ptxforge-workgroup-sync --split-input-file %s 2>&1 | FileCheck %s

// Reading shared memory before the last write in the same step cannot be fixed with two barriers.
// CHECK: shared memory is read before its last write in this loop body
gpu.module @m2 {
  gpu.func @read_before_write(%out: memref<4xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %s = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
    scf.for %k = %c0 to %c8 step %c1 {
      %v = memref.load %s[%c0] : memref<4xf32, #gpu.address_space<workgroup>>
      memref.store %v, %s[%c1] : memref<4xf32, #gpu.address_space<workgroup>>
    }
    gpu.return
  }
}

// -----

// promote's deallocs must be gone first (ptxforge-normalize-workgroup-allocs).
// CHECK: run ptxforge-normalize-workgroup-allocs first (dealloc found)
gpu.module @m3 {
  gpu.func @dealloc_left(%out: memref<f32>) kernel {
    %c0 = arith.constant 0 : index
    %s = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
    %v = memref.load %s[%c0] : memref<4xf32, #gpu.address_space<workgroup>>
    memref.store %v, %out[] : memref<f32>
    memref.dealloc %s : memref<4xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}
