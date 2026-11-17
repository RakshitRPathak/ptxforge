// RUN: ptxforge-opt --ptxforge-normalize-workgroup-allocs --split-input-file %s | FileCheck %s --check-prefix=NORM
// RUN: ptxforge-opt --ptxforge-normalize-workgroup-allocs --ptxforge-workgroup-sync --split-input-file %s | FileCheck %s --check-prefix=SYNC
//
// The shape the schedule produces: promote's byte buffers + views + deallocs inside the K loop of a kernel.
// normalize-workgroup-allocs folds each into one typed allocation; workgroup-sync makes them 16-byte
// aligned workgroup attributions and puts a barrier after the copies and one at the end of the step.

// NORM-LABEL: gpu.func @k_step
// NORM:       scf.for
// NORM-NEXT:  %[[A:.*]] = memref.alloc() : memref<64x16xf32, #gpu.address_space<workgroup>>
// NORM-NEXT:  %[[B:.*]] = memref.alloc() : memref<16x64xf32, #gpu.address_space<workgroup>>
// NORM-NOT:   memref.view
// NORM-NOT:   memref.dealloc
// NORM:       vector.store %{{.*}}, %[[A]]
// NORM:       vector.store %{{.*}}, %[[B]]

// SYNC-LABEL: gpu.func @k_step
// SYNC-SAME:  workgroup(%[[A:.*]] : memref<64x16xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i64},
// SYNC-SAME:  %[[B:.*]] : memref<16x64xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i64})
// SYNC:       scf.for
// SYNC-NOT:   memref.alloc
// SYNC:       vector.store %{{.*}}, %[[A]]
// SYNC-NEXT:  vector.store %{{.*}}, %[[B]]
// SYNC-NEXT:  gpu.barrier
// SYNC-NEXT:  vector.load %[[A]]
// SYNC-NEXT:  vector.load %[[B]]
// SYNC:       memref.store
// SYNC-NEXT:  gpu.barrier
// SYNC-NEXT:  }
// The loop that never touches shared memory gets no barrier.
// SYNC:       scf.for
// SYNC-NOT:   gpu.barrier
// SYNC:       gpu.return
gpu.module @m {
  gpu.func @k_step(%ga: memref<64x2048xf32>, %gb: memref<2048x64xf32>, %out: memref<64xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c2048 = arith.constant 2048 : index
    %tid = gpu.thread_id x
    scf.for %k = %c0 to %c2048 step %c16 {
      %rawa = memref.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
      %a = memref.view %rawa[%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<64x16xf32, #gpu.address_space<workgroup>>
      %rawb = memref.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
      %b = memref.view %rawb[%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<16x64xf32, #gpu.address_space<workgroup>>
      %va = vector.load %ga[%tid, %k] : memref<64x2048xf32>, vector<4xf32>
      %vb = vector.load %gb[%k, %tid] : memref<2048x64xf32>, vector<4xf32>
      vector.store %va, %a[%tid, %c0] : memref<64x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      vector.store %vb, %b[%c0, %tid] : memref<16x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      %ra = vector.load %a[%c0, %c0] : memref<64x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      %rb = vector.load %b[%c0, %c0] : memref<16x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      %p = arith.mulf %ra, %rb : vector<4xf32>
      %s = vector.reduction <add>, %p : vector<4xf32> into f32
      memref.store %s, %out[%tid] : memref<64xf32>
      memref.dealloc %rawa : memref<4096xi8, #gpu.address_space<workgroup>>
      memref.dealloc %rawb : memref<4096xi8, #gpu.address_space<workgroup>>
    }
    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %out[%i] : memref<64xf32>
      memref.store %v, %out[%c0] : memref<64xf32>
    }
    gpu.return
  }
}

// -----

// A view at a non-zero offset is not the plain case: normalize leaves it alone.
// NORM-LABEL: func.func @offset_view
// NORM:       memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
// NORM:       memref.view
func.func @offset_view(%v: vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %raw = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %a = memref.view %raw[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<64x16xf32, #gpu.address_space<workgroup>>
  vector.store %v, %a[%c0, %c0] : memref<64x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  return
}
