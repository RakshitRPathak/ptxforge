// RUN: ptxforge-opt --ptxforge-order-kernel-args --split-input-file %s | FileCheck %s
//
// gpu-kernel-outlining orders parameters by first use; here the body touches %c (the output) first,
// exactly like the fused GEMM kernel. After the pass the kernel takes (a, b, bias, c), the order of
// @host's arguments, and every use inside the body still refers to the same buffer.

// CHECK-LABEL: func.func @host
// CHECK-SAME:  (%[[A:.*]]: memref<4xf32>, %[[B:.*]]: memref<4xf32>, %[[BIAS:.*]]: memref<4xf32>, %[[C:.*]]: memref<4xf32>)
// CHECK:       gpu.launch_func @k::@k
// CHECK-SAME:  args(%[[A]] : memref<4xf32>, %[[B]] : memref<4xf32>, %[[BIAS]] : memref<4xf32>, %[[C]] : memref<4xf32>)
// CHECK:       gpu.func @k(%[[KA:.*]]: memref<4xf32>, %[[KB:.*]]: memref<4xf32>, %[[KBIAS:.*]]: memref<4xf32>, %[[KC:.*]]: memref<4xf32>)
// CHECK-DAG:   memref.load %[[KA]]
// CHECK-DAG:   memref.load %[[KB]]
// CHECK-DAG:   memref.load %[[KBIAS]]
// CHECK:       memref.store %{{.*}}, %[[KC]]
module attributes {gpu.container_module} {
  func.func @host(%a: memref<4xf32>, %b: memref<4xf32>, %bias: memref<4xf32>, %c: memref<4xf32>) {
    %one = arith.constant 1 : index
    gpu.launch_func @k::@k blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%c : memref<4xf32>, %a : memref<4xf32>, %b : memref<4xf32>, %bias : memref<4xf32>)
    return
  }
  gpu.module @k {
    gpu.func @k(%c: memref<4xf32>, %a: memref<4xf32>, %b: memref<4xf32>, %bias: memref<4xf32>) kernel {
      %i = arith.constant 0 : index
      %x = memref.load %a[%i] : memref<4xf32>
      %y = memref.load %b[%i] : memref<4xf32>
      %z = memref.load %bias[%i] : memref<4xf32>
      %p = arith.mulf %x, %y : f32
      %q = arith.addf %p, %z : f32
      memref.store %q, %c[%i] : memref<4xf32>
      gpu.return
    }
  }
}

// -----

// A captured value that is not a host argument (%tmp, a local buffer) moves after the host arguments,
// and the body's two stores still go to the right buffers.
// CHECK-LABEL: func.func @local_buffer_last
// CHECK-SAME:  (%[[A2:.*]]: memref<4xf32>, %[[C2:.*]]: memref<4xf32>)
// CHECK:       %[[T:.*]] = memref.alloc()
// CHECK:       args(%[[A2]] : memref<4xf32>, %[[C2]] : memref<4xf32>, %[[T]] : memref<4xf32>)
// CHECK:       gpu.func @k2(%[[KA2:.*]]: memref<4xf32>, %[[KC2:.*]]: memref<4xf32>, %[[KT:.*]]: memref<4xf32>)
// CHECK:       %[[X:.*]] = memref.load %[[KA2]]
// CHECK:       memref.store %[[X]], %[[KT]]
// CHECK:       memref.store %[[X]], %[[KC2]]
module attributes {gpu.container_module} {
  func.func @local_buffer_last(%a: memref<4xf32>, %c: memref<4xf32>) {
    %one = arith.constant 1 : index
    %tmp = memref.alloc() : memref<4xf32>
    gpu.launch_func @k2::@k2 blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%tmp : memref<4xf32>, %a : memref<4xf32>, %c : memref<4xf32>)
    return
  }
  gpu.module @k2 {
    gpu.func @k2(%t: memref<4xf32>, %a: memref<4xf32>, %c: memref<4xf32>) kernel {
      %i = arith.constant 0 : index
      %x = memref.load %a[%i] : memref<4xf32>
      memref.store %x, %t[%i] : memref<4xf32>
      memref.store %x, %c[%i] : memref<4xf32>
      gpu.return
    }
  }
}

// -----

// Already in source order: left untouched.
// CHECK-LABEL: func.func @unchanged
// CHECK:       args(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>)
// CHECK:       gpu.func @k3(%arg0: memref<4xf32>, %arg1: memref<4xf32>)
// CHECK-NEXT:  %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:  %{{.*}} = memref.load %arg0
module attributes {gpu.container_module} {
  func.func @unchanged(%a: memref<4xf32>, %c: memref<4xf32>) {
    %one = arith.constant 1 : index
    gpu.launch_func @k3::@k3 blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%a : memref<4xf32>, %c : memref<4xf32>)
    return
  }
  gpu.module @k3 {
    gpu.func @k3(%a: memref<4xf32>, %c: memref<4xf32>) kernel {
      %i = arith.constant 0 : index
      %x = memref.load %a[%i] : memref<4xf32>
      memref.store %x, %c[%i] : memref<4xf32>
      gpu.return
    }
  }
}
