#ifndef PTXFORGE_PASSES
#define PTXFORGE_PASSES

include "mlir/Pass/PassBase.td"

def FuseEpilogue : Pass<"ptxforge-fuse-epilogue"> {
  let constructor = "ptxforge::createFuseEpiloguePass()";
  let summary = "Fuse elementwise epilogue chains and mark them for tile-and-fuse into the producing contraction";
  let description = [{
    Stage 1 (pattern rewrite): merges back-to-back single-use elementwise `linalg.generic`
    ops (bias-add -> GELU/ReLU/...) using the upstream elementwise-fusion patterns, restricted
    by a control function so that we never fuse across a multi-use value or a non-elementwise op.

    Stage 2 (analysis + annotation): finds elementwise consumers whose operand is produced by
    a single-use contraction (matmul / batch_matmul / contraction interface) and tags them:
      * consumer  : `ptxforge.epilogue`
      * contraction: `ptxforge.fuse_into_consumer`
    The transform-dialect schedule (scripts/schedule_template.mlir) then tiles the tagged
    consumer to thread blocks and pulls the contraction into the tile loop, so the
    accumulator never round-trips through global memory.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect",
                           "mlir::arith::ArithDialect"];
}

def PadWorkgroupMemory : Pass<"ptxforge-pad-workgroup-memory"> {
  let constructor = "ptxforge::createPadWorkgroupMemoryPass()";
  let summary = "Pad rows of 2-D workgroup (shared) memory allocations to remove bank conflicts";
  let description = [{
    For every static 2-D `memref.alloc` in the GPU workgroup address space, chooses the
    smallest row padding that minimizes the modelled bank-conflict degree for a column-wise
    warp access (see PtxForge/BankModel.h), allocates the padded buffer, and replaces the
    original value with a `memref.subview` of it, so downstream users keep the logical shape.
  }];
  let options = [
    Option<"accessBytes", "access-bytes", "int64_t", /*default=*/"4",
           "Per-thread shared-memory access width in bytes (4 = scalar fp32, 16 = float4)">,
    Option<"alignBytes", "align-bytes", "int64_t", /*default=*/"0",
           "If > 0, padded row stride (bytes) must be a multiple of this (e.g. 16 for vector loads)">,
    Option<"maxPad", "max-pad", "int64_t", /*default=*/"32",
           "Largest padding (in elements) the pass is allowed to add">
  ];
  let dependentDialects = ["mlir::memref::MemRefDialect", "mlir::gpu::GPUDialect"];
}

def ApproximateMath : Pass<"ptxforge-approximate-math"> {
  let constructor = "ptxforge::createApproximateMathPass()";
  let summary = "Replace math.tanh, math.erf and math.absf with arith ops so NVPTX needs no libdevice";
  let description = [{
    GELU needs tanh (or erf). On NVPTX, convert-gpu-to-nvvm turns these into calls to CUDA's
    libdevice (`__nv_tanhf`, `__nv_erff`), which then has to be found and linked from a CUDA
    toolkit. This pass rewrites them first into the upstream MLIR polynomial approximations,
    which are plain arith ops, so the kernel lowers to self-contained PTX with no libdevice.
    The f32 tanh approximation is the one XLA uses (maximum error a few ulp). `math.absf`, which
    those approximations use and which LLVM 19's NVVM lowering also sends to libdevice, is rewritten
    exactly as a sign-bit clear.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::math::MathDialect"];
}

def EraseSelfCopies : Pass<"ptxforge-erase-self-copies"> {
  let constructor = "ptxforge::createEraseSelfCopiesPass()";
  let summary = "Erase memref.copy / linalg.copy ops whose source and target are the same buffer";
  let description = [{
    Bufferizing a tiled `scf.forall` writes each tile in place and then copies the subview onto
    itself (the `tensor.parallel_insert_slice`). MLIR 19 does not fold these no-op copies; inside
    a GPU kernel each one becomes a loop of redundant loads and stores, so this pass removes them.
  }];
}

def OrderKernelArgs : Pass<"ptxforge-order-kernel-args", "mlir::ModuleOp"> {
  let constructor = "ptxforge::createOrderKernelArgsPass()";
  let summary = "Order outlined GPU kernel parameters like the host function's arguments";
  let description = [{
    `gpu-kernel-outlining` orders kernel parameters by first use inside the kernel body, so the
    PTX entry point's signature changes with the schedule (the fused GEMM kernel came out as
    (C, A, B, bias) for a source function taking (A, B, bias, C)). Run right after outlining, this
    pass sorts each kernel's parameters, and its gpu.launch_func operands, by the argument number of
    the host-function argument they come from, so the kernel ABI matches the source signature.
  }];
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

def NormalizeWorkgroupAllocs : Pass<"ptxforge-normalize-workgroup-allocs"> {
  let constructor = "ptxforge::createNormalizeWorkgroupAllocsPass()";
  let summary = "Fold promote's byte buffer + memref.view into a typed shared-memory allocation";
  let description = [{
    `transform.structured.promote` allocates each shared-memory tile as an i8 buffer, views it as the
    tile type and deallocates it. This pass rewrites the alloc + view into a single typed
    `memref.alloc` in workgroup memory and erases the dealloc, so the padding pass can change its row
    length. Anything other than one view at byte offset 0 filling the buffer is left alone.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def WorkgroupSync : Pass<"ptxforge-workgroup-sync"> {
  let constructor = "ptxforge::createWorkgroupSyncPass()";
  let summary = "Make shared-memory allocations kernel attributions and add the K-loop barriers";
  let description = [{
    Run on outlined kernels. Every `memref.alloc` in a `gpu.func` kernel becomes a workgroup
    attribution with 16-byte alignment (`llvm.align`, for 128-bit shared loads); the schedule only puts
    allocations in per-block code, where an allocation means one buffer per thread block. Then, in
    every `scf.for` in a kernel whose body writes shared memory, a `gpu.barrier` goes before
    the first op that reads shared memory and another at the end of the body. Fails if a loop body
    reads shared memory before its last shared-memory write, or on a dynamic allocation.
  }];
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

#endif // PTXFORGE_PASSES
