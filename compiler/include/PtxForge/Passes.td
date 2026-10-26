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

#endif // PTXFORGE_PASSES
