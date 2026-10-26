// Transform-dialect schedule template. The autotuner substitutes the double-brace tile
// placeholders below (BM, BN, BK, TM, TN, PAD) via `ptxforge-tune schedule` (core/src/main.cpp),
// so a new tile config needs NO C++ rebuild.
//
// STATUS: starter. Written against LLVM 19 transform ops from memory — expect to adjust op
// signatures in milestone M1 (use `mlir-opt --help` / the TransformOps docs for your version).
//
// Pipeline (see scripts/lower_to_ptx.sh):
//   ptxforge-opt --ptxforge-fuse-epilogue in.mlir
//     | ptxforge-opt --transform-interpreter schedule.mlir         <- this file
//     | bufferize | gpu lowering | nvvm | translate | PTX

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // 1. The fused epilogue consumer is tagged by ptxforge-fuse-epilogue.
    %epilogue = transform.structured.match attributes{ptxforge.epilogue} in %root
        : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match attributes{ptxforge.fuse_into_consumer} in %root
        : (!transform.any_op) -> !transform.any_op

    // 2. Tile the epilogue to thread-block tiles ({{BM}} x {{BN}}) mapped onto the GPU grid.
    %tiled_epi, %forall = transform.structured.tile_using_forall %epilogue
        tile_sizes [{{BM}}, {{BN}}] (mapping = [#gpu.block<y>, #gpu.block<x>])
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. Pull the matmul into the block loop: accumulator tile never leaves the SM.
    %fused_mm, %forall2 = transform.structured.fuse_into_containing_op %matmul into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. Tile the reduction dimension by {{BK}}.
    %mm_k, %k_loop = transform.structured.tile_using_for %fused_mm tile_sizes [0, 0, {{BK}}]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // TODO(M1): promote A/B tiles to workgroup memory (transform.structured.promote or
    //           bufferization.alloc_tensor with #gpu.address_space<workgroup>)
    // TODO(M1): map per-thread {{TM}} x {{TN}} micro-tiles to threads (#gpu.thread<...>)
    // TODO(M2): run --ptxforge-pad-workgroup-memory on the bufferized IR
    // TODO(M3): vectorize + lower to gpu.subgroup_mma (tensor cores) for fp16
    transform.yield
  }
}
