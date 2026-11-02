// Transform-dialect schedule for the fused GEMM + bias + GELU kernel.
// `ptxforge-tune schedule --BM 64 --BN 64 --BK 16 --TM 4 --TN 4 --out build/ir/schedule.mlir` fills the
// double-brace placeholders (TX = BN/TN and TY = BM/TM threads per block), so a new tile config needs
// no C++ rebuild. Checked against MLIR 19 (scripts/lower_to_ptx.sh, tests/check_lowering_cpu.py).
//
// Input: the output of --ptxforge-fuse-epilogue, which tags the fused epilogue `ptxforge.epilogue` and
// the matmul `ptxforge.fuse_into_consumer`. The function takes memrefs and writes C through
// bufferization.materialize_in_destination (see compiler/samples/gemm_bias_gelu.mlir).
//
// Not yet here (docs/ROADMAP.md): A/B tiles staged in shared memory (then --ptxforge-pad-workgroup-memory
// has buffers to pad), register-level unrolling, and tensor cores for fp16.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    %epi = transform.structured.match attributes{ptxforge.epilogue} in %root
        : (!transform.any_op) -> !transform.any_op
    %mm = transform.structured.match attributes{ptxforge.fuse_into_consumer} in %root
        : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // 1. Block level: one {{BM}} x {{BN}} output tile per thread block. The matmul (and the fill that
    //    zeroes its accumulator) is pulled into the block loop, so its result never goes to global memory.
    %epi_b, %blocks = transform.structured.tile_using_forall %epi tile_sizes [{{BM}}, {{BN}}]
        (mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm_b, %blocks2 = transform.structured.fuse_into_containing_op %mm into %blocks
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_b, %blocks3 = transform.structured.fuse_into_containing_op %fill into %blocks2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 2. Thread level: each thread owns a {{TM}} x {{TN}} micro-tile and computes the matmul,
    //    bias-add and GELU for it, keeping the accumulator private.
    %epi_t, %threads = transform.structured.tile_using_forall %epi_b tile_sizes [{{TM}}, {{TN}}]
        (mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm_t, %threads2 = transform.structured.fuse_into_containing_op %mm_b into %threads
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_t, %threads3 = transform.structured.fuse_into_containing_op %fill_b into %threads2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. Reduction: walk K in steps of {{BK}}.
    %mm_k, %kloop = transform.structured.tile_using_for %mm_t tile_sizes [0, 0, {{BK}}]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. Vectorize each thread's work (fill, the TM x BK x TN matmul step, the epilogue), so the
    //    accumulator can live in registers. Then clean up and let the result reuse C's buffer.
    %f = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %fv = transform.structured.vectorize_children_and_apply_patterns %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fv {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.any_op
    transform.apply_cse to %fv : !transform.any_op
    transform.bufferization.eliminate_empty_tensors %fv : !transform.any_op

    // 5. Bufferize, then map the block loop to the grid and the thread loop to {{TX}} x {{TY}} threads.
    %buf = transform.bufferization.one_shot_bufferize %root {memcpy_op = "linalg.copy"}
        : (!transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %buf : (!transform.any_op) -> !transform.any_op
    %launch = transform.gpu.map_forall_to_blocks %func generate_gpu_launch
        : (!transform.any_op) -> !transform.any_op
    transform.gpu.map_nested_forall_to_threads %launch block_dims = [{{TX}}, {{TY}}, 1]
        : (!transform.any_op) -> !transform.any_op

    // 6. Keep the accumulator in registers across the K loop, then lower vector ops to scalar FMAs.
    %func2 = transform.structured.match ops{["func.func"]} in %buf : (!transform.any_op) -> !transform.any_op
    %func3 = transform.structured.hoist_redundant_vector_transfers %func2 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func3 {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
      transform.apply_patterns.vector.lower_broadcast
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
