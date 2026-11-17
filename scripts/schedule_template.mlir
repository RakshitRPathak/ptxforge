// Transform-dialect schedule for the fused GEMM + bias + GELU kernel.
// `ptxforge-tune schedule --BM 64 --BN 64 --BK 16 --TM 4 --TN 4 --out build/ir/schedule.mlir` fills the
// double-brace placeholders, so a new tile config needs no C++ rebuild:
//   TX = BN/TN, TY = BM/TM          threads per block (one thread per TM x TN micro-tile)
//   CA_R x CA_C, CB_R x CB_C        the piece of the A (BM x BK) and B (BK x BN) tile each thread copies
//                                   into shared memory per K step (whole float4s, spread over TX*TY threads)
// Checked against MLIR 19 (scripts/lower_to_ptx.sh, tests/check_lowering_cpu.py).
//
// Input: the output of --ptxforge-fuse-epilogue, which tags the fused epilogue `ptxforge.epilogue` and
// the matmul `ptxforge.fuse_into_consumer`. The function takes memrefs and writes C through
// bufferization.materialize_in_destination (see compiler/samples/gemm_bias_gelu.mlir).
//
// Shape of the kernel this produces, per thread block:
//   for k in 0..K step BK:
//     copy A[BM x BK] and B[BK x BN] into shared memory, every thread a CA/CB piece   (float4 stores)
//     barrier
//     each thread: TM x TN accumulator += A_s[TM rows] * B_s[TN cols]   (accumulator in registers)
//     barrier
//   each thread: C = GELU(acc + bias) for its TM x TN micro-tile
// The barriers are added after the schedule by --ptxforge-workgroup-sync (see the note at step 6).
//
// Not yet here (docs/ROADMAP.md): double buffering, register-level unrolling over K, tensor cores.

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

    // 2. The block walks K in steps of {{BK}}. Each step multiplies a {{BM}} x {{BK}} slice of A by a
    //    {{BK}} x {{BN}} slice of B, which step 4 stages in shared memory for the whole block.
    %mm_k, %kloop = transform.structured.tile_using_for %mm_b tile_sizes [0, 0, {{BK}}]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // The fill and the epilogue are per thread: a {{TM}} x {{TN}} micro-tile each.
    %fill_t, %tf = transform.structured.tile_using_forall %fill_b tile_sizes [{{TM}}, {{TN}}]
        (mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %epi_t, %te = transform.structured.tile_using_forall %epi_b tile_sizes [{{TM}}, {{TN}}]
        (mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. Bufferize. The result is written into C in place; bufferizing the tiled loops leaves copies of
    //    buffers onto themselves, which are erased now, before vectorization would turn each into a
    //    block-sized vector load and store.
    %f0 = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f0 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.any_op
    transform.bufferization.eliminate_empty_tensors %f0 : !transform.any_op
    %buf = transform.bufferization.one_shot_bufferize %root {memcpy_op = "linalg.copy"}
        : (!transform.any_op) -> !transform.any_op
    %bf = transform.structured.match ops{["func.func"]} in %buf : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %bf : !transform.any_op
    %bf2 = transform.apply_registered_pass "ptxforge-erase-self-copies" to %bf : (!transform.any_op) -> !transform.any_op

    // 4. Stage the K step's A and B slices in shared memory: promote copies them into workgroup buffers
    //    and the matmul reads from those. Then every thread computes its {{TM}} x {{TN}} micro-tile of the
    //    step, and the two copies are spread over all {{TX}} x {{TY}} threads, a {{CA_R}} x {{CA_C}} piece of
    //    A and a {{CB_R}} x {{CB_C}} piece of B each (linear thread ids, so any tile shape fits the block).
    %mmb = transform.structured.match attributes{ptxforge.fuse_into_consumer} in %buf
        : (!transform.any_op) -> !transform.any_op
    %mmp = transform.structured.promote %mmb { operands_to_promote = [0, 1], use_full_tile_buffers = [false, false],
        memory_space = #gpu.address_space<workgroup> } : (!transform.any_op) -> !transform.any_op
    %mm_t, %tm = transform.structured.tile_using_forall %mmp tile_sizes [{{TM}}, {{TN}}]
        (mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %kl = transform.structured.match ops{["scf.for"]} in %buf : (!transform.any_op) -> !transform.any_op
    %copies = transform.structured.match ops{["linalg.copy"]} in %kl : (!transform.any_op) -> !transform.any_op
    %copy_a, %copy_b = transform.split_handle %copies : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %ca, %caf = transform.structured.tile_using_forall %copy_a tile_sizes [{{CA_R}}, {{CA_C}}]
        (mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %cb, %cbf = transform.structured.tile_using_forall %copy_b tile_sizes [{{CB_R}}, {{CB_C}}]
        (mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 5. Vectorize everything each thread does: the float4 copies, the TM x BK x TN matmul step and the
    //    epilogue.
    %f = transform.structured.match ops{["func.func"]} in %buf : (!transform.any_op) -> !transform.any_op
    %fv = transform.structured.vectorize_children_and_apply_patterns %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fv {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fv : !transform.any_op

    // 6. Map the block loop to the grid and the thread loops to {{TX}} x {{TY}} threads. No barriers yet
    //    (sync_after_distribute = false): a gpu.barrier inside the K loop would stop step 7 from keeping
    //    the accumulator in registers. --ptxforge-workgroup-sync adds the two each K step needs later.
    %launch = transform.gpu.map_forall_to_blocks %fv generate_gpu_launch
        : (!transform.any_op) -> !transform.any_op
    transform.gpu.map_nested_forall_to_threads %launch block_dims = [{{TX}}, {{TY}}, 1] sync_after_distribute = false
        : (!transform.any_op) -> !transform.any_op

    // 7. Keep the accumulator in registers across the K loop. Hoisting only accepts transfers on a
    //    non-view buffer with identical indices, so fold the subviews into the indices and CSE first.
    //    The block-sized accumulator buffer is then written once and read once by the same thread, so
    //    store-to-load forwarding removes it. Finally lower vector ops to scalar FMAs.
    %func2 = transform.structured.match ops{["func.func"]} in %buf : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op
    %func3 = transform.structured.hoist_redundant_vector_transfers %func2 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func3 : (!transform.any_op) -> ()
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
