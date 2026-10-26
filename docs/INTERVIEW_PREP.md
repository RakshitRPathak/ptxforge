# Interview prep — questions this project invites (with answer sketches)

Answer in your own words with your own measurements. The sketches only point at the idea.

## Your project, directly
1. **Walk me through the pipeline.** linalg on tensors → fuse epilogue → tile + fuse via transform schedule → bufferize → pad shared memory → gpu→nvvm→llvm → PTX → driver JIT → benchmark → trace.
2. **Why fuse the epilogue?** Bias+GELU after GEMM is memory-bound; fusing keeps the accumulator in registers and removes a full read+write of an M×N tensor from HBM.
3. **Why a transform-dialect schedule rather than a C++ tiling pass?** Schedules are data: the tuner changes tile sizes without recompiling; passes stay small and testable.
4. **How do you know your tuned kernel is correct?** Gate on a sampled fp64 reference before timing; failed configs can't win.
5. **What did the knowledge graph/LLM actually buy you?** *(answer from your M5 ablation — including "nothing" if that's what you found).*
6. **Why is your cost model wrong, and how wrong?** No L2 reuse, crude occupancy→latency map, no instruction mix. Report Spearman; use it only to rank/prune.
7. **What would you do with CUPTI that you can't do now?** Measured DRAM bytes → true AI; bank-conflict counters to validate the model; L2 hit rate.

## SSA / IR / compiler core
8. What is SSA? Why do phi nodes (block arguments in MLIR) exist? What's a dominance tree and why must defs dominate uses?
9. MLIR vs LLVM IR: dialects, regions, ops with attributes, progressive lowering. Why not lower straight to LLVM IR?
10. What's a `PatternRewriter` and why must you mutate IR only through it? What does the greedy driver guarantee (and not guarantee)?
11. Phase ordering: why is pass order/parameter choice hard? (Passes enable/disable each other; search space is combinatorial; heuristics are workload-dependent.) Be careful claiming "NP-hard" — say "believed intractable in general; heuristics + search in practice" unless you can cite.
12. What does `linalg` buy you over raw loops? (Named ops, indexing maps, fusion/tiling as transformations on structured ops.)

## GPU microarchitecture
13. Warp, SM, occupancy: what limits blocks per SM? (registers, shared memory, threads, block count). Why isn't maximum occupancy always best? (ILP/register blocking.)
14. Shared-memory banks: 32 banks × 4 B. When is access conflict-free? Broadcast? Why does `+1` padding work and why does it fail for 16-byte vector accesses? (Run `bank_sim`.)
15. Coalescing: what pattern makes a warp's global load one transaction?
16. Register spills: how does `__launch_bounds__` cap registers? What does a spill cost (local memory → L1/L2)?
17. Roofline: what are the axes, what is the ridge point, what does being left of it mean? Why does a bigger tile move a kernel right?
18. Tensor cores: what does `mma.sync` compute, what operand layouts/fragments, why `cp.async` + multistage pipelines.
19. Why do you need `__syncthreads()` twice in the K-loop? (Load-before-use, use-before-overwrite.)

## C++ / systems
20. Why RTTI-free `dyn_cast`/`isa` in LLVM? What is CRTP used for in MLIR pass/op definitions (`PassWrapper`, generated `Base<DerivedT>`)?
21. `SmallVector` vs `std::vector`; `DenseMap` vs `unordered_map`; when does each win?
22. Driver API vs Runtime API: contexts, modules, `cuModuleLoadData` JIT of PTX, why `CudaRunner` retains the *primary* context (so cuBLAS can share it).
23. How would you make the benchmark robust to noisy neighbors/throttling?

## Behavioral hooks
24. "Tell me about a bug." → pick a real one from your DEVLOG (a wrong-kernel-looked-fast story is ideal).
25. "What would you do next with 3 more months?" → tensor cores, FlashAttention fusion, real counters, upstream a pass.

## Say this, not that
- Say: "The harness is verified; the MLIR path currently lowers X; the rest is Y." Not: "It's a full compiler."
- Say: "Here is the trace file; the ablation says…" Not: "The AI finds optimal kernels."
