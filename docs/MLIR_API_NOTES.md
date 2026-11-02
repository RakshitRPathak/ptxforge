# MLIR API drift cheat-sheet

The passes build against **LLVM/MLIR 19.1** (Ubuntu 24.04's `libmlir-19-dev`). The first build needed one fix
(the first row below); everything that came up while getting linalg → PTX working is in `docs/DEVLOG.md`.
The rest of this table is for moving to LLVM 20+, where these APIs change.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `unknown type name '<Pass>Options'` from `Passes.h.inc` | pass declares `options` but only `GEN_PASS_DEF_*` is defined | also `#define GEN_PASS_DECL_<PASS>` before the include (done in `PadWorkgroupMemory.cpp`) |
| kernel gets dozens of extra parameters | constants outside `gpu.launch` are captured when outlining | run `--gpu-launch-sink-index-computations` right before outlining, after every greedy rewrite |
| PTX calls `__nv_tanhf` / `__nv_fabsf` | `convert-gpu-to-nvvm` maps math ops to libdevice | `--ptxforge-approximate-math` first |
| `unrealized_conversion_cast` of `vector<4x4xf32>` in the kernel | n-D vector math isn't lowered by `convert-gpu-to-nvvm` in 19 | `--convert-math-to-llvm` before the NVVM pipeline |

| `applyPatternsAndFoldGreedily` not found / deprecated | Renamed in LLVM 20 | `applyPatternsGreedily(op, std::move(patterns))` |
| `modifyOpInPlace` not a member of `PatternRewriter` | LLVM < 18 | use `updateRootInPlace` |
| `rewriter.create<Op>(...)` deprecated warnings | LLVM 20+ moved to `Op::create(rewriter, ...)` | switch syntax, or ignore the warning on 19 |
| `GEN_PASS_REGISTRATION` emits `registerPtxForgePasses` undeclared | `-name PtxForge` missing from `mlir_tablegen` | already set in `compiler/CMakeLists.txt`; check the generated `Passes.h.inc` |
| `linalg::isElementwise` / `isaContractionOpInterface` missing | header or namespace moved | include `mlir/Dialect/Linalg/Utils/Utils.h`; for the contraction check use `linalg::isaContractionOpInterface(LinalgOp)` from `mlir/Dialect/Linalg/IR/LinalgInterfaces.h` |
| `populateElementwiseOpsFusionPatterns` signature mismatch | `ControlFusionFn` type changed | it takes `std::function<bool(OpOperand *)>`; check `Linalg/Transforms/Transforms.h` |
| `SubViewOp::inferResultType` arg types | takes `ArrayRef<int64_t>` | pass `SmallVector<int64_t>` explicitly |
| `mlir-opt: unknown option --transform-interpreter` | older LLVM | use `-test-transform-dialect-interpreter` |
| lit: `execute_external=True is deprecated` | newer `lit` | `ShTest(execute_external=False)` (done in `compiler/test/lit.cfg.py`) |
| Linker: undefined `MLIR…` symbols | missing LINK_LIBS | add the dialect lib named in the error to `add_mlir_library` |
| `mlir_check_all_link_libraries` failure | RTTI mismatch / lib name | match `LLVM_ENABLE_RTTI`, check library names with `llvm-config --libs` |

Debug loop: `ptxforge-opt --debug-only=ptxforge-fuse-epilogue ...` (needs an assertions-enabled LLVM —
the apt.llvm.org packages are usually release builds; use `--mlir-print-ir-after-all` instead).
