# MLIR API drift cheat-sheet

The C++ passes were written for **LLVM/MLIR 19** and have **not been compiled** by the kit's author
(no LLVM available in the authoring sandbox). Expect a handful of compile errors on first build;
this table covers the ones most likely to bite. Fixing them is milestone M1; each fix is recorded in the dev log.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `applyPatternsAndFoldGreedily` not found / deprecated | Renamed in LLVM 20 | `applyPatternsGreedily(op, std::move(patterns))` |
| `modifyOpInPlace` not a member of `PatternRewriter` | LLVM < 18 | use `updateRootInPlace` |
| `rewriter.create<Op>(...)` deprecated warnings | LLVM 20+ moved to `Op::create(rewriter, ...)` | switch syntax, or ignore the warning on 19 |
| `GEN_PASS_REGISTRATION` emits `registerPtxForgePasses` undeclared | `-name PtxForge` missing from `mlir_tablegen` | already set in `compiler/CMakeLists.txt`; check the generated `Passes.h.inc` |
| `linalg::isElementwise` / `isaContractionOpInterface` missing | header or namespace moved | include `mlir/Dialect/Linalg/Utils/Utils.h`; for the contraction check use `linalg::isaContractionOpInterface(LinalgOp)` from `mlir/Dialect/Linalg/IR/LinalgInterfaces.h` |
| `populateElementwiseOpsFusionPatterns` signature mismatch | `ControlFusionFn` type changed | it takes `std::function<bool(OpOperand *)>`; check `Linalg/Transforms/Transforms.h` |
| `SubViewOp::inferResultType` arg types | takes `ArrayRef<int64_t>` | pass `SmallVector<int64_t>` explicitly |
| `mlir-opt: unknown option --transform-interpreter` | older LLVM | use `-test-transform-dialect-interpreter` |
| Linker: undefined `MLIR…` symbols | missing LINK_LIBS | add the dialect lib named in the error to `add_mlir_library` |
| `mlir_check_all_link_libraries` failure | RTTI mismatch / lib name | match `LLVM_ENABLE_RTTI`, check library names with `llvm-config --libs` |

Debug loop: `ptxforge-opt --debug-only=ptxforge-fuse-epilogue ...` (needs an assertions-enabled LLVM —
the apt.llvm.org packages are usually release builds; use `--mlir-print-ir-after-all` instead).
