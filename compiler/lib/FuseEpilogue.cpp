// FuseEpilogue.cpp — pass `ptxforge-fuse-epilogue`.
//
// Written against LLVM/MLIR 19. MLIR's C++ API drifts every release; if something fails to
// compile, check the rename table in docs/MLIR_API_NOTES.md first.
#include "PtxForge/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ptxforge-fuse-epilogue"

namespace ptxforge {
#define GEN_PASS_DEF_FUSEEPILOGUE
#include "PtxForge/Passes.h.inc"
} // namespace ptxforge

using namespace mlir;

namespace {

/// A producer->consumer edge is fusable only if both ends are pure elementwise linalg ops
/// and the intermediate value has exactly one use (otherwise fusion duplicates work or
/// keeps the intermediate alive, defeating the point).
bool isFusableElementwiseEdge(OpOperand *fusedOperand) {
  auto producer = fusedOperand->get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(fusedOperand->getOwner());
  if (!producer || !consumer)
    return false;
  if (!producer->getResult(0).hasOneUse())
    return false;
  return linalg::isElementwise(producer) && linalg::isElementwise(consumer);
}

/// Tag (contraction -> elementwise consumer) pairs for the tile-and-fuse schedule.
unsigned annotateEpiloguePairs(Operation *root) {
  unsigned tagged = 0;
  root->walk([&](linalg::LinalgOp consumer) {
    if (!linalg::isElementwise(consumer))
      return;
    for (OpOperand &in : consumer->getOpOperands()) {
      auto producer = in.get().getDefiningOp<linalg::LinalgOp>();
      if (!producer || !linalg::isaContractionOpInterface(producer))
        continue;
      if (!producer->getResult(0).hasOneUse())
        continue; // matmul result also used elsewhere: cannot sink into the consumer
      OpBuilder b(consumer->getContext());
      consumer->setAttr("ptxforge.epilogue", b.getUnitAttr());
      producer->setAttr("ptxforge.fuse_into_consumer", b.getUnitAttr());
      ++tagged;
      LLVM_DEBUG(llvm::dbgs() << "tagged epilogue pair: " << *producer << "  ->  " << *consumer
                              << "\n");
      break;
    }
  });
  return tagged;
}

struct FuseEpiloguePass : public ptxforge::impl::FuseEpilogueBase<FuseEpiloguePass> {
  void runOnOperation() override {
    Operation *root = getOperation();

    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(patterns, isFusableElementwiseEdge);
    // LLVM 20 renamed this to applyPatternsGreedily.
    if (failed(applyPatternsAndFoldGreedily(root, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    unsigned n = annotateEpiloguePairs(root);
    LLVM_DEBUG(llvm::dbgs() << "ptxforge-fuse-epilogue: tagged " << n << " pair(s)\n");
    (void)n;
  }
};

} // namespace

std::unique_ptr<Pass> ptxforge::createFuseEpiloguePass() {
  return std::make_unique<FuseEpiloguePass>();
}
