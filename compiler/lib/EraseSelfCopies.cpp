// EraseSelfCopies.cpp — pass `ptxforge-erase-self-copies`.
//
// After bufferizing a tiled scf.forall, each tile is written in place and then "inserted" by a
// copy from a subview onto the identical subview. Those copies are no-ops; remove them.
#include "PtxForge/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace ptxforge {
#define GEN_PASS_DEF_ERASESELFCOPIES
#include "PtxForge/Passes.h.inc"
} // namespace ptxforge

using namespace mlir;

namespace {

struct EraseMemrefSelfCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
    if (op.getSource() != op.getTarget())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseLinalgSelfCopy : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op, PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() || op.getInputs() != op.getOutputs())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseSelfCopiesPass : public ptxforge::impl::EraseSelfCopiesBase<EraseSelfCopiesPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EraseMemrefSelfCopy, EraseLinalgSelfCopy>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ptxforge::createEraseSelfCopiesPass() {
  return std::make_unique<EraseSelfCopiesPass>();
}
