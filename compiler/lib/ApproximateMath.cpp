// ApproximateMath.cpp — pass `ptxforge-approximate-math`.
//
// Rewrites math.tanh and math.erf into the upstream MLIR polynomial approximations (pure arith),
// so a GELU epilogue lowers to self-contained PTX without linking CUDA's libdevice.
// The approximations use math.absf, which convert-gpu-to-nvvm (LLVM 19) also sends to libdevice
// (__nv_fabsf), so absf is rewritten exactly as "clear the sign bit".
#include "PtxForge/Passes.h"

#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace ptxforge {
#define GEN_PASS_DEF_APPROXIMATEMATH
#include "PtxForge/Passes.h.inc"
} // namespace ptxforge

using namespace mlir;

namespace {

/// |x| for IEEE floats (scalar or vector): clear the sign bit. Bit-exact, including -0.0,
/// infinities and NaN payloads.
struct ExpandAbsF : public OpRewritePattern<math::AbsFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::AbsFOp op, PatternRewriter &rewriter) const override {
    Type ty = op.getType();
    auto ft = dyn_cast<FloatType>(getElementTypeOrSelf(ty));
    if (!ft)
      return failure();
    unsigned bits = ft.getWidth();
    Type elemInt = rewriter.getIntegerType(bits);
    Type intTy = elemInt;
    if (auto vt = dyn_cast<VectorType>(ty))
      intTy = vt.clone(elemInt);
    Location loc = op.getLoc();
    Value asInt = rewriter.create<arith::BitcastOp>(loc, intTy, op.getOperand());
    TypedAttr maskAttr = rewriter.getIntegerAttr(elemInt, llvm::APInt::getSignedMaxValue(bits));
    if (auto vt = dyn_cast<VectorType>(intTy))
      maskAttr = DenseElementsAttr::get(vt, maskAttr);
    Value mask = rewriter.create<arith::ConstantOp>(loc, maskAttr);
    Value cleared = rewriter.create<arith::AndIOp>(loc, asInt, mask);
    rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, ty, cleared);
    return success();
  }
};

struct ApproximateMathPass : public ptxforge::impl::ApproximateMathBase<ApproximateMathPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePolynomialApproximateTanhPattern(patterns);
    populatePolynomialApproximateErfPattern(patterns);
    patterns.add<ExpandAbsF>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ptxforge::createApproximateMathPass() {
  return std::make_unique<ApproximateMathPass>();
}
