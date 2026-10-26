// PadWorkgroupMemory.cpp — pass `ptxforge-pad-workgroup-memory`.
//
// memref.alloc<RxC x T, #gpu.address_space<workgroup>>
//   ==>  %big = memref.alloc : memref<Rx(C+pad) x T, workgroup>
//        %sub = memref.subview %big[0,0][R,C][1,1]       // users keep the logical shape
//
// Known sharp edge: users that demand an *identity* layout (some gpu.subgroup_mma_* ops,
// memref.collapse_shape) will reject the strided subview type. Handle them by consuming the
// padded buffer directly (leadDimension = C+pad) — that is milestone M2 in docs/ROADMAP.md.
#include "PtxForge/BankModel.h"
#include "PtxForge/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ptxforge-pad-workgroup-memory"

namespace ptxforge {
#define GEN_PASS_DEF_PADWORKGROUPMEMORY
#include "PtxForge/Passes.h.inc"
} // namespace ptxforge

using namespace mlir;

namespace {

struct PadWorkgroupAlloc : public OpRewritePattern<memref::AllocOp> {
  PadWorkgroupAlloc(MLIRContext *ctx, int64_t accessBytes, int64_t alignBytes, int64_t maxPad)
      : OpRewritePattern(ctx), accessBytes(accessBytes), alignBytes(alignBytes),
        maxPad(maxPad) {}

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType ty = op.getType();
    if (op->hasAttr("ptxforge.padded"))
      return failure();
    if (!ty.hasStaticShape() || ty.getRank() != 2 || !ty.getLayout().isIdentity())
      return failure();

    auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(ty.getMemorySpace());
    if (!space || space.getValue() != gpu::AddressSpace::Workgroup)
      return failure();

    Type elem = ty.getElementType();
    if (!elem.isIntOrFloat() || elem.getIntOrFloatBitWidth() < 8)
      return failure();
    int64_t elemBytes = elem.getIntOrFloatBitWidth() / 8;

    // A dealloc of the original value would no longer type-check against a subview.
    for (Operation *user : op->getUsers())
      if (isa<memref::DeallocOp>(user))
        return failure();

    int64_t rows = ty.getDimSize(0), cols = ty.getDimSize(1);
    ptxforge::PadChoice choice = ptxforge::choosePad(
        static_cast<int>(cols), static_cast<int>(elemBytes), static_cast<int>(accessBytes),
        static_cast<int>(alignBytes), static_cast<int>(maxPad));

    if (choice.pad == 0) {
      // Nothing to do; mark so the greedy driver does not revisit.
      rewriter.modifyOpInPlace(op, [&] { op->setAttr("ptxforge.padded", rewriter.getUnitAttr()); });
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "padding " << ty << " by " << choice.pad << " elems (conflict "
                            << choice.degree << "-way)\n");

    auto paddedTy = MemRefType::get({rows, cols + choice.pad}, elem, AffineMap(),
                                    ty.getMemorySpace());
    Location loc = op.getLoc();
    auto padded = rewriter.create<memref::AllocOp>(loc, paddedTy);
    if (IntegerAttr align = op.getAlignmentAttr())
      padded.setAlignmentAttr(align);
    padded->setAttr("ptxforge.padded", rewriter.getUnitAttr());

    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(rows), rewriter.getIndexAttr(cols)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    auto subTy = cast<MemRefType>(
        memref::SubViewOp::inferResultType(paddedTy, {0, 0}, {rows, cols}, {1, 1}));
    auto sub = rewriter.create<memref::SubViewOp>(loc, subTy, padded, offsets, sizes, strides);
    rewriter.replaceOp(op, sub.getResult());
    return success();
  }

  int64_t accessBytes, alignBytes, maxPad;
};

struct PadWorkgroupMemoryPass
    : public ptxforge::impl::PadWorkgroupMemoryBase<PadWorkgroupMemoryPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    int64_t ab = accessBytes, al = alignBytes, mp = maxPad; // Option<T> -> plain ints
    patterns.add<PadWorkgroupAlloc>(&getContext(), ab, al, mp);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ptxforge::createPadWorkgroupMemoryPass() {
  return std::make_unique<PadWorkgroupMemoryPass>();
}
