// WorkgroupMemory.cpp — passes `ptxforge-normalize-workgroup-allocs` and `ptxforge-workgroup-sync`.
//
// The schedule stages each K step's A and B tiles in shared memory with `transform.structured.promote`
// (memory_space = workgroup). Promotion leaves three things that the NVPTX lowering cannot use as is:
//
//   1. Each buffer is a byte allocation viewed as the tile, plus a dealloc:
//        %raw = memref.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
//        %a   = memref.view %raw[%c0][] : ... to memref<64x16xf32, #gpu.address_space<workgroup>>
//        memref.dealloc %raw
//      `ptxforge-normalize-workgroup-allocs` folds that into `memref.alloc() : memref<64x16xf32, ...>`
//      and drops the dealloc, so `ptxforge-pad-workgroup-memory` can pad the row length.
//
//   2. The allocations sit inside the K loop, inside the kernel. NVVM has no dynamic shared-memory
//      allocation; shared memory is a static attribution of the kernel. `ptxforge-workgroup-sync` runs
//      on outlined kernels (gpu.func) and turns every memref.alloc in one into a workgroup attribution.
//      That is exactly the allocation's meaning there: the schedule placed it in the per-block code
//      (outside any per-thread loop), so it is one buffer per block. Reusing the buffer across K steps
//      is safe because of the barriers below. Each attribution asks for 16-byte alignment
//      (`llvm.align`): the threads read the tiles with 128-bit ld.shared.v4, and without it the PTX
//      declares the arrays `.align 4`.
//
//   3. No barriers. The schedule maps loops to threads with sync_after_distribute = false, because a
//      gpu.barrier in the K loop stops hoist_redundant_vector_transfers from keeping each thread's
//      accumulator in registers. `ptxforge-workgroup-sync` then adds exactly the two a K step needs, in
//      every scf.for in a launch whose body writes shared memory:
//        - before the first op that reads shared memory (the tiles are complete before anyone reads),
//        - at the end of the body (nobody overwrites a tile another thread is still reading).
//      It refuses a loop body whose shared-memory reads come before its last shared-memory write.
#include "PtxForge/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace ptxforge {
#define GEN_PASS_DEF_NORMALIZEWORKGROUPALLOCS
#define GEN_PASS_DEF_WORKGROUPSYNC
#include "PtxForge/Passes.h.inc"
} // namespace ptxforge

using namespace mlir;

namespace {

bool isWorkgroup(Type t) {
  auto ty = dyn_cast<MemRefType>(t);
  if (!ty) return false;
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(ty.getMemorySpace());
  return space && space.getValue() == gpu::AddressSpace::Workgroup;
}

// ------------------------------------------------------------------ normalize-workgroup-allocs

struct NormalizeWorkgroupAllocsPass
    : public ptxforge::impl::NormalizeWorkgroupAllocsBase<NormalizeWorkgroupAllocsPass> {
  void runOnOperation() override {
    SmallVector<memref::AllocOp> raws;
    getOperation()->walk([&](memref::AllocOp a) {
      auto ty = a.getType();
      if (isWorkgroup(ty) && ty.getRank() == 1 && ty.hasStaticShape() && ty.getElementType().isInteger(8))
        raws.push_back(a);
    });
    for (memref::AllocOp raw : raws) {
      memref::ViewOp view;
      SmallVector<memref::DeallocOp> deallocs;
      bool ok = true;
      for (Operation *user : raw->getUsers()) {
        if (auto v = dyn_cast<memref::ViewOp>(user); v && !view) view = v;
        else if (auto d = dyn_cast<memref::DeallocOp>(user)) deallocs.push_back(d);
        else ok = false;
      }
      // Only the plain case: one view at byte offset 0 with a static shape that fills the buffer.
      if (!ok || !view || !view.getSizes().empty() || !matchPattern(view.getByteShift(), m_Zero()))
        continue;
      MemRefType tile = view.getType();
      if (!tile.hasStaticShape() || !tile.getLayout().isIdentity() || !tile.getElementType().isIntOrFloat())
        continue;
      int64_t bytes = tile.getNumElements() * tile.getElementTypeBitWidth() / 8;
      if (bytes != raw.getType().getNumElements())
        continue;
      OpBuilder b(raw);
      auto typed = b.create<memref::AllocOp>(raw.getLoc(), tile);
      if (IntegerAttr align = raw.getAlignmentAttr()) typed.setAlignmentAttr(align);
      view.replaceAllUsesWith(typed.getResult());
      view.erase();
      for (memref::DeallocOp d : deallocs) d.erase();
      raw.erase();
    }
  }
};

// ------------------------------------------------------------------ workgroup-sync

// Does `op` (or anything nested in it) read or write shared memory?
void sharedEffects(Operation *op, bool &reads, bool &writes) {
  op->walk([&](Operation *inner) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(inner);
    if (!iface) return;
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (auto &e : effects) {
      Value v = e.getValue();
      if (!v || !isWorkgroup(v.getType())) continue;
      if (isa<MemoryEffects::Read>(e.getEffect())) reads = true;
      if (isa<MemoryEffects::Write>(e.getEffect())) writes = true;
    }
  });
}

struct WorkgroupSyncPass : public ptxforge::impl::WorkgroupSyncBase<WorkgroupSyncPass> {
  void runOnOperation() override {
    bool failed = false;
    getOperation()->walk([&](gpu::GPUFuncOp kernel) {
      if (!kernel.isKernel()) return;
      // 1. Allocations become workgroup attributions of the kernel.
      SmallVector<memref::AllocOp> allocs;
      kernel.getBody().walk([&](memref::AllocOp a) { allocs.push_back(a); });
      for (memref::AllocOp a : allocs) {
        MemRefType ty = a.getType();
        if (!ty.hasStaticShape() || !a.getDynamicSizes().empty()) {
          a.emitError("ptxforge-workgroup-sync: dynamic allocation inside a GPU kernel");
          failed = true;
          continue;
        }
        if (llvm::any_of(a->getUsers(), [](Operation *u) { return isa<memref::DeallocOp>(u); })) {
          a.emitError("ptxforge-workgroup-sync: run ptxforge-normalize-workgroup-allocs first (dealloc found)");
          failed = true;
          continue;
        }
        auto wgTy = MemRefType::get(ty.getShape(), ty.getElementType(), ty.getLayout(),
                                    gpu::AddressSpaceAttr::get(&getContext(), gpu::AddressSpace::Workgroup));
        if (wgTy != ty) {
          a.emitError("ptxforge-workgroup-sync: allocation inside a GPU kernel is not in workgroup memory");
          failed = true;
          continue;
        }
        int64_t align = std::max<int64_t>(16, a.getAlignment().value_or(0));
        BlockArgument attr = kernel.addWorkgroupAttribution(ty, a.getLoc());
        kernel.setWorkgroupAttributionAttr(kernel.getNumWorkgroupAttributions() - 1, "llvm.align",
                                           IntegerAttr::get(IntegerType::get(&getContext(), 64), align));
        a.getResult().replaceAllUsesWith(attr);
        a.erase();
      }

      // 2. Barriers in every loop whose body writes shared memory.
      kernel.getBody().walk([&](scf::ForOp loop) {
        Block *body = loop.getBody();
        Operation *firstRead = nullptr, *lastWrite = nullptr;
        for (Operation &op : body->without_terminator()) {
          bool r = false, w = false;
          sharedEffects(&op, r, w);
          if (r && !firstRead) firstRead = &op;
          if (w) lastWrite = &op;
        }
        if (!lastWrite) return;
        if (firstRead && firstRead->isBeforeInBlock(lastWrite)) {
          loop.emitError("ptxforge-workgroup-sync: shared memory is read before its last write in this loop body");
          failed = true;
          return;
        }
        OpBuilder b(loop.getContext());
        if (firstRead) {
          b.setInsertionPoint(firstRead);
          b.create<gpu::BarrierOp>(firstRead->getLoc());
        }
        b.setInsertionPoint(body->getTerminator());
        b.create<gpu::BarrierOp>(loop.getLoc());
      });
    });
    if (failed) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ptxforge::createNormalizeWorkgroupAllocsPass() {
  return std::make_unique<NormalizeWorkgroupAllocsPass>();
}
std::unique_ptr<Pass> ptxforge::createWorkgroupSyncPass() {
  return std::make_unique<WorkgroupSyncPass>();
}
