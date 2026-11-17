// OrderKernelArgs.cpp — pass `ptxforge-order-kernel-args`.
//
// `gpu-kernel-outlining` gives the outlined kernel one parameter per value captured from the host
// function, in the order the kernel body first *uses* them. For the fused GEMM + bias + GELU kernel
// that is (C, A, B, bias): the output's tile is addressed before the inputs are read. The PTX entry
// point therefore takes its pointers in an order that depends on the schedule, not on the source
// signature `@gemm_bias_gelu(A, B, bias, C)`, and a caller that follows the source signature (as
// scripts/run_ptx_on_gpu.py does) passes the wrong buffer for every parameter.
//
// This pass makes the kernel ABI follow the source: for every gpu.launch_func, kernel operands that are
// arguments of the enclosing host function are sorted by that function's argument number; any other
// operands keep their relative order after them. The gpu.func signature and the launch are permuted
// together, so behaviour is unchanged. A kernel launched from several places is left alone unless every
// launch asks for the same permutation.
#include "PtxForge/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <numeric>

namespace ptxforge {
#define GEN_PASS_DEF_ORDERKERNELARGS
#include "PtxForge/Passes.h.inc"
} // namespace ptxforge

using namespace mlir;

namespace {

// newOrder[k] = index of the old kernel operand that becomes operand k.
SmallVector<unsigned> desiredOrder(gpu::LaunchFuncOp launch) {
  auto host = launch->getParentOfType<FunctionOpInterface>();
  Block *entry = host && !host.isExternal() ? &host.getFunctionBody().front() : nullptr;
  ValueRange operands = launch.getKernelOperands();
  auto key = [&](unsigned i) -> std::pair<unsigned, unsigned> {
    if (auto arg = dyn_cast<BlockArgument>(operands[i]); arg && entry && arg.getOwner() == entry)
      return {0, arg.getArgNumber()};
    return {1, i};
  };
  SmallVector<unsigned> order(operands.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return key(a) < key(b); });
  return order;
}

bool isIdentity(ArrayRef<unsigned> order) {
  for (unsigned i = 0; i < order.size(); ++i)
    if (order[i] != i) return false;
  return true;
}

void permuteKernel(gpu::GPUFuncOp kernel, ArrayRef<unsigned> order) {
  const unsigned n = kernel.getNumArguments();
  // Append the arguments again in the new order, move every use over, then drop the originals.
  // Workgroup/private attributions sit after the arguments in the entry block and are untouched.
  for (unsigned k = 0; k < n; ++k) {
    const unsigned old = order[k];
    kernel.insertArgument(n + k, kernel.getArgumentTypes()[old], kernel.getArgAttrDict(old),
                          kernel.getArgument(old).getLoc());
  }
  for (unsigned k = 0; k < n; ++k)
    kernel.getArgument(order[k]).replaceAllUsesWith(kernel.getArgument(n + k));
  llvm::BitVector drop(2 * n);
  drop.set(0, n);
  kernel.eraseArguments(drop);
}

struct OrderKernelArgsPass : public ptxforge::impl::OrderKernelArgsBase<OrderKernelArgsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbols;

    // Group launches by kernel and agree on one permutation per kernel.
    llvm::DenseMap<Operation *, SmallVector<gpu::LaunchFuncOp>> launches;
    SmallVector<Operation *> kernels; // deterministic order
    module.walk([&](gpu::LaunchFuncOp launch) {
      auto kernel = symbols.lookupNearestSymbolFrom<gpu::GPUFuncOp>(launch, launch.getKernel());
      if (!kernel) return;
      auto [it, inserted] = launches.try_emplace(kernel.getOperation());
      if (inserted) kernels.push_back(kernel.getOperation());
      it->second.push_back(launch);
    });

    for (Operation *op : kernels) {
      auto kernel = cast<gpu::GPUFuncOp>(op);
      auto &uses = launches[op];
      SmallVector<unsigned> order = desiredOrder(uses.front());
      if (order.size() != kernel.getNumArguments()) continue; // not a plain outlined kernel
      bool agree = llvm::all_of(uses, [&](gpu::LaunchFuncOp l) { return desiredOrder(l) == order; });
      if (!agree) {
        kernel.emitRemark("ptxforge-order-kernel-args: launches disagree on argument order; left unchanged");
        continue;
      }
      if (isIdentity(order)) continue;

      permuteKernel(kernel, order);
      for (gpu::LaunchFuncOp launch : uses) {
        SmallVector<Value> operands(launch.getKernelOperands());
        SmallVector<Value> reordered;
        for (unsigned old : order) reordered.push_back(operands[old]);
        launch.getKernelOperandsMutable().assign(reordered);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> ptxforge::createOrderKernelArgsPass() {
  return std::make_unique<OrderKernelArgsPass>();
}
