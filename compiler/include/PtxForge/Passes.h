#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace ptxforge {

std::unique_ptr<mlir::Pass> createFuseEpiloguePass();
std::unique_ptr<mlir::Pass> createPadWorkgroupMemoryPass();

#define GEN_PASS_REGISTRATION
#include "PtxForge/Passes.h.inc"

} // namespace ptxforge
