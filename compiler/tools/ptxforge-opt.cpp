// ptxforge-opt — mlir-opt with the PTXForge passes registered.
//   ptxforge-opt --ptxforge-fuse-epilogue samples/gemm_bias_gelu.mlir
#include "PtxForge/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  ptxforge::registerPtxForgePasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "PTXForge optimizer driver\n", registry));
}
