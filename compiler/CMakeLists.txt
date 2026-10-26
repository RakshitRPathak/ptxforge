cmake_minimum_required(VERSION 3.20)
project(ptxforge-compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point at a prebuilt MLIR 19, e.g.  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# LLVM is usually built without RTTI; match it.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# --- TableGen: pass declarations/definitions ------------------------------------
set(LLVM_TARGET_DEFINITIONS include/PtxForge/Passes.td)
mlir_tablegen(include/PtxForge/Passes.h.inc -gen-pass-decls -name PtxForge)
add_public_tablegen_target(PtxForgePassesIncGen)

# --- library ----------------------------------------------------------------------
add_mlir_library(PtxForgePasses
  lib/FuseEpilogue.cpp
  lib/PadWorkgroupMemory.cpp

  DEPENDS
  PtxForgePassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransformUtils
)

# --- driver -------------------------------------------------------------------------
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(ptxforge-opt tools/ptxforge-opt.cpp)
llvm_update_compile_flags(ptxforge-opt)
target_link_libraries(ptxforge-opt PRIVATE
  PtxForgePasses
  ${dialect_libs} ${conversion_libs} ${extension_libs}
  MLIROptLib
  MLIRPass
)
mlir_check_all_link_libraries(ptxforge-opt)

# --- lit/FileCheck tests:  cmake --build . --target check-ptxforge  ----------------------------
# Needs `lit` (pip install lit) and FileCheck (ships with LLVM; set -DLLVM_BIN_DIR=/usr/lib/llvm-19/bin).
find_program(LIT_EXE NAMES lit llvm-lit)
if(LIT_EXE)
  add_custom_target(check-ptxforge
    COMMAND ${CMAKE_COMMAND} -E env PTXFORGE_TOOLS_DIR=${CMAKE_CURRENT_BINARY_DIR} LLVM_BIN_DIR=${LLVM_TOOLS_BINARY_DIR}
            ${LIT_EXE} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ptxforge-opt)
endif()
