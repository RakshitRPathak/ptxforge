# Dev container: CUDA toolkit + prebuilt LLVM/MLIR 19 from apt.llvm.org (no source build).
# docker build -t ptxforge -f docker/Dockerfile .
# docker run --gpus all -it -v $PWD:/work ptxforge          (omit --gpus on a machine with no GPU)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget gnupg lsb-release software-properties-common ca-certificates \
      cmake ninja-build git python3 python3-pip ccache \
 && rm -rf /var/lib/apt/lists/*

# LLVM 19 + MLIR dev packages
RUN wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && chmod +x /tmp/llvm.sh \
 && /tmp/llvm.sh 19 \
 && apt-get install -y --no-install-recommends \
      llvm-19-dev libmlir-19-dev mlir-19-tools clang-19 lld-19 \
 && rm -rf /var/lib/apt/lists/*

ENV MLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
    LLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
    PATH=/usr/lib/llvm-19/bin:$PATH \
    CCACHE_DIR=/work/.ccache

RUN pip3 install --no-cache-dir numpy

WORKDIR /work
# Typical inner loop (incremental pass rebuilds are seconds with ninja + lld + ccache):
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#         -DPTXFORGE_BUILD_COMPILER=ON -DPTXFORGE_BUILD_RUNTIME=ON \
#         -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLVM_USE_LINKER=lld
#   cmake --build build
