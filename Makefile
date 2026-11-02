.PHONY: all test bank demo cuda parity studio clean compiler compiler-test ptx
PY ?= python3
CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra
INC := -Icore/include -Icompiler/include -DPTXFORGE_ROOT_DIR='"$(CURDIR)"'
CUDA_HOME ?= /usr/local/cuda
LLVM_BIN ?= /usr/lib/llvm-19/bin
MLIR_DIR ?= /usr/lib/llvm-19/lib/cmake/mlir
OPT := $(CURDIR)/build-mlir/ptxforge-opt

all: build/ptxforge-tune

build/ptxforge-tune: core/src/main.cpp $(wildcard core/include/ptxforge/*.hpp) compiler/include/PtxForge/BankModel.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) core/src/main.cpp -o $@

build/ptxforge-tests: core/tests/test_core.cpp $(wildcard core/include/ptxforge/*.hpp)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) core/tests/test_core.cpp -o $@

bank:
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Icompiler/include tools/bank_sim.cpp -o build/bank_sim
	./build/bank_sim

parity: build/ptxforge-tune   ## C++ cost model vs the Python reference on every config (needs numpy-free python3 only)
	$(PY) tests/parity_check.py build/ptxforge-tune

test: bank build/ptxforge-tests parity
	./build/ptxforge-tests
	cd reference/py_tuner && $(PY) -m unittest discover -s tests -t . 

demo: build/ptxforge-tune  ## SIMULATED end-to-end run (NOT real numbers) -> studio/demo_simulated.html
	mkdir -p runs
	./build/ptxforge-tune tune --runner sim --gpu t4 --advisor kg --budget 24 --out runs/sim.json
	$(PY) scripts/make_studio.py runs/sim.json studio/demo_simulated.html

cuda:  ## real GPU runner (needs CUDA toolkit + a GPU). Built and run on a Colab T4.
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) -DPTXFORGE_WITH_CUDA -DPTXFORGE_WITH_CUBLAS \
	    core/src/main.cpp core/src/cuda_runner.cpp -o build/ptxforge-tune-cuda \
	    -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda -lcublas -lcudart

compiler:  ## ptxforge-opt against a prebuilt MLIR 19 (Ubuntu 24.04: apt install libmlir-19-dev mlir-19-tools llvm-19-dev clang-19)
	cmake -S compiler -B build-mlir -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=$(MLIR_DIR) \
	    -DCMAKE_C_COMPILER=$(LLVM_BIN)/clang -DCMAKE_CXX_COMPILER=$(LLVM_BIN)/clang++
	cmake --build build-mlir

ptx: compiler build/ptxforge-tune  ## lower the 2048^3 sample to PTX for a T4 -> build/ir/06_kernel.ptx
	rm -rf build/ir
	PTXFORGE_OPT=$(OPT) PTXFORGE_TUNE=$(CURDIR)/build/ptxforge-tune bash scripts/lower_to_ptx.sh compiler/samples/gemm_bias_gelu.mlir build/ir sm_75

compiler-test: ptx  ## FileCheck tests, CPU numerical check of the lowering, and the committed PTX is up to date
	cd build-mlir && PTXFORGE_TOOLS_DIR=$(CURDIR)/build-mlir LLVM_BIN_DIR=$(LLVM_BIN) lit -v $(CURDIR)/compiler/test
	$(PY) tests/check_lowering_cpu.py --opt $(OPT) --n 256 --tile 64,64,16,4,4
	$(PY) tests/check_lowering_cpu.py --opt $(OPT) --n 256 --tile 128,64,8,8,4
	$(PY) tests/check_lowering_cpu.py --opt $(OPT) --n 192 --tile 64,32,16,4,2
	diff -q build/ir/06_kernel.ptx kernels/generated/gemm_bias_gelu_2048_sm75.ptx \
	  || { echo "kernels/generated/gemm_bias_gelu_2048_sm75.ptx is stale: copy build/ir/06_kernel.ptx over it"; exit 1; }

clean:
	rm -rf build build-mlir runs/*.json
