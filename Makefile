.PHONY: all test bank demo cuda parity studio clean
PY ?= python3
CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra
INC := -Icore/include -Icompiler/include -DPTXFORGE_ROOT_DIR='"$(CURDIR)"'
CUDA_HOME ?= /usr/local/cuda

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

cuda:  ## real GPU runner (needs CUDA toolkit + a GPU). Never built in the authoring sandbox.
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) -DPTXFORGE_WITH_CUDA -DPTXFORGE_WITH_CUBLAS \
	    core/src/main.cpp core/src/cuda_runner.cpp -o build/ptxforge-tune-cuda \
	    -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda -lcublas -lcudart

clean:
	rm -rf build runs/*.json
