import os
import lit.formats
config.name = "ptxforge"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.getcwd(), "test-out")
tools = os.environ.get("PTXFORGE_TOOLS_DIR", os.path.join(os.getcwd()))
llvm_bin = os.environ.get("LLVM_BIN_DIR", "")
config.environment["PATH"] = os.pathsep.join([tools, llvm_bin, os.environ.get("PATH", "")])
