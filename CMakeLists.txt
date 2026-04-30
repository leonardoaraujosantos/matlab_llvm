cmake_minimum_required(VERSION 3.20)
project(matlab_llvm C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

include_directories(${CMAKE_SOURCE_DIR}/include)

# --- LLVM / MLIR ------------------------------------------------------------
# Enabled by default; set MATLAB_LLVM_WITH_MLIR=OFF to skip (useful if brew's
# llvm isn't installed or for CI machines that only build the frontend stack).
option(MATLAB_LLVM_WITH_MLIR "Enable MLIR-based lowering stack" ON)

# --- Coverage ---------------------------------------------------------------
# Off by default; turn on with -DMATLAB_LLVM_COVERAGE=ON to instrument the
# matlabc binary (and therefore runtime/matlab_runtime.c) with Clang's
# source-based coverage. Drives the runtime-coverage baseline used by
# docs/port_runtime_2_cpp.md. See scripts/runtime_coverage.sh.
option(MATLAB_LLVM_COVERAGE "Build matlabc with Clang source-based coverage" OFF)
if(MATLAB_LLVM_COVERAGE)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang"
     OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "MATLAB_LLVM_COVERAGE requires Clang for both C and C++ "
      "(found C=${CMAKE_C_COMPILER_ID}, CXX=${CMAKE_CXX_COMPILER_ID}).")
  endif()
  set(MATLAB_LLVM_COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping)
  message(STATUS "Coverage build enabled "
                 "(flags: ${MATLAB_LLVM_COVERAGE_FLAGS})")
endif()

if(MATLAB_LLVM_WITH_MLIR)
  # Prefer Homebrew's keg-only llvm on macOS if the user hasn't set LLVM_DIR.
  if(APPLE AND NOT DEFINED LLVM_DIR AND EXISTS "/opt/homebrew/opt/llvm")
    set(LLVM_DIR "/opt/homebrew/opt/llvm/lib/cmake/llvm" CACHE PATH "")
    set(MLIR_DIR "/opt/homebrew/opt/llvm/lib/cmake/mlir" CACHE PATH "")
  endif()

  find_package(LLVM CONFIG REQUIRED)
  find_package(MLIR CONFIG REQUIRED)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  # Match LLVM's RTTI / exception flags to avoid mismatched ABI on static link.
  if(NOT LLVM_ENABLE_RTTI)
    add_compile_options(-fno-rtti)
  endif()
endif()

add_library(MatlabBasic STATIC
  lib/Basic/SourceManager.cpp
  lib/Basic/Diagnostic.cpp
)

add_library(MatlabLex STATIC
  lib/Lex/Lexer.cpp
)
target_link_libraries(MatlabLex PUBLIC MatlabBasic)

add_library(MatlabAST STATIC
  lib/AST/AST.cpp
  lib/AST/ASTDumper.cpp
  lib/AST/Formatter.cpp
)
target_link_libraries(MatlabAST PUBLIC MatlabBasic)

add_library(MatlabParse STATIC
  lib/Parse/Parser.cpp
)
target_link_libraries(MatlabParse PUBLIC MatlabLex MatlabAST)

add_library(MatlabFlowchart STATIC
  lib/Flowchart/Loader.cpp
  lib/Flowchart/GraphToAST.cpp
  lib/Flowchart/ASTToGraph.cpp
)
target_link_libraries(MatlabFlowchart PUBLIC MatlabParse MatlabBasic)

add_library(MatlabSema STATIC
  lib/Sema/Scope.cpp
  lib/Sema/Type.cpp
  lib/Sema/Resolver.cpp
  lib/Sema/TypeInference.cpp
  lib/Sema/SemaDumper.cpp
)
target_link_libraries(MatlabSema PUBLIC MatlabAST MatlabBasic)

add_library(MatlabMIR STATIC
  lib/MIR/MIR.cpp
  lib/MIR/Builder.cpp
  lib/MIR/Printer.cpp
  lib/MIR/Lowering.cpp
)
target_link_libraries(MatlabMIR PUBLIC MatlabSema MatlabAST MatlabBasic)

if(MATLAB_LLVM_WITH_MLIR)
  add_library(MatlabMLIR STATIC
    lib/MLIR/Context.cpp
    lib/MLIR/TypeMapper.cpp
    lib/MLIR/Lowering.cpp
    lib/MLIR/Dialect/MatlabDialect.cpp
    lib/MLIR/Passes/SlotPromotion.cpp
    lib/MLIR/Passes/LowerScalarsToArith.cpp
    lib/MLIR/Passes/LowerFixedPoint.cpp
    lib/MLIR/Passes/LowerIO.cpp
    lib/MLIR/Passes/LowerParfor.cpp
    lib/MLIR/Passes/LowerSeqLoops.cpp
    lib/MLIR/Passes/LowerAnonCalls.cpp
    lib/MLIR/Passes/LowerUserCalls.cpp
    lib/MLIR/Passes/LowerScalarSlots.cpp
    lib/MLIR/Passes/LowerTensorOps.cpp
    lib/MLIR/Passes/IfStoreToSelect.cpp
    lib/MLIR/Passes/Mem2RegLite.cpp
    lib/MLIR/Passes/LowerToLLVMIR.cpp
    lib/MLIR/Passes/EmitC.cpp
    lib/MLIR/Passes/EmitPython.cpp
    lib/MLIR/Passes/EmitTypeScript.cpp
    lib/MLIR/Passes/HWForLoop.cpp
    lib/MLIR/Passes/HWStateInfer.cpp
    lib/MLIR/Passes/HWLegalize.cpp
    lib/MLIR/Passes/HWBitWidthInfer.cpp
    lib/MLIR/Passes/EmitSystemVerilog.cpp
    lib/MLIR/Passes/RefineIfConds.cpp
    lib/MLIR/Passes/RefineFuncSigs.cpp
    lib/MLIR/Passes/RefineSlotTypes.cpp
    lib/MLIR/Passes/SplitIsEmptyOr.cpp
    lib/MLIR/Passes/LowerStaticFiArrays.cpp
    lib/MLIR/Passes/LowerPersistentFiArrays.cpp
    lib/MLIR/Passes/HWUnrollFor.cpp
    lib/MLIR/Passes/ScanHWPragmas.cpp
    lib/MLIR/Passes/ApplyPortTypePragmas.cpp
    lib/MLIR/Passes/EmitHardwareReport.cpp
    lib/MLIR/Passes/ConstMulCSD.cpp
    lib/MLIR/Passes/LowerFiSaturate.cpp
  )
  target_link_libraries(MatlabMLIR
    PUBLIC  MatlabSema MatlabAST MatlabBasic
    PRIVATE MLIRIR MLIRFuncDialect MLIRArithDialect MLIRSCFDialect
            MLIRTensorDialect MLIRMemRefDialect MLIRControlFlowDialect
            MLIRLLVMDialect MLIRSupport MLIRParser MLIRAsmParser
            MLIRTransforms MLIRTransformUtils MLIRPass MLIRAnalysis
            MLIRRewrite
            MLIRArithToLLVM MLIRControlFlowToLLVM MLIRFuncToLLVM
            MLIRSCFToControlFlow MLIRReconcileUnrealizedCasts
            MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
            MLIRTargetLLVMIRExport)
  target_compile_definitions(MatlabMLIR PUBLIC MATLAB_LLVM_WITH_MLIR=1)
endif()

add_executable(matlab-lsp tools/matlab-lsp/main.cpp)
# MatlabFlowchart and MatlabSema PUBLIC-link the rest transitively;
# don't list those again to avoid `ld: ignoring duplicate libraries`.
target_link_libraries(matlab-lsp
  PRIVATE MatlabFlowchart MatlabSema)
if(MATLAB_LLVM_WITH_MLIR)
  # The LSP binary only needs llvm::json from LLVMSupport — but since
  # MATLAB_LLVM_WITH_MLIR implies a usable llvm install, link against
  # LLVMSupport directly so we don't pick up the full MLIR stack.
  target_link_libraries(matlab-lsp PRIVATE LLVMSupport)
endif()

add_executable(matlabc tools/matlabc/main.cpp)
# MatlabMIR and MatlabParse PUBLIC-link the rest transitively;
# listing them explicitly here would duplicate them on the link line
# (ld64 warns "ignoring duplicate libraries").
target_link_libraries(matlabc
  PRIVATE MatlabMIR MatlabFlowchart)
if(MATLAB_LLVM_WITH_MLIR)
  target_link_libraries(matlabc PRIVATE MatlabMLIR)
  target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_MLIR=1)
  # Link the runtime into matlabc so the JIT-backed -repl mode can
  # resolve matlab_* and matlab_ws_* symbols from the running process
  # via LLJIT's DynamicLibrarySearchGenerator::GetForCurrentProcess.
  # The runtime now compiles as C++ (Phase 3 of docs/port_runtime_2_cpp.md);
  # its body stays C-shaped and is wrapped in extern "C" so the exported
  # symbols match the names JIT-emitted code looks up.
  # Phase 2 split: REPL/DAP debug machinery lives in runtime_debug.cpp;
  # Phase 2.5 split: complex matrix family + FFT live in
  # runtime_complex.cpp. The numerical real-side core stays in
  # matlab_runtime.cpp. All three share private layouts via
  # runtime/runtime_internal.h.
  target_sources(matlabc PRIVATE
    runtime/matlab_runtime.cpp
    runtime/runtime_debug.cpp
    runtime/runtime_complex.cpp)
  # Export the matlab_* symbols so the JIT's dynamic lookup finds them
  # (by default executables hide symbols not referenced internally).
  if(APPLE)
    target_link_options(matlabc PRIVATE -rdynamic)
  else()
    target_link_options(matlabc PRIVATE -Wl,--export-dynamic)
  endif()
  # Pull in MLIR's ExecutionEngine + the LLVM JIT libraries it needs.
  target_link_libraries(matlabc PRIVATE
    MLIRExecutionEngine
    LLVMOrcJIT LLVMExecutionEngine)
endif()

# Instrument matlabc (which absorbs runtime/matlab_runtime.c via
# target_sources above) for Clang source-based coverage.
if(MATLAB_LLVM_COVERAGE)
  target_compile_options(matlabc PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
  target_link_options(matlabc PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
endif()

enable_testing()

# --- Runtime unit tests -----------------------------------------------------
# Direct C tests for runtime/matlab_runtime.c — no JIT, no MATLAB
# frontend. Each test_*.c links the runtime TU directly. See
# docs/port_runtime_2_cpp.md (Phase 1).
foreach(_rt_test IN ITEMS linalg shape rng complex reduce signal stats image more fft)
  set(_rt_target "runtime-test-${_rt_test}")
  add_executable(${_rt_target}
    test/Runtime/test_${_rt_test}.c
    runtime/matlab_runtime.cpp
    runtime/runtime_debug.cpp
    runtime/runtime_complex.cpp)
  target_include_directories(${_rt_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/test/Runtime)
  target_link_libraries(${_rt_target} PRIVATE m)
  if(MATLAB_LLVM_COVERAGE)
    target_compile_options(${_rt_target} PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
    target_link_options(${_rt_target}    PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
  endif()
  add_test(NAME runtime-tests-${_rt_test}
           COMMAND $<TARGET_FILE:${_rt_target}>)
endforeach()

add_test(
  NAME frontend-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/run_tests.sh $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
)
add_test(
  NAME flowchart-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/run_tests.sh $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart
)
add_test(
  NAME flowchart-emit-matlab-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitMatlab/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitMatlab
)
add_test(
  NAME flowchart-cross-backend-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/CrossBackend/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/CrossBackend
)
add_test(
  NAME flowchart-lsp-tests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Flowchart/Lsp/run_tests.py
          $<TARGET_FILE:matlab-lsp>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/Lsp
)
add_test(
  NAME flowchart-dap-tests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Flowchart/Dap/run_tests.py
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/Dap
)
add_test(
  NAME flowchart-emit-mflow-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitMflow/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitMflow
)
if(MATLAB_LLVM_WITH_MLIR)
  add_test(
    NAME run-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  # -emit-c path: run every .m through cc and compare output.
  add_test(
    NAME run-tests-emit-c
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests_emitc.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  set_tests_properties(run-tests-emit-c PROPERTIES
    ENVIRONMENT "MODE=c")
  # -emit-cpp path: same but with c++.
  add_test(
    NAME run-tests-emit-cpp
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests_emitc.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  set_tests_properties(run-tests-emit-cpp PROPERTIES
    ENVIRONMENT "MODE=cpp")
  # Strict-warnings lanes: catch quality regressions in the emitted
  # source (type confusion, implicit decls, sign mismatches, ...) that
  # would otherwise slip past the default -w suite.
  add_test(
    NAME run-tests-emit-c-strict
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests_emitc.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  set_tests_properties(run-tests-emit-c-strict PROPERTIES
    ENVIRONMENT "MODE=c;STRICT=1")
  add_test(
    NAME run-tests-emit-cpp-strict
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests_emitc.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  set_tests_properties(run-tests-emit-cpp-strict PROPERTIES
    ENVIRONMENT "MODE=cpp;STRICT=1")
  # -emit-python path: run every .m through python3 and compare output.
  add_test(
    NAME run-tests-emit-python
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests_emitpython.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  # -emit-typescript path: run every .m through tsx/bun/ts-node and
  # compare output. Skips silently if no TS runner is on PATH.
  add_test(
    NAME run-tests-emit-typescript
    COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_tests_emitts.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
  )
  # Fail-fast contract: unsupported MLIR ops must produce a clear error
  # and a non-zero exit rather than silently generating broken C.
  add_test(
    NAME emitc-fail-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitCFail/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitCFail
  )
  # Shape goldens: diff the exact emitted C / C++ source against a per-
  # input .c.expected / .cpp.expected. Catches cosmetic regressions that
  # the behavioural run-tests lanes miss (they only diff stdout).
  add_test(
    NAME emitc-shape-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitC/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitC
  )
  # Debug-hook injection: drives `matlabc -emit-mlir -g` over fixtures
  # that exercise blank lines, comments, control-flow blocks, and
  # function bodies; asserts each matlab_dbg_hook call lands on a real,
  # executable source line. Guards both the per-fixture line lists and
  # the "stepping never lands on a blank line" invariant.
  add_test(
    NAME debug-hook-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/Debug/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Debug
  )
  # End-to-end DAP scenarios: spawns matlabc -dap, drives the protocol
  # with a Python harness, asserts breakpoint hits, step-vs-breakpoint
  # stop reasons, stackTrace / scopes / variables, setVariable,
  # conditional breakpoints, and log points.
  add_test(
    NAME debug-dap-tests
    COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Debug/run_dap_tests.py
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Debug
  )
  # JIT/static parity: every shape that the static -emit-* paths
  # accept must also compile cleanly through `matlabc -dap` (which
  # JITs via mlir::ExecutionEngine + DebugMode mirror calls). Covers
  # the matrix of user-function shapes (logical return, recursive,
  # void, multi-return, ...) and end-to-end DAP runs of every program
  # in examples/mflow/. Guards the regression where the JIT silently
  # bailed on programs whose user functions returned/mirrored an i1.
  add_test(
    NAME debug-dap-jit-userfn-tests
    COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Debug/run_jit_userfn_tests.py
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Debug
  )
  # SystemVerilog emission shape test (Phase 1 scope, ASIC target).
  # Diffs the emitted .sv against per-input goldens and — when
  # Verilator is on PATH — runs `verilator --lint-only -Wall` on each
  # one to catch inferred-latch hazards / missing widths that the
  # golden alone would miss. See docs/emit_systemverilog.md.
  add_test(
    NAME emit-sv-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitSV/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitSV
  )
  # Synthesizability gate — for each fixture, asserts that
  # `-emit-systemverilog` rejects with a matching diagnostic and that
  # `-check-synthesizable` does the same without writing stdout.
  add_test(
    NAME emit-sv-fail-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitSVFail/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitSVFail
  )

  # fi-spec ↔ SV declaration regression suite. Each fixture
  # declares ports with a known `fi(_, signed|unsigned, W, F)` spec
  # via either a `% hdl: port(...)` pragma or a typed driver call,
  # and asserts the emitted SV port-list / register declarations
  # carry the matching signedness + bit width. Substring-based to
  # stay focused on the property under test.
  add_test(
    NAME emit-sv-ports-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitSVPorts/run_tests.sh $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitSVPorts
  )

  # DWARF emission shape test: asserts that `matlabc -emit-llvm -g`
  # produces an LLVM IR module with a !DICompileUnit / !DIFile /
  # !DISubprogram / !DILocation graph (line tables only) and that
  # `-emit-llvm` without -g produces none of that. The downstream
  # lldb/gdb stepping path is exercised by hand-clang-compiling the IR
  # — covered separately in docs/debug.md, not in CTest, since
  # entitlements for runtime attach vary by host.
  add_test(
    NAME debug-dwarf-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/Debug/run_dwarf_tests.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Debug
  )
  # CocoTB co-simulation lane. Drives the SV DUT and the emitted
  # Python reference in lockstep through every supported
  # examples/hdl/*.m module and asserts bit-exact equivalence.
  # Skip-if-missing on verilator / cocotb absence (rc=77).
  add_test(
    NAME cocotb-tests
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitCocoTB/run_tests.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitCocoTB
  )
  set_tests_properties(cocotb-tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
