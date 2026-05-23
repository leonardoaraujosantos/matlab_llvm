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

# --- Symbolic Math Toolbox via SymPP ---------------------------------------
# Off by default; flip on with -DMATLAB_LLVM_WITH_SYM=ON. Pulls in SymPP
# (https://github.com/leonardoaraujosantos/SymPP) plus its GMP + MPFR
# transitive deps. Adds the matlab_sym opaque runtime type and the
# corresponding sym/syms/diff/int/simplify/solve builtins. Point at a
# specific SymPP install with -DSymPP_DIR=/path/to/SymPP/build.
option(MATLAB_LLVM_WITH_SYM "Enable Symbolic Math Toolbox via SymPP" OFF)
if(MATLAB_LLVM_WITH_SYM)
  if(NOT DEFINED SymPP_DIR)
    # Prefer the install tree (ships FindGMP / FindMPFR alongside the
    # config) over the build tree (which doesn't, since those modules
    # only get copied into <prefix>/lib/cmake/SymPP at install time).
    foreach(_h IN ITEMS
        "/tmp/sympp_install/lib/cmake/SymPP"
        "/opt/homebrew/lib/cmake/SymPP"
        "/Users/leonardoaraujo/work/SymPP/build")
      if(EXISTS "${_h}/SymPPConfig.cmake")
        set(SymPP_DIR "${_h}" CACHE PATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
  find_package(SymPP CONFIG REQUIRED)
  message(STATUS "Found SymPP — Symbolic Math Toolbox enabled")
endif()

# --- Plotting (Cairo backend) ----------------------------------------------
# Off by default; turn on with -DMATLAB_LLVM_WITH_PLOT=ON. Adds runtime
# plotting symbols (matlab_plot, matlab_figure, matlab_savefig, ...) backed
# by Cairo's image/svg/pdf surfaces. Headless: no subprocess, no display
# server, no temp files. Cross-platform target: macOS, Linux, iOS.
option(MATLAB_LLVM_WITH_PLOT "Enable matlab_plot runtime (Cairo backend)" OFF)
if(MATLAB_LLVM_WITH_PLOT)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(CAIRO REQUIRED IMPORTED_TARGET cairo cairo-svg cairo-pdf)
  message(STATUS "Found Cairo ${CAIRO_VERSION} — matlab_plot enabled")
  set(MATLAB_PLOT_SOURCES
    runtime/plot/c_api.cpp
    runtime/plot/figure.cpp
    runtime/plot/cairo_render.cpp
    runtime/plot/colormap.cpp
    runtime/plot/contour.cpp)
endif()

# --- Verilog-A lint ---------------------------------------------------------
# Off by default; turn on with -DMATLAB_LLVM_WITH_VA_LINT=ON to register a
# CTest target that runs scripts/va_lint.sh (OpenVAF / ADMS) over every
# example .va emitted from examples/verilog_a/.  The lint script exits 0
# (skip) when neither tool is installed, so the CTest lane is safe to
# enable unconditionally on CI machines.
option(MATLAB_LLVM_WITH_VA_LINT
       "Enable Verilog-A lint CTest lane (OpenVAF / ADMS wrapper)" OFF)
# --- Verilog-A cosim --------------------------------------------------------
# Off by default; turn on with -DMATLAB_LLVM_WITH_VA_COSIM=ON to register a
# CTest target that pipes every example .va through ngspice + OpenVAF
# (preferred) or Xyce + ADMS.  Skips cleanly when neither toolchain is
# installed.  Limited to the 1-in/1-out subset of examples that fit the
# canonical AC-sweep template; composite blocks (AM / I/Q / DAC) need
# per-block testbenches that aren't in scope for this lane.
option(MATLAB_LLVM_WITH_VA_COSIM
       "Enable Verilog-A cosim CTest lane (ngspice+OpenVAF / Xyce+ADMS)" OFF)

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
  lib/Flowchart/SignalFlowLowering.cpp
  lib/Flowchart/SubsystemToMatlab.cpp
  lib/Flowchart/MflowLinkSim.cpp
)
target_link_libraries(MatlabFlowchart PUBLIC MatlabParse MatlabBasic)

add_library(MatlabStateChart STATIC
  lib/StateChart/StateChartIR.cpp
  lib/StateChart/Lowering.cpp
  lib/StateChart/Interpreter.cpp
)
target_link_libraries(MatlabStateChart PUBLIC MatlabFlowchart MatlabBasic)

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
    lib/MLIR/Passes/LowerPlot.cpp
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
    lib/MLIR/Passes/UnifyMixedWidthStores.cpp
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
  PRIVATE MatlabStateChart MatlabSema)
if(MATLAB_LLVM_WITH_MLIR)
  # The LSP binary only needs llvm::json from LLVMSupport — but since
  # MATLAB_LLVM_WITH_MLIR implies a usable llvm install, link against
  # LLVMSupport directly so we don't pick up the full MLIR stack.
  target_link_libraries(matlab-lsp PRIVATE LLVMSupport)
endif()

add_executable(matlabc
  tools/matlabc/main.cpp
  tools/matlabc/MflowLinkJit.cpp)
# Runtime TUs live under runtime/ and runtime/toolbox/<name>/ — they
# all reference shared headers (matlab_runtime.h / runtime_internal.h)
# via unqualified #include, so the include path needs runtime/ on it.
target_include_directories(matlabc PRIVATE ${CMAKE_SOURCE_DIR}/runtime)
# MatlabMIR and MatlabParse PUBLIC-link the rest transitively;
# listing them explicitly here would duplicate them on the link line
# (ld64 warns "ignoring duplicate libraries").
target_link_libraries(matlabc
  PRIVATE MatlabMIR MatlabStateChart)
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
    runtime/runtime_complex.cpp
    runtime/runtime_sparse.cpp
    runtime/toolbox/prop/runtime_prop.cpp
    runtime/toolbox/comm/runtime_comm.cpp
    runtime/toolbox/rf/runtime_rf.cpp
    runtime/toolbox/pde/runtime_pde.cpp
    runtime/toolbox/optim/runtime_optim.cpp
    runtime/toolbox/mpc/runtime_mpc.cpp
    runtime/toolbox/ident/runtime_ident.cpp
    runtime/toolbox/gads/runtime_gads.cpp
    runtime/toolbox/stats/runtime_stats.cpp
    runtime/toolbox/images/runtime_images.cpp
    runtime/toolbox/curvefit/runtime_curvefit.cpp
    runtime/toolbox/wavelet/runtime_wavelet.cpp
    runtime/toolbox/dsp/runtime_dsp.cpp
    runtime/toolbox/stateflow/runtime_mstateflow.cpp)
  if(MATLAB_LLVM_WITH_SYM)
    target_sources(matlabc PRIVATE runtime/toolbox/sym/runtime_sym.cpp)
    target_link_libraries(matlabc PRIVATE SymPP::sympp)
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_SYM=1)
  endif()
  if(MATLAB_LLVM_WITH_PLOT)
    target_sources(matlabc PRIVATE ${MATLAB_PLOT_SOURCES})
    target_link_libraries(matlabc PRIVATE PkgConfig::CAIRO)
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_PLOT=1)
  endif()
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

# --- Sanitizer support for the runtime test lane ----------------------------
# Opt-in via -DMATLAB_LLVM_RUNTIME_ASAN=ON. When set, every
# runtime-test-* executable is compiled with -fsanitize=address,undefined
# so the existing runtime-tests-* ctest entries double as a sanitizer lane.
# Recommended workflow:
#   cmake -S . -B build-asan -DMATLAB_LLVM_RUNTIME_ASAN=ON -DMATLAB_LLVM_WITH_MLIR=OFF
#   cmake --build build-asan
#   ASAN_OPTIONS=abort_on_error=0 UBSAN_OPTIONS=halt_on_error=0 \
#     ctest --test-dir build-asan -R '^runtime-tests-'
# The MLIR frontend is excluded from this knob because it pulls LLVM
# headers that interact badly with ASan instrumentation; the runtime
# tests are pure C-ABI and stand alone.
option(MATLAB_LLVM_RUNTIME_ASAN
       "Build runtime test executables with -fsanitize=address,undefined"
       OFF)
set(_rt_san_flags "")
if(MATLAB_LLVM_RUNTIME_ASAN)
  set(_rt_san_flags
      -fsanitize=address
      -fsanitize=undefined
      -fno-omit-frame-pointer
      -fno-sanitize-recover=undefined)
  message(STATUS "Runtime tests: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
endif()

# --- Runtime unit tests -----------------------------------------------------
# Direct C tests for runtime/matlab_runtime.c — no JIT, no MATLAB
# frontend. Each test_*.c links the runtime TU directly. See
# docs/port_runtime_2_cpp.md (Phase 1).
foreach(_rt_test IN ITEMS linalg shape rng complex reduce signal stats image more fft struct_cell fi elementwise unary fi_arrays int_arrays strings ode pde pde_io comm optim prop rf mstateflow)
  set(_rt_target "runtime-test-${_rt_test}")
  add_executable(${_rt_target}
    test/Runtime/test_${_rt_test}.c
    runtime/matlab_runtime.cpp
    runtime/runtime_debug.cpp
    runtime/runtime_complex.cpp
    runtime/runtime_sparse.cpp
    runtime/toolbox/prop/runtime_prop.cpp
    runtime/toolbox/comm/runtime_comm.cpp
    runtime/toolbox/rf/runtime_rf.cpp
    runtime/toolbox/pde/runtime_pde.cpp
    runtime/toolbox/optim/runtime_optim.cpp
    runtime/toolbox/mpc/runtime_mpc.cpp
    runtime/toolbox/ident/runtime_ident.cpp
    runtime/toolbox/gads/runtime_gads.cpp
    runtime/toolbox/stats/runtime_stats.cpp
    runtime/toolbox/images/runtime_images.cpp
    runtime/toolbox/curvefit/runtime_curvefit.cpp
    runtime/toolbox/wavelet/runtime_wavelet.cpp
    runtime/toolbox/dsp/runtime_dsp.cpp
    runtime/toolbox/stateflow/runtime_mstateflow.cpp)
  target_include_directories(${_rt_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/test/Runtime)
  target_link_libraries(${_rt_target} PRIVATE m)
  if(MATLAB_LLVM_COVERAGE)
    target_compile_options(${_rt_target} PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
    target_link_options(${_rt_target}    PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
  endif()
  if(MATLAB_LLVM_RUNTIME_ASAN)
    target_compile_options(${_rt_target} PRIVATE ${_rt_san_flags})
    target_link_options(${_rt_target}    PRIVATE ${_rt_san_flags})
  endif()
  add_test(NAME runtime-tests-${_rt_test}
           COMMAND $<TARGET_FILE:${_rt_target}>)
  if(MATLAB_LLVM_RUNTIME_ASAN)
    # macOS ASan rejects detect_leaks=1; the other knobs cap the noise
    # on UBSan findings so a single fault doesn't abort the whole test.
    set_tests_properties(runtime-tests-${_rt_test} PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=abort_on_error=0:check_initialization_order=1;UBSAN_OPTIONS=halt_on_error=0:print_stacktrace=1")
  endif()
endforeach()

# --- Enforce no new C-style casts in modernised runtime TUs -----------------
# The five new-runtime modules went through a mechanical static_cast<>
# conversion. Lock that in with -Wold-style-cast as an error on these
# TUs only; the legacy matlab_runtime.cpp + sibling TUs predate the
# convention and would generate thousands of warnings if flagged.
set_source_files_properties(
  runtime/toolbox/comm/runtime_comm.cpp
  runtime/toolbox/optim/runtime_optim.cpp
  runtime/toolbox/mpc/runtime_mpc.cpp
  runtime/toolbox/ident/runtime_ident.cpp
  runtime/toolbox/gads/runtime_gads.cpp
  runtime/toolbox/stats/runtime_stats.cpp
  runtime/toolbox/images/runtime_images.cpp
  runtime/toolbox/curvefit/runtime_curvefit.cpp
  runtime/toolbox/wavelet/runtime_wavelet.cpp
  runtime/toolbox/dsp/runtime_dsp.cpp
  runtime/toolbox/rf/runtime_rf.cpp
  runtime/toolbox/prop/runtime_prop.cpp
  runtime/toolbox/stateflow/runtime_mstateflow.cpp
  PROPERTIES COMPILE_OPTIONS "-Wold-style-cast;-Werror=old-style-cast")

if(MATLAB_LLVM_WITH_PLOT)
  foreach(_plot_test IN ITEMS hello basic)
    set(_plot_target "runtime-plot-${_plot_test}")
    add_executable(${_plot_target}
      test/Runtime/test_plot_${_plot_test}.cpp
      runtime/matlab_runtime.cpp
      runtime/runtime_debug.cpp
      runtime/runtime_complex.cpp
      runtime/runtime_sparse.cpp
      runtime/toolbox/prop/runtime_prop.cpp
      runtime/toolbox/comm/runtime_comm.cpp
      runtime/toolbox/rf/runtime_rf.cpp
      runtime/toolbox/pde/runtime_pde.cpp
      ${MATLAB_PLOT_SOURCES})
    target_include_directories(${_plot_target} PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime)
    target_link_libraries(${_plot_target} PRIVATE m PkgConfig::CAIRO)
    add_test(NAME runtime-plot-${_plot_test}
             COMMAND $<TARGET_FILE:${_plot_target}>)
  endforeach()
endif()

add_test(
  NAME frontend-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/run_tests.sh $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
)
# REPL smoke tests — headline coverage is cross-turn user-function
# persistence (a `function ... end` defined in REPL turn N can be
# called from turn N+1; `loadStateChart` chart_tick functions work
# the same way). Closes the matlabc cross-unit JIT gap that affected
# any interactive function-then-call workflow.
add_test(
  NAME repl-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Repl/run_tests.sh $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Repl
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
# mflowLink Tier-B smoke lane — `matlabc -simulate --dry-run` lowers a
# signal-flow .mflow to the MflowLinkModel IR and prints the sorted
# execution order; the Errors/ corpus checks rejected algebraic loops.
add_test(
  NAME flowchart-simulate-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/Simulate/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/Simulate
)
# mflowLink Tier-C smoke lane — actually run the in-process simulation
# interpreter (lib/Flowchart/MflowLinkSim.cpp) on the headline demos
# and check analytic invariants on the CSV output.
add_test(
  NAME flowchart-simulate-run-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateRun/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateRun
)
# mflowLink Tier-D smoke lane — drive `matlabc -simulate --sim-dap`
# over stdio with a scripted DAP conversation and check that the
# simulator pauses at entry, steps forward / backward, and resumes
# to stopTime. Covers the snapshot ring (settings.snapshot.depth).
add_test(
  NAME flowchart-simulate-dap-tests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap/run_tests.py
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap
)
# mflowLink Tier-E smoke lane — block-level stepping. Walks the topo
# order one block at a time, then commits a major step on cursor wrap.
add_test(
  NAME flowchart-simulate-dap-block-tests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap/run_block_step.py
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap
)
# mflowLink Tier-F smoke lane — time + signal breakpoints. Arms each
# kind, verifies `stopped { reason: "breakpoint" }`, and checks that
# breakpoints are sticky until `restart` re-arms them.
add_test(
  NAME flowchart-simulate-dap-breakpoint-tests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap/run_breakpoints.py
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap
)
# mStateflow DAP smoke lane — drive a chart `.mflow` over DAP, inject
# locals + events, and verify the chart transitions through the
# expected state trajectory. Exercises the temporalCount / duration
# interpreter paths end-to-end (no other test fires these operators
# because the default -simulate driver doesn't inject inputs).
add_test(
  NAME flowchart-simulate-dap-chart-tests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap/StateChart/run_tests.py
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/SimulateDap/StateChart
)
# mStateflow cocotb SIL lane — feed the three integer-typed charts
# through `matlabc -emit-cocotb`, check the harness directory is
# structurally complete (SV + ref.py + testbench + Makefile +
# helpers), parse the generated Python, and confirm the chart's
# tick function is exposed. Runs `make sim` end-to-end when
# verilator + cocotb are both on PATH.
add_test(
  NAME flowchart-emit-chart-cocotb
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_chart_cocotb_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
)
# mflowLink §17.5 #6 — cross-dialect composition. Compile a .m
# that calls `mflowlink_run("model.mflow")` via runtime/
# runtime_mflowlink_call.cpp and confirm the round-trip works.
add_test(
  NAME flowchart-cross-dialect-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/CrossDialect/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/CrossDialect
)
# mflowLink Tier-G smoke lane — emit + compile + run + diff. Each
# example .mflow gets emitted to a standalone C++ file via
# `-emit-mflowlink-cpp`, built with `runtime/scripts/build_mflowlink.sh`,
# and its CSV output is diffed byte-for-byte against the in-process
# `matlabc -simulate` interpreter on the same input.
add_test(
  NAME flowchart-emit-mflowlink-cpp-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitMflowLinkCpp/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitMflowLinkCpp
)
# Embedded Coder Tier-1 smoke lane — `matlabc -emit-python` /
# `-emit-cpp` --subsystem <name>` against every fixture under
# `examples/mflowlink/coder/`. The Python side imports the emitted
# module and tests the subsystem entry against analytically-known
# values; the C++ side compiles via clang++ + runs + diffs the
# printf'd output. See `docs/embedded_coder_roadmap.md` Tier 1–2.
add_test(
  NAME flowchart-emit-subsystem-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
)
# Tier-7 whole-diagram emit lane — `matlabc -emit-python <model.mflow>`
# (no `--subsystem`) routes through `DiagramToMatlab` to produce a
# top-level `simulate()` running the diagram's time loop. The tests
# import the emitted module, call `simulate()`, and check the
# per-sink log columns against analytic references.
add_test(
  NAME flowchart-emit-diagram-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_diagram_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
)
# Opt-in Verilog-A lint lane (Tier-10 of docs/verilog_a_plan.md).
# Compiles + runs every example under examples/verilog_a/ to emit
# its .va, then pipes the result into scripts/va_lint.sh
# (OpenVAF / ADMS wrapper).  Skips cleanly when neither linter is
# on PATH; safe to enable on CI machines that don't have OpenVAF.
if(MATLAB_LLVM_WITH_VA_LINT)
  add_test(
    NAME run-emit-va-admslint
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitVA/run_lint.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitVA
  )
endif()
if(MATLAB_LLVM_WITH_VA_COSIM)
  add_test(
    NAME run-emit-va-cosim
    COMMAND ${CMAKE_SOURCE_DIR}/test/EmitVA/run_cosim.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/EmitVA
  )
endif()
# Tier-5h — Embedded Coder Verilator lint lane.  Opt-in CMake flag
# (off by default).  When on, emits each examples/mflowlink/coder/
# SV-capable subsystem through `-emit-sv` and runs the result
# through `verilator --lint-only`. Skips cleanly if verilator
# isn't on PATH (the lint script exits 0 with a warning).  Catches
# regressions the structural smoke tests can't — e.g. timing
# loops, undriven nets, lint-class errors that the SV emitter
# produces but yosys synth tolerates.
option(MATLAB_LLVM_WITH_EMIT_SUBSYSTEM_SV_COSIM
       "Enable Verilator lint CTest lane for Embedded Coder subsystems" OFF)
if(MATLAB_LLVM_WITH_EMIT_SUBSYSTEM_SV_COSIM)
  add_test(
    NAME flowchart-emit-subsystem-sv-verilator
    COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_verilator.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
  )
  # Tier-5j — behavioural cosim: compile each SV-capable fixture
  # with Verilator (--cc --exe --build), drive the same stimulus
  # through both the SV binary AND the Python emit's class wrapper,
  # then compare per-tick outputs with tolerance. Catches numerical
  # divergence the lint lane can't.  Curated to the subset of
  # fixtures whose SV emit is currently bit-exact-equivalent to
  # the Python emit (Tier-5j carve-outs in docs/ track the rest).
  add_test(
    NAME flowchart-emit-subsystem-sv-cosim
    COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_verilator_cosim.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
  )
  # Tier-5l — yosys generic synthesis gate. Catches what the lint
  # and cosim lanes can't: latch inference, unsynthesisable
  # constructs, ungrouped state, plus a per-fixture gate-count
  # floor as a regression sentinel. Skips cleanly when yosys
  # isn't on PATH.
  add_test(
    NAME flowchart-emit-subsystem-sv-yosys
    COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_yosys_synth.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
  )
  # Tier-7d — whole-diagram cocotb SIL emit. Verifies
  # `matlabc -emit-cocotb FILE.mflow --dut <block>` writes a full
  # cocotb harness directory (SV + Python reference + testbench)
  # and that the harness imports cleanly. When cocotb + verilator
  # are both on PATH, also runs the testbench end-to-end and asserts
  # zero SIL mismatches. Gated under the same flag as the rest of
  # the SV-cosim lanes — opting in already implies a Verilator-
  # capable host.
  add_test(
    NAME flowchart-emit-diagram-cocotb
    COMMAND ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem/run_diagram_cocotb_tests.sh
            $<TARGET_FILE:matlabc>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Flowchart/EmitSubsystem
  )
endif()
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
  # Run serially under `ctest -j*` — the DAP scenarios each spawn a
  # matlabc subprocess and the client uses bounded 10 s timeouts on
  # request/response handshakes. Under heavy parallel load (e.g. the
  # 7 emit-* lanes simultaneously compiling 387 .m programs each)
  # CPU starvation occasionally pushes individual scenarios past
  # their timeout. Pinning this test to serial costs ~30 s but
  # removes the flake.
  #
  # MATLABC_DAP_MIN_TIMEOUT raises the per-call timeout floor in
  # dap_client.py: several REPL scenarios pass tight 5 s waits tuned for
  # a warm dev machine, but the first JIT-backed `evaluate` (LLVM codegen
  # warmup) can exceed that on a cold/slower CI runner — surfacing as
  # spurious "timeout waiting for event" failures.  A 30 s floor only
  # moves the failure deadline out; passing tests still return promptly.
  set_tests_properties(debug-dap-tests PROPERTIES
    RUN_SERIAL TRUE
    ENVIRONMENT "MATLABC_DAP_MIN_TIMEOUT=30")
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
  if(MATLAB_LLVM_WITH_SYM)
    # Symbolic Math Toolbox end-to-end tests. Drives -emit-cpp through
    # clang + libsympp + the matlab runtime; compares stdout to per-
    # fixture .stdout files. Skip-if-missing on absent SymPP install.
    add_test(
      NAME run-tests-sym
      COMMAND ${CMAKE_SOURCE_DIR}/test/RunSym/run_tests.sh $<TARGET_FILE:matlabc>
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/RunSym
    )
    set_tests_properties(run-tests-sym PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

# --- Test timeouts (tiered: bound hangs without false-failing slow lanes) ----
# Lanes fall into distinct categories with very different runtimes, so a flat
# timeout is wrong (a 300 s cap false-times-out the slow execute lanes; a
# generous cap lets a hung DAP lane freeze the gate). Four tiers:
#
#   A. DAP / sim-dap lanes — drive `matlabc --sim-dap`, which waits on DAP
#      stdin and can hang forever in a headless run (this caused the original
#      31-min freeze, plus a 7-day-old stale process).  TIGHT 300 s fail-fast.
#   B. run-tests* execute / source-emit lanes — recompile the whole runtime per
#      fixture (no ccache on the script path), so they are very slow (~tens of
#      minutes on CI) but always TERMINATE.  Generous 5400 s + a shared
#      RESOURCE_LOCK so they serialize instead of thrashing under `-j`
#      (run-tests-emit-c measured ~67 s alone vs ~663 s contended).
#   C. external-toolchain lanes — Verilator / yosys / iverilog / cocotb / ADMS
#      per fixture; CPU-heavy but terminate.  1800 s + the same RESOURCE_LOCK.
#   D. everything else (light) — 1800 s default; only catches an accidental
#      hang, never false-fails (these run in well under a minute).
foreach(_t flowchart-simulate-dap-tests flowchart-simulate-dap-block-tests
           flowchart-simulate-dap-breakpoint-tests flowchart-simulate-dap-chart-tests
           flowchart-dap-tests debug-dap-tests debug-dap-jit-userfn-tests)
  if(TEST ${_t})
    set_tests_properties(${_t} PROPERTIES TIMEOUT 300)
  endif()
endforeach()
foreach(_t run-tests run-tests-emit-c run-tests-emit-cpp
           run-tests-emit-c-strict run-tests-emit-cpp-strict
           run-tests-emit-python run-tests-emit-typescript run-tests-sym)
  if(TEST ${_t})
    set_tests_properties(${_t} PROPERTIES RESOURCE_LOCK mlc_cpu_heavy TIMEOUT 5400)
  endif()
endforeach()
foreach(_t cocotb-tests flowchart-emit-chart-cocotb flowchart-emit-diagram-cocotb
           flowchart-emit-subsystem-sv-verilator flowchart-emit-subsystem-sv-cosim
           flowchart-emit-subsystem-sv-yosys run-emit-va-admslint run-emit-va-cosim
           emit-sv-tests)
  if(TEST ${_t})
    set_tests_properties(${_t} PROPERTIES RESOURCE_LOCK mlc_cpu_heavy TIMEOUT 1800)
  endif()
endforeach()
get_property(_mlc_all_tests DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY TESTS)
foreach(_mlc_t IN LISTS _mlc_all_tests)
  get_test_property(${_mlc_t} TIMEOUT _mlc_existing)
  if(NOT _mlc_existing)
    set_tests_properties(${_mlc_t} PROPERTIES TIMEOUT 1800)
  endif()
endforeach()
