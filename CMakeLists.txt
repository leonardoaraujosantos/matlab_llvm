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
option(MATLAB_LLVM_WITH_SYM "Build the matlab_sym SymPP-backed shared library" OFF)
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
  message(STATUS "Found SymPP — building libmatlab_sym shared library")
  # #50 Phase 3 — build the SymPP-backed shared library that the
  # runtime_sym_stub.cpp wrappers dlopen at first sym call.  The
  # shared lib statically pulls SymPP::sympp so users only need
  # libmatlab_sym.dylib (and its transitive GMP/MPFR deps) at runtime,
  # not the SymPP install tree.  matlabc itself NO LONGER links
  # libsympp — the stub layer is sufficient.
  add_library(matlab_sym SHARED runtime/toolbox/sym/runtime_sym.cpp)
  target_include_directories(matlab_sym PRIVATE ${CMAKE_SOURCE_DIR}/runtime)
  target_link_libraries(matlab_sym PRIVATE SymPP::sympp)
  # Ship the .dylib next to matlabc so the dlopen's @executable_path
  # probe finds it.  On Linux, $ORIGIN-relative lookup does the same.
  set_target_properties(matlab_sym PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# --- Plotting (Cairo backend) ----------------------------------------------
# Off by default; turn on with -DMATLAB_LLVM_WITH_PLOT=ON. Adds runtime
# plotting symbols (matlab_plot, matlab_figure, matlab_savefig, ...) backed
# by Cairo's image/svg/pdf surfaces. Headless: no subprocess, no display
# server, no temp files. Cross-platform target: macOS, Linux, iOS.
option(MATLAB_LLVM_WITH_PLOT "Enable matlab_plot runtime (Cairo backend)" OFF)
if(MATLAB_LLVM_WITH_PLOT)
  find_package(PkgConfig REQUIRED)
  # Cairo headers are required at compile time for the type declarations
  # used by cairo_dl.cpp + cairo_render.cpp.  The actual libcairo binary
  # is loaded at runtime via dlopen (#50 Phase 2) — programs that don't
  # plot never trigger the dlopen and ship without any libcairo dep.
  pkg_check_modules(CAIRO REQUIRED cairo cairo-svg cairo-pdf)
  message(STATUS "Found Cairo ${CAIRO_VERSION} — matlab_plot enabled "
                 "(headers only; libcairo loaded via dlopen at runtime)")
  set(MATLAB_PLOT_SOURCES
    runtime/plot/c_api.cpp
    runtime/plot/figure.cpp
    runtime/plot/cairo_render.cpp
    runtime/plot/cairo_dl.cpp
    runtime/plot/colormap.cpp
    runtime/plot/contour.cpp)
  set(MATLAB_PLOT_INCLUDE_DIRS ${CAIRO_INCLUDE_DIRS})
endif()

# --- BLAS / LAPACK acceleration --------------------------------------------
# Default ON on Apple (Accelerate is preinstalled and free), OFF
# everywhere else so the runtime stays library-agnostic for cross-
# compile / embedded targets. Per docs/lapack_roadmap.md §0, the
# emitted C / C++ source NEVER depends on BLAS — only this build of
# the runtime does, and only when this flag is on. Verified by
# `nm runtime/matlab_runtime.cpp.o | grep cblas_` returning empty
# when WITH_BLAS=OFF.
if(NOT DEFINED MATLAB_LLVM_WITH_BLAS)
  if(APPLE)
    set(MATLAB_LLVM_WITH_BLAS ON CACHE BOOL
        "Link runtime against BLAS / LAPACK (host-CPU dense linalg
         acceleration; preserves the cross-emit invariant)" FORCE)
  else()
    set(MATLAB_LLVM_WITH_BLAS OFF CACHE BOOL
        "Link runtime against BLAS / LAPACK (host-CPU dense linalg
         acceleration; preserves the cross-emit invariant)" FORCE)
  endif()
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
  lib/AST/Cloner.cpp
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
  lib/Sema/RewriteDspSoForSv.cpp
  lib/Sema/CallSiteAnalyzer.cpp
  lib/Sema/Monomorphize.cpp
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
    lib/MLIR/Passes/ForwardParforCaptures.cpp
    lib/MLIR/Passes/LowerGpuKernels.cpp
    lib/MLIR/Passes/EmitMetal.cpp
    lib/MLIR/Passes/EmitCUDA.cpp
    lib/MLIR/Passes/EmitOpenCL.cpp
    lib/MLIR/Passes/PromoteNoneParams.cpp
    lib/MLIR/Passes/PromoteBinopTypes.cpp
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

# --------------------------------------------------------------------------
# MatlabRuntime — single static archive bundling every unconditional runtime
# TU (core + every shipped toolbox + GPU dispatcher).  See docs/runtime.md
# §3 and issue #50 for the architectural argument.
#
# Why one library:
#   * Deduplicates the source list — pre-#50, the same 19 .cpp files were
#     enumerated in target_sources(matlabc) + every runtime-test-* +
#     runtime-plot-* foreach.  Now a single list in one place.
#   * External callers (the IDE, the bench harness, downstream packagers)
#     can link `libMatlabRuntime.a` instead of recompiling 19 TUs every
#     time they build a `.m`.  Per-toolbox dead-stripping is the linker's
#     job (-Wl,-dead_strip on macOS, -Wl,--gc-sections on Linux); a
#     program that only calls `disp(1)` won't pay for plot/sym/PDE/etc.
#
# Conditional toolboxes (Cairo plot, SymPP sym, Metal GPU, BLAS) stay
# attached to matlabc separately below — those carry external dependencies
# we don't want to force-link into every consumer (Cairo / SymPP will
# migrate to dlopen in #50 Phase 2-3).
add_library(MatlabRuntime STATIC
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
    runtime/toolbox/finance/runtime_finance.cpp
    runtime/toolbox/econ/runtime_econ.cpp
    runtime/toolbox/fusion/runtime_fusion.cpp
    runtime/toolbox/robotics/runtime_robotics.cpp
    runtime/toolbox/navigation/runtime_navigation.cpp
    runtime/toolbox/dlnet/runtime_dlnet.cpp
    runtime/toolbox/dlnet/runtime_onnx.cpp
    runtime/toolbox/rl/runtime_rl.cpp
    runtime/toolbox/stateflow/runtime_mstateflow.cpp
    runtime/gpu/runtime_gpu.cpp
    runtime/toolbox/gpu/runtime_gpu_helpers.cpp
    # #50 Phase 3 — Symbolic Math Toolbox dlopen stub.  Always
    # compiled into MatlabRuntime; on first sym call the stub
    # dlopens libmatlab_sym.dylib (built when MATLAB_LLVM_WITH_SYM=ON).
    # When the user's program doesn't use sym, dead-strip drops every
    # stub and no dlopen runs.  matlabc no longer needs the libsympp
    # link at all.
    runtime/toolbox/sym/runtime_sym_stub.cpp)
target_include_directories(MatlabRuntime PUBLIC ${CMAKE_SOURCE_DIR}/runtime)
target_link_libraries(MatlabRuntime PUBLIC m ${CMAKE_DL_LIBS})
# -ffunction-sections + -fdata-sections give the linker per-symbol
# granularity so external callers' dead-strip pass can drop unused
# toolboxes one function at a time.  Mac's ld64 is per-symbol by
# default; Linux's bfd / lld want the section split.  Cheap on compile
# time; saves bytes on the link.
target_compile_options(MatlabRuntime PRIVATE
    -ffunction-sections -fdata-sections)

# #54: when configured with -DMATLAB_LLVM_WITH_PLOT=ON, fold the Cairo
# plot TUs into the same MatlabRuntime archive so external callers
# (per docs/build_and_run.md) get matlab_plot / matlab_figure / etc.
# from a single -lMatlabRuntime instead of having to recompile the
# plot sources themselves.  Dead-strip still drops the plot symbols
# from any program that doesn't call them (same per-symbol sections
# as the rest of the archive).
if(MATLAB_LLVM_WITH_PLOT)
  target_sources(MatlabRuntime PRIVATE ${MATLAB_PLOT_SOURCES})
  target_include_directories(MatlabRuntime PRIVATE ${MATLAB_PLOT_INCLUDE_DIRS})
  target_compile_definitions(MatlabRuntime PRIVATE MATLAB_LLVM_WITH_PLOT=1)
endif()

# Helper — force-load every symbol from MatlabRuntime into a consumer.
# matlabc's MLIR REPL JIT resolves runtime symbols via
# LLJIT's DynamicLibrarySearchGenerator::GetForCurrentProcess, so every
# matlab_* entry has to be in the executable's symbol table even if the
# linker can't statically prove a need for it.  External callers (the
# script-replacement invocation, the IDE, packagers) link WITHOUT
# force-load so dead-strip kicks in normally.
function(matlab_runtime_force_load TGT)
  if(APPLE)
    target_link_options(${TGT} PRIVATE
        "-Wl,-force_load,$<TARGET_FILE:MatlabRuntime>")
  else()
    target_link_options(${TGT} PRIVATE
        "LINKER:--whole-archive"
        "$<TARGET_FILE:MatlabRuntime>"
        "LINKER:--no-whole-archive")
  endif()
endfunction()

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
  # The runtime is now consolidated in the MatlabRuntime static archive
  # (see the add_library above).  matlabc force-links every symbol so
  # the MLIR REPL JIT can resolve them via the process's symbol table.
  target_link_libraries(matlabc PRIVATE MatlabRuntime)
  matlab_runtime_force_load(matlabc)
  if(MATLAB_LLVM_WITH_SYM)
    # #50 Phase 3: matlabc only depends on the matlab_sym SHARED library
    # via dlopen (the stub in MatlabRuntime handles the load lazily).
    # Add an order-only dep so the .dylib is built before matlabc — the
    # @executable_path dlopen probes find it next to the binary.
    add_dependencies(matlabc matlab_sym)
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_SYM=1)
  endif()
  # LAPACK acceleration epic (#45). When ON, the runtime's hand-rolled
  # dense linalg kernels dispatch to a host BLAS / LAPACK library for
  # matrices above an amortisation threshold. The naive fallback stays
  # the only path when this flag is OFF — preserving the cross-emit
  # invariant (no cblas_* symbol references in the runtime object).
  # See docs/lapack_roadmap.md §0 for the architectural argument.
  if(MATLAB_LLVM_WITH_BLAS)
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_BLAS=1)
    if(APPLE)
      # Apple Accelerate ships LAPACK + BLAS + vDSP + vImage; preinstalled
      # on every Mac. Hand-tuned for the AMX matrix coprocessor on
      # Apple Silicon.
      target_link_libraries(matlabc PRIVATE "-framework Accelerate")
      message(STATUS "BLAS: linking Apple Accelerate (-framework Accelerate)")
    else()
      # Linux / other — let CMake's FindBLAS / FindLAPACK pick the
      # preferred lib (OpenBLAS / MKL / generic). Default backend
      # picked by the BLA_VENDOR variable; defaults to system search.
      find_package(BLAS)
      find_package(LAPACK)
      if(BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(matlabc PRIVATE
            ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        message(STATUS "BLAS: ${BLAS_LIBRARIES}")
        message(STATUS "LAPACK: ${LAPACK_LIBRARIES}")
      else()
        message(WARNING
            "MATLAB_LLVM_WITH_BLAS=ON but no BLAS / LAPACK found via CMake's"
            " FindBLAS / FindLAPACK; disabling. Pass -DBLA_VENDOR=<vendor>"
            " (e.g. OpenBLAS / Intel10_64lp / Generic) to pick one.")
        # Strip the define so the runtime falls back to the naive path
        # rather than failing to link.
        get_target_property(_defs matlabc COMPILE_DEFINITIONS)
        list(REMOVE_ITEM _defs MATLAB_LLVM_WITH_BLAS=1)
        set_target_properties(matlabc PROPERTIES COMPILE_DEFINITIONS "${_defs}")
      endif()
    endif()
  endif()
  if(MATLAB_LLVM_WITH_PLOT)
    # The plot TUs now ship inside MatlabRuntime (see add_library above);
    # matlabc force-loads the archive, so we only need to flip the macro
    # that gates the matlabc-side plot dispatch tables.
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_PLOT=1)
  endif()
  # GPU Coder — Metal backend (T2).  Auto-on on macOS, off elsewhere.
  # The .mm TU is compiled as Objective-C++ via the file extension;
  # link against Metal + MetalPerformanceShaders + Foundation
  # frameworks (always available on macOS, no Homebrew dep).
  if(APPLE AND NOT DEFINED MATLAB_LLVM_GPU_METAL)
    set(MATLAB_LLVM_GPU_METAL ON CACHE BOOL "Build Metal GPU backend" FORCE)
  endif()
  if(MATLAB_LLVM_GPU_METAL AND APPLE)
    target_sources(matlabc PRIVATE runtime/gpu/metal/runtime_gpu_metal.mm)
    target_link_libraries(matlabc PRIVATE
      "-framework Metal"
      "-framework MetalPerformanceShaders"
      "-framework Foundation")
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_GPU_METAL=1)
  endif()
  # GPU Coder — CUDA backend (T3).  Opt-in (default OFF) so existing
  # CI / macOS builds are unchanged; enable with -DMATLAB_LLVM_GPU_CUDA=ON
  # on a Linux box with an NVIDIA GPU.  The TU uses only the CUDA driver
  # API + NVRTC + a few hand-declared cuBLAS prototypes, so it links
  # against just libcuda (driver, system), libnvrtc, and libcublas — no
  # full nvcc toolkit needed.  Discovery prefers a system CUDA install
  # (find_package(CUDAToolkit)); if that's absent it falls back to the
  # pip-wheel CUDA libs (nvidia-cuda-nvrtc-cu12 / nvidia-cublas-cu12),
  # which is how this is validated on the dev box.  Override any path
  # with -DMATLAB_CUDA_INCLUDE_DIRS / -DMATLAB_CUDA_LIBRARIES.
  option(MATLAB_LLVM_GPU_CUDA "Build CUDA GPU backend" OFF)
  if(MATLAB_LLVM_GPU_CUDA)
    if(NOT DEFINED MATLAB_CUDA_LIBRARIES OR MATLAB_CUDA_LIBRARIES STREQUAL "")
      set(_cuda_inc "")
      set(_cuda_libs "")
      set(_cuda_rpath "")
      # 1. Prefer a system CUDA toolkit.
      find_package(CUDAToolkit QUIET)
      # Driver stub (libcuda) — provided by the NVIDIA driver, not the
      # toolkit; search the usual system locations + any CUDA stubs dir.
      find_library(MATLAB_CUDA_DRIVER_LIB cuda
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib
              ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)
      if(CUDAToolkit_FOUND)
        list(APPEND _cuda_inc ${CUDAToolkit_INCLUDE_DIRS})
        list(APPEND _cuda_libs CUDA::nvrtc CUDA::cublas)
        message(STATUS "CUDA backend: using system CUDAToolkit "
                       "${CUDAToolkit_VERSION} (${CUDAToolkit_INCLUDE_DIRS})")
      else()
        # 2. Fall back to pip-wheel CUDA libs under a Python site-packages.
        #    Locate via the active python's site dir, then glob.
        find_program(_py NAMES python3 python)
        set(_site "")
        if(_py)
          execute_process(COMMAND ${_py} -c
            "import site,sys;print(next((p for p in site.getsitepackages()+[site.getusersitepackages()] if p),''))"
            OUTPUT_VARIABLE _site OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        endif()
        file(GLOB _nvrtc_lib  "${_site}/nvidia/cuda_nvrtc/lib/libnvrtc.so*")
        file(GLOB _cublas_lib "${_site}/nvidia/cublas/lib/libcublas.so*")
        # Driver-API + nvrtc headers: prefer triton's complete include
        # tree (ships crt/), else the cuda_nvrtc wheel's include.
        set(_drv_inc "")
        foreach(_cand
            "${_site}/triton/backends/nvidia/include"
            "${_site}/nvidia/cuda_runtime/include")
          if(EXISTS "${_cand}/cuda.h")
            set(_drv_inc "${_cand}")
            break()
          endif()
        endforeach()
        if(_nvrtc_lib AND _cublas_lib AND _drv_inc
           AND EXISTS "${_site}/nvidia/cuda_nvrtc/include/nvrtc.h")
          list(GET _nvrtc_lib 0 _nvrtc_lib0)
          list(GET _cublas_lib 0 _cublas_lib0)
          list(APPEND _cuda_inc "${_drv_inc}"
                                "${_site}/nvidia/cuda_nvrtc/include")
          list(APPEND _cuda_libs "${_nvrtc_lib0}" "${_cublas_lib0}")
          get_filename_component(_d1 "${_nvrtc_lib0}" DIRECTORY)
          get_filename_component(_d2 "${_cublas_lib0}" DIRECTORY)
          list(APPEND _cuda_rpath "${_d1}" "${_d2}")
          message(STATUS "CUDA backend: using pip-wheel CUDA libs "
                         "(${_nvrtc_lib0}, ${_cublas_lib0})")
        endif()
      endif()
      if(NOT _cuda_libs OR NOT MATLAB_CUDA_DRIVER_LIB)
        message(FATAL_ERROR
          "MATLAB_LLVM_GPU_CUDA=ON but no CUDA toolchain was found.\n"
          "  Install the CUDA toolkit, or the pip wheels "
          "(pip install nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12), or set\n"
          "  -DMATLAB_CUDA_INCLUDE_DIRS=... -DMATLAB_CUDA_LIBRARIES=... "
          "and ensure libcuda (driver) is present.")
      endif()
      list(APPEND _cuda_libs "${MATLAB_CUDA_DRIVER_LIB}")
      set(MATLAB_CUDA_INCLUDE_DIRS "${_cuda_inc}" CACHE STRING "CUDA include dirs")
      set(MATLAB_CUDA_LIBRARIES    "${_cuda_libs}" CACHE STRING "CUDA libraries")
      set(MATLAB_CUDA_RPATH        "${_cuda_rpath}" CACHE STRING "CUDA rpath dirs")
    endif()
    target_sources(matlabc PRIVATE runtime/gpu/cuda/runtime_gpu_cuda.cpp)
    target_include_directories(matlabc PRIVATE ${MATLAB_CUDA_INCLUDE_DIRS})
    target_link_libraries(matlabc PRIVATE ${MATLAB_CUDA_LIBRARIES})
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_GPU_CUDA=1)
    foreach(_rp ${MATLAB_CUDA_RPATH})
      target_link_options(matlabc PRIVATE "-Wl,-rpath,${_rp}")
    endforeach()
  endif()
  # GPU Coder — OpenCL backend (T4).  Opt-in (default OFF).  The TU
  # hand-declares the OpenCL 1.2 API, so it links against just the ICD
  # loader (libOpenCL) — no OpenCL SDK headers required.  Works against
  # any vendor ICD (NVIDIA / AMD / Intel).  Override the lib path with
  # -DMATLAB_OPENCL_LIB= for loaders shipped without an unversioned .so.
  option(MATLAB_LLVM_GPU_OPENCL "Build OpenCL GPU backend" OFF)
  if(MATLAB_LLVM_GPU_OPENCL)
    # Resolve into a throwaway var; rely on CMake treating "...-NOTFOUND"
    # as false so a stale cache never short-circuits re-discovery.
    set(_ocl_lib "${MATLAB_OPENCL_LIB}")  # explicit -D override wins
    if(NOT _ocl_lib)
      find_library(_ocl_found NAMES OpenCL
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64 /usr/lib)
      if(_ocl_found)
        set(_ocl_lib "${_ocl_found}")
      else()
        # ICD loader without an unversioned symlink (e.g. only libOpenCL.so.1).
        file(GLOB _ocl_cand
          "/usr/lib/x86_64-linux-gnu/libOpenCL.so*"
          "/lib/x86_64-linux-gnu/libOpenCL.so*"
          "/usr/lib64/libOpenCL.so*")
        if(_ocl_cand)
          list(GET _ocl_cand 0 _ocl_lib)
        endif()
      endif()
    endif()
    if(NOT _ocl_lib)
      message(FATAL_ERROR
        "MATLAB_LLVM_GPU_OPENCL=ON but no OpenCL ICD loader (libOpenCL) was "
        "found.\n  Install an ICD loader (e.g. ocl-icd-libopencl1) or set "
        "-DMATLAB_OPENCL_LIB=/path/to/libOpenCL.so[.1].")
    endif()
    set(MATLAB_OPENCL_LIB "${_ocl_lib}" CACHE FILEPATH
        "OpenCL ICD loader" FORCE)
    message(STATUS "OpenCL backend: linking ${_ocl_lib}")
    target_sources(matlabc PRIVATE runtime/gpu/opencl/runtime_gpu_opencl.cpp)
    target_link_libraries(matlabc PRIVATE "${_ocl_lib}")
    target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_GPU_OPENCL=1)
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
  # MatlabRuntime is now its own static library — instrument it too so
  # the runtime TUs show up in coverage reports.  Profile-instrumented
  # objects link cleanly into both matlabc (force-load) and the
  # runtime-test-* test executables (normal link); the runtime
  # variable counters are merged at process exit.
  target_compile_options(MatlabRuntime PRIVATE ${MATLAB_LLVM_COVERAGE_FLAGS})
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
  add_executable(${_rt_target} test/Runtime/test_${_rt_test}.c)
  target_include_directories(${_rt_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/test/Runtime)
  # Pull the whole runtime archive — these C tests each call into a
  # handful of toolbox-specific entries; without force-load, the linker
  # might drop adjacent TUs the C test references via fn-pointer
  # tables.  Same force-load shape matlabc uses.
  target_link_libraries(${_rt_target} PRIVATE MatlabRuntime)
  matlab_runtime_force_load(${_rt_target})
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
# The new-runtime modules went through a mechanical static_cast<>
# conversion. Lock that in with -Wold-style-cast as an error on these
# TUs only; the legacy matlab_runtime.cpp + sibling TUs predate the
# convention and would generate thousands of warnings if flagged.
# Applies file-by-file inside the MatlabRuntime library — the same
# per-file granularity matches the original setting.
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
  runtime/toolbox/finance/runtime_finance.cpp
  runtime/toolbox/econ/runtime_econ.cpp
  runtime/toolbox/fusion/runtime_fusion.cpp
  runtime/toolbox/robotics/runtime_robotics.cpp
  runtime/toolbox/navigation/runtime_navigation.cpp
  runtime/toolbox/dlnet/runtime_dlnet.cpp
  runtime/toolbox/dlnet/runtime_onnx.cpp
  runtime/toolbox/rl/runtime_rl.cpp
  runtime/toolbox/rf/runtime_rf.cpp
  runtime/toolbox/prop/runtime_prop.cpp
  runtime/toolbox/stateflow/runtime_mstateflow.cpp
  runtime/gpu/runtime_gpu.cpp
  runtime/toolbox/gpu/runtime_gpu_helpers.cpp
  TARGET_DIRECTORY MatlabRuntime
  PROPERTIES COMPILE_OPTIONS "-Wold-style-cast;-Werror=old-style-cast")

if(MATLAB_LLVM_WITH_PLOT)
  foreach(_plot_test IN ITEMS hello basic)
    set(_plot_target "runtime-plot-${_plot_test}")
    # Plot TUs are inside MatlabRuntime now (see add_library above);
    # force-load pulls every entry into the test binary so the symbol
    # table mirrors matlabc's runtime ABI.
    add_executable(${_plot_target}
      test/Runtime/test_plot_${_plot_test}.cpp)
    target_link_libraries(${_plot_target} PRIVATE MatlabRuntime)
    matlab_runtime_force_load(${_plot_target})
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
# GPU Coder Tier-6 — AOT emission gate.  Verifies `matlabc
# -emit-{cuda,metal,opencl}` produces structurally-valid bundles
# (kernel source + host driver + Makefile + README) for a representative
# `coder.gpu.kernelfun`-tagged MATLAB program.  Checks the kernel
# source contains the canonical target-specific preamble (so the
# emit pass walked the body, not just emitted a placeholder).
#
# Does NOT require GPU hardware — emission only.  On macOS, also
# confirms the Metal bundle compiles against the system Metal SDK.
# Real device validation runs separately via the Metal smoke tests
# (gpu_metal_jit_smoke.mm, gpu_metal_mps_smoke.mm) on macOS runners.
add_test(
  NAME gpu-emit-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_gpu_emit_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
)
# GPU array-capture outliner lane (issue #24) — MATLAB_GPU_OUTLINE=1.
# Confirms a flat element-wise `coder.gpu.kernelfun` kernel is lifted
# into a standalone __gpu_kernel_N llvm.func and runs bit-exact vs the
# default lane through the CPU dispatch fallback, and that an
# unsupported-class kernel (Mandelbrot) declines cleanly without
# corrupting the program.  No GPU hardware required.
add_test(
  NAME gpu-outline-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/Run/run_gpu_outline_tests.sh
          $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/Run
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
