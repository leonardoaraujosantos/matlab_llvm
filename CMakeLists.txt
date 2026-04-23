cmake_minimum_required(VERSION 3.20)
project(matlab_llvm CXX)

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
)
target_link_libraries(MatlabAST PUBLIC MatlabBasic)

add_library(MatlabParse STATIC
  lib/Parse/Parser.cpp
)
target_link_libraries(MatlabParse PUBLIC MatlabLex MatlabAST)

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
    lib/MLIR/Passes/LowerIO.cpp
    lib/MLIR/Passes/LowerParfor.cpp
    lib/MLIR/Passes/LowerSeqLoops.cpp
    lib/MLIR/Passes/LowerAnonCalls.cpp
    lib/MLIR/Passes/LowerUserCalls.cpp
    lib/MLIR/Passes/LowerScalarSlots.cpp
    lib/MLIR/Passes/LowerTensorOps.cpp
    lib/MLIR/Passes/LowerToLLVMIR.cpp
    lib/MLIR/Passes/EmitC.cpp
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

add_executable(matlabc tools/matlabc/main.cpp)
target_link_libraries(matlabc
  PRIVATE MatlabMIR MatlabParse MatlabSema MatlabAST MatlabLex MatlabBasic)
if(MATLAB_LLVM_WITH_MLIR)
  target_link_libraries(matlabc PRIVATE MatlabMLIR)
  target_compile_definitions(matlabc PRIVATE MATLAB_LLVM_WITH_MLIR=1)
endif()

enable_testing()
add_test(
  NAME frontend-tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/run_tests.sh $<TARGET_FILE:matlabc>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
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
endif()
