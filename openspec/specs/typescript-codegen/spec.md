# TypeScript Codegen Spec

## Purpose
Documents the observed behavior of `matlabc`'s `-emit-typescript` (alias `-emit-ts`) backend, which translates the lowered MLIR module to a self-contained TypeScript module backed by an NDArray runtime. The emitter mirrors the Python backend's structure but uses TypeScript syntax and, because TypeScript cannot overload operators, expresses matrix arithmetic as NDArray method calls. It is the least-exercised emission lane and carries known numeric and feature gaps.

## Requirements

### Requirement: Emit self-contained TypeScript source
The system SHALL, in `-emit-typescript` / `-emit-ts` mode, translate the lowered MLIR module to a TypeScript module and write it to stdout.

#### Scenario: Emitting TypeScript for a script
- **WHEN** a user runs `matlabc -emit-typescript foo.m`
- **THEN** the system SHALL print a self-contained TypeScript module that runs on Bun, tsx, or ts-node (src: tools/matlabc/main.cpp; src: lib/MLIR/Passes/EmitTypeScript.cpp)

#### Scenario: Alias flag
- **WHEN** a user passes `-emit-ts`
- **THEN** the system SHALL behave identically to `-emit-typescript` (src: tools/matlabc/main.cpp)

### Requirement: NDArray-backed matrices
The system SHALL represent matrices as an NDArray type backed by a flat `Float64Array` plus a shape vector, with scalars represented as 1x1 arrays.

#### Scenario: Matrix literal
- **WHEN** the program contains a matrix literal
- **THEN** the system SHALL construct an NDArray with row-major data and a shape array (src: runtime/shim/numpy_ts.ts)

### Requirement: Method-call matrix operators
The system SHALL emit matrix arithmetic and comparisons as NDArray method calls because TypeScript has no operator overloading.

#### Scenario: Element-wise addition
- **WHEN** the program computes `A + B` on matrices
- **THEN** the system SHALL emit `A.add(B)` and similarly map `.* / .^ - == ~= < <= > >=` to NDArray methods (src: runtime/shim/numpy_ts.ts; src: lib/MLIR/Passes/EmitTypeScript.cpp)

### Requirement: NDArray and runtime shim imports
The system SHALL import the `numpy_ts` NDArray library and the `matlab_runtime` shim, including each only when referenced.

#### Scenario: Resolving imports at run time
- **WHEN** the emitted module is run alongside the shim files in `runtime/shim`
- **THEN** the `import * as np from "./numpy_ts"` and `import * as rt from "./matlab_runtime"` paths SHALL resolve so the program executes (src: runtime/shim/numpy_ts.ts; src: runtime/shim/matlab_runtime.ts)

### Requirement: Golden test coverage with skips
The system SHALL verify TypeScript output by executing it and comparing stdout against goldens, honoring skip and per-target override markers.

#### Scenario: TypeScript run lane
- **WHEN** the `run-tests-emit-typescript` lane runs a program
- **THEN** the system SHALL emit TypeScript, execute it with a detected runner, and diff stdout against `.stdout-typescript` or `.stdout`, skipping `.skip-emit-typescript` cases (src: CMakeLists.txt; src: test/Run/run_tests_emitts.sh)

### Requirement: Documented gaps
The system SHALL treat the TypeScript backend as the least-mature lane, rejecting unsupported features such as Symbolic Math at emit time and skipping numerically sensitive or fixed-point cases.

#### Scenario: Symbolic input
- **WHEN** a program uses Symbolic Math operations
- **THEN** the system SHALL emit a diagnostic rather than incorrect TypeScript (src: lib/MLIR/Passes/EmitTypeScript.cpp; doc: docs/fixed_point_toolbox_roadmap.md)
