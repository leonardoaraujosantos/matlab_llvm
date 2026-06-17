# IR Lowering Spec

## Purpose
The IR lowering pipeline turns the annotated AST into executable lower-level IR. It first builds a structured-control-flow MIR, then lowers to a custom `matlab` MLIR dialect, maps Sema types onto MLIR types, and runs a sequence of passes that promote slots to SSA, monomorphize user calls, refine function signatures, and lower scalar / tensor operations toward standard dialects and runtime calls. (src: lib/MIR/Builder.cpp, lib/MLIR/Lowering.cpp, lib/MLIR/TypeMapper.cpp, lib/MLIR/Passes)

## Requirements

### Requirement: AST-to-MIR construction
The system SHALL build a structured MIR from the resolved AST using a tag-based `Op` model with nested regions for control flow, without first constructing a basic-block CFG. (src: include/matlab/MIR/MIR.h, src: lib/MIR/Builder.cpp)

#### Scenario: Op creation
- **WHEN** the builder lowers AST expressions and statements
- **THEN** the system SHALL allocate typed `Op`s with operands and results (constants, alloc/load/store, arithmetic, comparisons, subscript/field/range/concat, calls, handles, and `IfOp`/`ForOp`/`WhileOp` carrying nested regions) (src: include/matlab/MIR/MIR.h, src: lib/MIR/Builder.cpp)

#### Scenario: Operator mapping
- **WHEN** AST binary/unary/postfix operators are lowered
- **THEN** the system SHALL map them deterministically to MIR op kinds (e.g. `BinOp::Mul` → `MatMul`, `BinOp::ElemMul` → `EMul`, `UnOp::Minus` → `Neg`, `PostfixOp::CTranspose` → `CTranspose`) (src: lib/MIR/Lowering.cpp)

### Requirement: Slot-based variable model
The system SHALL materialize an alloc slot per parameter and local variable, spilling incoming parameters into their slots at function entry and loading output slots at the implicit return. (src: lib/MIR/Lowering.cpp)

#### Scenario: Parameter spill and output load
- **WHEN** a function is lowered
- **THEN** the system SHALL emit `alloc` + `store` for each parameter at entry, pre-allocate local slots in the prologue, and load all output slots at the return, tracking bindings to slots via a slot map (src: lib/MIR/Lowering.cpp, test/MIR/function_with_outputs.m)

### Requirement: Lowering to the matlab MLIR dialect
The system SHALL lower the program to a custom `matlab` MLIR dialect emitted as unregistered operations (e.g. `matlab.alloc`, `matlab.load`, `matlab.store`, `matlab.add`, `matlab.call`, `matlab.subscript`). (src: lib/MLIR/Lowering.cpp, src: lib/MLIR/Dialect/MatlabDialect.cpp)

#### Scenario: Unregistered dialect ops
- **WHEN** the lowerer walks the AST
- **THEN** the system SHALL emit `matlab.*` operations by string name, with the `MatlabDialect` allowing unknown operations so later passes can rewrite them without registered C++ op classes (src: lib/MLIR/Lowering.cpp, src: lib/MLIR/Dialect/MatlabDialect.cpp, test/MLIR/literals_and_arith.m)

#### Scenario: Dialect registration
- **WHEN** the MLIR context is constructed
- **THEN** the system SHALL register the standard dialects it depends on (arith, cf, func, LLVM, memref, scf, tensor) alongside the `MatlabDialect` (src: lib/MLIR/Context.cpp)

### Requirement: Sema-type to MLIR-type mapping
The system SHALL map Sema `Type`s onto MLIR types according to dtype, shape, and kind. (src: lib/MLIR/TypeMapper.cpp, src: include/matlab/MLIR/TypeMapper.h)

#### Scenario: Scalar and array mapping
- **WHEN** a Sema type is mapped
- **THEN** the system SHALL map scalar Double→`f64`, Single→`f32`, IntN→signless `iN`, Logical→`i1`, Char→`i8`, and Complex/Cell/Object→`!llvm.ptr`; map known-rank arrays to ranked tensors and unknown-rank arrays to unranked tensors; and map fixed-point types to sized integers per their spec (src: lib/MLIR/TypeMapper.cpp)

### Requirement: Slot promotion to SSA
The system SHALL promote intra-block `matlab.alloc`/`load`/`store` chains to SSA values, eliminating the slot where all loads are resolved. (src: lib/MLIR/Passes/SlotPromotion.cpp)

#### Scenario: Straight-line promotion
- **WHEN** an `alloc` is used only by loads and stores within a single block
- **THEN** the system SHALL replace each load with the current stored value and erase the stores and alloc once all loads are promoted (src: lib/MLIR/Passes/SlotPromotion.cpp, test/Opt/straight_line_arith.expected)

### Requirement: User-call lowering and monomorphization
The system SHALL lower `matlab.call` to `func.call`, retype `none`-typed signatures from call-site and return operand types, and clone callees per distinct call signature. (src: lib/MLIR/Passes/LowerUserCalls.cpp)

#### Scenario: Single-site retyping
- **WHEN** a user function has a `none`-typed signature and is called
- **THEN** the system SHALL retype its parameters from call-site operand types and its results from `func.return` operand types, propagating concrete scalar types through arithmetic and recursion (src: lib/MLIR/Passes/LowerUserCalls.cpp)

#### Scenario: Per-signature cloning
- **WHEN** a user function is called with distinct argument-type signatures (e.g. `sq(5)` and `sq([1 2 3])`)
- **THEN** the system SHALL bucket the call sites by signature, clone the function per bucket, redirect each bucket to its specialization, and re-run lowering on the clones (src: lib/MLIR/Passes/LowerUserCalls.cpp, doc: docs/sema.md)

### Requirement: Function-signature refinement
The system SHALL patch function signatures after type refinement so callee parameter and result types agree with call sites and returns, idempotently. (src: lib/MLIR/Passes/RefineFuncSigs.cpp)

#### Scenario: Signature patching
- **WHEN** a `func.call` supplies concrete operand types to a callee whose parameters or results are still `none`, or a body returns `!llvm.ptr` where the signature declared a tensor
- **THEN** the system SHALL update the callee signature to match, and SHALL remain safe to run repeatedly (src: lib/MLIR/Passes/RefineFuncSigs.cpp)

### Requirement: Scalar and tensor operation lowering
The system SHALL lower scalar `matlab.*` ops to the arith dialect and tensor `matlab.*` ops to runtime `matlab_*` C-ABI calls. (src: lib/MLIR/Passes/LowerScalarsToArith.cpp, src: lib/MLIR/Passes/LowerTensorOps.cpp)

#### Scenario: Scalar arithmetic to arith
- **WHEN** a `matlab.*` op has only scalar primitive operands and results
- **THEN** the system SHALL rewrite it to the corresponding arith op (e.g. `matlab.const_float` → `arith.constant`), promoting `i1` logical operands to `f64` for float arithmetic, and leave tensor ops for the tensor pass (src: lib/MLIR/Passes/LowerScalarsToArith.cpp)

#### Scenario: Tensor ops to runtime calls
- **WHEN** a `matlab.*` op has tensor-typed operands after signatures are refined
- **THEN** the system SHALL lower element-wise binary ops to runtime calls such as `matlab_add_mm`/`matlab_sub_mm`, lower row/column concatenation to `matlab_mat_from_buf`, and lower tensor `matlab.alloc` to `llvm.alloca` of `!llvm.ptr` (src: lib/MLIR/Passes/LowerTensorOps.cpp)
