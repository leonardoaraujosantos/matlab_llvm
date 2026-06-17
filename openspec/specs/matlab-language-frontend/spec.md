# MATLAB Language Frontend Spec

## Purpose
The MATLAB language frontend lexes, parses, and semantically analyzes `.m` source into an annotated `TranslationUnit` AST. It tokenizes MATLAB syntax (including its context-sensitive ambiguities), parses scripts / functions / classdefs with full control flow, then resolves every name to a binding and infers a static type for every expression so that downstream lowering is deterministic. (src: lib/Lex/Lexer.cpp, lib/Parse/Parser.cpp, lib/Sema/Resolver.cpp, doc: docs/sema.md)

## Requirements

### Requirement: Tokenizing MATLAB source
The system SHALL tokenize `.m` source into the token kinds declared in the token-kind table, including keywords, arithmetic / element-wise / comparison / logical operators, integer / float / imaginary / string / char literals, line continuation, and comments. (src: include/matlab/Lex/TokenKinds.def, src: lib/Lex/Lexer.cpp)

#### Scenario: Keywords and operators
- **WHEN** the lexer reads control-flow keywords (`if`, `for`, `while`, `switch`, `try`, `function`, `classdef`, `parfor`, …) and dotted element-wise operators (`.^`, `.*`, `./`, `.\`)
- **THEN** the system SHALL emit the matching keyword and operator token kinds defined in `TokenKinds.def` (src: include/matlab/Lex/TokenKinds.def)

#### Scenario: Numeric literal forms
- **WHEN** the source contains hex (`0x...`), binary (`0b...`), floating-point, exponent, or imaginary (`i`/`j` suffix) numbers
- **THEN** the system SHALL lex them as the appropriate literal token, including treating a trailing `.` followed by an operator as a separate dotted operator rather than part of the number (src: lib/Lex/Lexer.cpp, test/Lexer/numbers.m)

#### Scenario: Comments and line continuation
- **WHEN** the source contains `%`/`#` line comments, `%{`…`%}` block comments, or a `...` line continuation
- **THEN** the system SHALL consume them so they do not appear as ordinary tokens, with `...` suppressing the statement terminator across the newline (src: lib/Lex/Lexer.cpp, test/Lexer/comments_and_continuation.m)

### Requirement: Transpose versus character-literal disambiguation
The system SHALL disambiguate the apostrophe (`'`) between the transpose operator and a single-quoted character literal based on the preceding token. (src: lib/Lex/Lexer.cpp, test/Lexer/transpose_vs_string.m)

#### Scenario: Apostrophe after a value
- **WHEN** an apostrophe immediately follows an identifier, literal, `)`, `]`, `}`, `end`, or `.`
- **THEN** the system SHALL lex it as the transpose operator token (src: lib/Lex/Lexer.cpp)

#### Scenario: Apostrophe in operand position
- **WHEN** an apostrophe appears in any other position
- **THEN** the system SHALL lex it as the start of a single-quoted `char_literal` (src: lib/Lex/Lexer.cpp, test/Lexer/transpose_vs_string.m)

### Requirement: Parsing scripts, functions, and classdef files
The system SHALL parse a `.m` file into a `TranslationUnit` containing an optional `Script`, top-level `Function`s (with nested functions), and `ClassDef`s, distinguishing script files from function files from classdef files. (src: lib/Parse/Parser.cpp, src: include/matlab/AST/AST.h, test/Parser/function_file.m)

#### Scenario: Script versus function file
- **WHEN** a file begins with statements (no leading `function`/`classdef`) versus one or more `function` blocks versus a `classdef`
- **THEN** the system SHALL build a `Script`, a set of `Function`s (each able to hold `Nested` functions), or a `ClassDef`, respectively (src: lib/Parse/Parser.cpp, src: include/matlab/AST/AST.h)

#### Scenario: Multi-return and ignored outputs
- **WHEN** the source assigns `[a, b] = f(...)` (with `~` permitted for an ignored output) or declares a function with an output list `[O1, O2] = name(...)`
- **THEN** the system SHALL parse the output list and the multi-return assignment, including `varargin`/`varargout` parameter names (src: lib/Parse/Parser.cpp, test/Parser/multi_assign.m)

### Requirement: Control-flow statement parsing
The system SHALL parse the full MATLAB control-flow surface — `if`/`elseif`/`else`, `for`, `parfor`, `while`, `switch`/`case`/`otherwise`, `try`/`catch`, and `break`/`continue`/`return` — plus `global`, `persistent`, and `import` declarations. (src: lib/Parse/Parser.cpp, test/Parser/control_flow.m)

#### Scenario: Branching and loops
- **WHEN** the source contains nested `if`/`elseif`/`else`, `for`/`parfor`, `while`, `switch` with multiple `case` clauses and one `otherwise`, or `try`/`catch`
- **THEN** the system SHALL build the corresponding `IfStmt` / `ForStmt` (with `IsParfor` set for `parfor`) / `WhileStmt` / `SwitchStmt` / `TryStmt` AST nodes (src: lib/Parse/Parser.cpp, src: include/matlab/AST/AST.h)

### Requirement: Expression parsing with MATLAB precedence and indexing
The system SHALL parse expressions with a Pratt parser honoring MATLAB operator precedence and right-associative power (`^`/`.^`), and SHALL parse matrix / cell literals, ranges (`a:b:c`), call-or-index syntax, field access, dynamic field access, and the `end` keyword inside index contexts. (src: lib/Parse/Parser.cpp, test/Parser/expressions.m, test/Parser/end_in_indexing.m)

#### Scenario: Power right-associativity and whitespace-sensitive matrices
- **WHEN** the source contains `^`/`.^` chains or a bracketed matrix literal whose element separation depends on whitespace (e.g. `[1 -2]` versus `[1-2]`)
- **THEN** the system SHALL parse power as right-associative and apply MATLAB's whitespace rules to split matrix elements (src: lib/Parse/Parser.cpp, test/Parser/matrix_whitespace.m)

#### Scenario: `end` in subscripts
- **WHEN** `end` appears inside an indexing context such as `a(end)`, `a(1:end)`, or `a(end-1)`
- **THEN** the system SHALL parse it as an `EndExpr`, and SHALL reject `end` outside of an index context (src: lib/Parse/Parser.cpp, test/Parser/end_in_indexing.m)

### Requirement: Anonymous functions and function handles
The system SHALL parse anonymous functions `@(params) body` and named function handles `@name` into `AnonFunction` and `FuncHandle` AST nodes. (src: lib/Parse/Parser.cpp, src: include/matlab/AST/AST.h, test/Parser/anon_and_handle.m)

#### Scenario: Handle forms
- **WHEN** the source contains `@(x) x + 1` or `@sin`
- **THEN** the system SHALL produce an `AnonFunction` capturing its parameters and body, or a `FuncHandle` naming the referenced function (src: lib/Parse/Parser.cpp, test/Parser/anon_and_handle.m)

### Requirement: Name resolution to bindings
The system SHALL resolve every `NameExpr` to a `Binding` (with a `BindingKind` of Var, Param, Output, Global, Persistent, Function, Builtin, Import, or Class), using a two-pass walk that pre-declares assignment targets and pre-registers builtins in the global scope. (src: lib/Sema/Resolver.cpp, src: include/matlab/Sema/Scope.h, doc: docs/sema.md)

#### Scenario: Builtin pre-registration
- **WHEN** the resolver runs before any user code
- **THEN** the system SHALL seed the global scope with the registered builtin names via `registerBuiltins`, so a call like `disp(x)` resolves to a `Builtin` binding without an import step (src: lib/Sema/Resolver.cpp)

#### Scenario: Forward reference within a body
- **WHEN** a name is assigned later in the same function or script body, or is a `for`-loop variable / `try`-`catch` error variable / `global` / `persistent` target
- **THEN** the system SHALL pre-declare its binding during the pre-pass so earlier uses resolve (src: lib/Sema/Resolver.cpp, doc: docs/sema.md)

#### Scenario: Undefined name
- **WHEN** a name cannot be resolved and the resolver is not in REPL mode
- **THEN** the system SHALL emit an `undefined name` diagnostic (src: lib/Sema/Resolver.cpp, test/Sema/undefined_name.expected)

### Requirement: Call-versus-index disambiguation
The system SHALL classify each `CallOrIndex` node as a call or an index based on the resolved callee binding. (src: lib/Sema/Resolver.cpp, test/Sema/call_vs_index.m)

#### Scenario: Variable subscript versus function call
- **WHEN** the callee resolves to a Var/Param/Output/Global/Persistent binding versus a Function/Builtin/Class binding
- **THEN** the system SHALL mark the node as `Index` for the former (e.g. `x(2)`) and `Call` for the latter (e.g. `sin(0.5)`), and SHALL treat a `FieldAccess` callee whose base is pinned to a class with a matching method as a method `Call` (src: lib/Sema/Resolver.cpp, test/Sema/call_vs_index.m)

### Requirement: classdef OOP with inheritance and operator overloading
The system SHALL resolve `classdef`s with properties, methods, static methods, enumeration members, and an optional superclass, and SHALL pin class identity so method and operator-overload dispatch can route statically. (src: lib/Sema/Resolver.cpp, src: include/matlab/AST/AST.h, test/Sema/object_type.m, test/Sema/operator_dispatch_rewrite.m)

#### Scenario: Method self-pinning
- **WHEN** the resolver processes a constructor versus a non-constructor instance method
- **THEN** the system SHALL pin the constructor's output binding to the class and pin a non-constructor method's first parameter (`obj`) to the class, so in-body property access routes through the class property table (src: lib/Sema/Resolver.cpp, doc: docs/sema.md)

#### Scenario: Operator overload dispatch
- **WHEN** a binary operator (`plus`, `minus`, `eq`, `lt`, …) is overloaded for a user class and applied to instances of that class
- **THEN** the system SHALL pin the second operand to the class and rewrite the operator into the corresponding method call so it lowers as a dispatch (src: lib/Sema/Resolver.cpp, test/Sema/operator_dispatch_rewrite.m)

### Requirement: Static type inference
The system SHALL infer a static `Type` for every expression and an `InferredType` for every binding over the resolved AST, using a lattice of `Dtype`, `Shape`, and `Type::Kind` (Any, Array, StringArray, Cell, Struct, FuncHandle, Numerictype, Fimath, Object) and falling back to `Any` when it cannot be sure. (src: lib/Sema/TypeInference.cpp, src: include/matlab/Sema/Type.h, doc: docs/sema.md)

#### Scenario: Promotion and broadcasting
- **WHEN** a binary operation combines operands of differing dtype or shape
- **THEN** the system SHALL compute the result dtype via `promoteDtype` and the result shape via `broadcastShape` (MATLAB implicit expansion), using dynamic extents where dimensions conflict (src: lib/Sema/Type.cpp, test/Sema/shape_inference.m)

#### Scenario: Object types for class instances
- **WHEN** a constructor call or operator overload on an instance is type-inferred
- **THEN** the system SHALL assign a `Type::Kind::Object` carrying the owning `ClassDef` (e.g. `Box(3)` infers `object<Box>`) (src: lib/Sema/TypeInference.cpp, test/Sema/object_type.m)

### Requirement: Monomorphization of user functions
The system SHALL clone user functions per call-site argument signature and stamp concrete argument types and arity overrides onto each clone, enabled by default and disabled when `MATLAB_LLVM_SEMA_MONO=0`. (src: lib/Sema/Monomorphize.cpp, src: lib/Sema/CallSiteAnalyzer.cpp, doc: docs/sema.md)

#### Scenario: Per-signature cloning
- **WHEN** a user function is called with distinct argument-type signatures or distinct arities
- **THEN** the system SHALL bucket the call sites, clone the function per signature with a mangled name, stamp `ParamTypeStamps` (and `NarginOverride` for reduced arity), rewrite the call sites, and re-run Sema to a fixpoint (src: lib/Sema/Monomorphize.cpp, doc: docs/sema.md)

#### Scenario: Disabling Sema-time monomorphization
- **WHEN** `MATLAB_LLVM_SEMA_MONO=0` is set in the environment
- **THEN** the system SHALL skip Sema-time cloning and leave the equivalent specialization to the late MLIR `runMonomorphiseUserCalls` pass (doc: docs/sema.md, src: lib/MLIR/Passes/LowerUserCalls.cpp)
