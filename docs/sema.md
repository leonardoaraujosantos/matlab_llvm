# Sema — A Tour

This document is a guided walk through `lib/Sema/` and
`include/matlab/Sema/`. It's aimed at someone who knows what a
parser produces (an AST) and wants to understand what the compiler
does between *AST in* and *MLIR out*.

Sema is a small layer with a focused job:

1. Walk the AST.
2. For every name, decide *what* it refers to (a parameter, a local
   variable, a builtin, a class…).
3. For every expression, infer a static type.
4. Annotate AST nodes in place with that information.

After Sema runs, lowering can ask each `NameExpr` "who are you?" and
each `Expr` "what type are you?" without re-doing any work.

The directory has five `.cpp` files. They line up with the four
phases below; we'll cover them in the order data flows.

```
lib/Sema/Scope.cpp           — symbol-table primitives + SemaContext
lib/Sema/Resolver.cpp        — scope construction + name resolution
lib/Sema/Type.cpp            — Type / Shape / Dtype values + helpers
lib/Sema/TypeInference.cpp   — type propagation over the resolved AST
lib/Sema/SemaDumper.cpp      — pretty-printer for `--dump-sema`
```

## 1. The data model: `Binding`, `Scope`, `SemaContext`

The header to read first is `include/matlab/Sema/Scope.h`. It defines
three things.

**`BindingKind`** — what role a name plays. The Sema/lowering pipeline
branches on this:

```cpp
enum class BindingKind : uint8_t {
  Var, Param, Output, Global, Persistent,
  Function, Builtin, Import, Class,
};
```

**`Binding`** — one entry in a symbol table. Beyond `Kind` and `Name`
it carries:

- `FuncDef` — the AST node for user functions (null for builtins).
- `ClassDef` — the AST node for classdefs.
- `DeclaredType` — optional, e.g. for builtin signatures.
- `InferredType` — filled by type inference, read by lowering.
- `PinnedClass` — see §3.4 below; lets `obj.field` dispatch
  statically when we know what class `obj` is.
- `WrittenTo` / `ReadFrom` — used for unused-var diagnostics.

**`Scope`** — an `unordered_map<string, Binding*>` plus a parent
pointer. `lookup(N)` walks up the parent chain;
`lookupLocal(N)` doesn't.

`Scope` does **not** own bindings. `SemaContext` does. Every
`Scope*`, `Binding*`, `Type*` you see in Sema lives in arenas owned by
a single `SemaContext`, which is destroyed alongside the
`TranslationUnit`. This is why the AST can hold raw `Binding*` and
`Type*` pointers safely — they stay valid for as long as the AST does.

## 2. The pipeline at a glance

The driver runs Sema in this order (see `Resolver::resolve` in
`lib/Sema/Resolver.cpp:92`):

```
Parser → AST → Resolver → TypeInference → Monomorphize → (lowering)
                  ↑                            ↑
             SemaContext (arena)        clones per call-site
                                        signature; stamps
                                        concrete arg types
```

Two passes inside `Resolver` itself:

1. **Pre-pass**: walk every function/script body and pre-declare a
   binding for every assignment LHS. This means a later use can find
   a forward-declared variable.
2. **Resolution pass**: walk every expression and set `NameExpr::Ref`
   to the right `Binding*`, classify each `CallOrIndex` as a *call*
   vs. an *index*, and pin classes when an assignment looks like a
   constructor.

Then `TypeInference` runs over the now-fully-resolved AST, computing
`Type*` for every expression and `InferredType` for every binding.

`Monomorphize` (`lib/Sema/Monomorphize.cpp`, added by issue #38 /
PR #39) then walks the TU and clones user functions per call-site
signature, stamping concrete arg types on each clone's
`Function::ParamTypeStamps` so AST→MLIR lowering emits concrete
`func.func` signatures rather than `(none, ...) -> none`. Enabled by
default; set `MATLAB_LLVM_SEMA_MONO=0` to disable (the late MLIR
`runMonomorphiseUserCalls` then handles the same job at the matlab.call
layer). The fixpoint loop interleaves cloning, call-site rewriting,
type stamping, and Sema re-runs until no further specialisation is
discovered — `Resolver::collectAssignments` clears stale
`ParamRefs`/`OutputRefs` on each re-run so binding state stays clean.

The Sema-time pass leaves three classes to the late MLIR mono:
matrix-typed call sites (the ptr-shape settling depends on
`LowerTensorOps` materialising tensor literals first), arity-varying
callees (`add2(5)` + `add2(5, 7)` need per-arity clones with
`matlab.nargin_value`), and `varargin`/`varargout` (per-arity
cell-pack/unpack shape). Absorbing those classes Sema-side is
tracked in issue #40.

## 3. Resolver, walked through

### 3.1 Builtins are pre-declared in the global scope

`Resolver::registerBuiltins` (`lib/Sema/Resolver.cpp:19`) seeds the
global scope with around 150 builtin names — `zeros`, `disp`, `fft`,
`save`, `load`, etc. They are declared with `BindingKind::Builtin`.
This is how a call to `disp(x)` resolves without any `import` or
linker step: `disp` is just a binding with `Kind == Builtin`, and
the lowerer special-cases that kind to emit a runtime call.

If you want the parser to recognize `myThing` as a known function,
add it to that list. If you want the lowerer to actually do
something with it, that's a separate change in `LowerTensorOps.cpp`.

### 3.2 Two-pass scope construction

For every function body, `Resolver::resolveFunction`
(`lib/Sema/Resolver.cpp:315`) runs a pre-pass before the real
resolution walk:

```cpp
void Resolver::resolveFunction(Function &F, Scope *Parent) {
  F.FnScope = Sema.newScope(Parent, std::string(F.Name));
  collectAssignments(F, F.FnScope);   // ← pre-pass
  if (F.Body) resolveBlock(*F.Body, F.FnScope);
  for (Function *N : F.Nested) resolveFunction(*N, F.FnScope);
}
```

`collectAssignments` walks the body looking *only* at LHS positions
in `AssignStmt`, `ForStmt`, `try…catch X`, `global X`, `persistent X`
and pre-declares a `Var` (or stronger) binding. The reason is that
MATLAB allows forward use within a function body in some patterns —
also, for-loop variables and try-catch error variables aren't
assignments in the parser's eyes but still bind a name.

LHS expressions are not always plain names: `a(i) = …` and
`obj.field = …` and `c{1} = …` all bind the *root* name. The
pre-pass peels off `CallOrIndex` / `CellIndex` / `FieldAccess` /
`DynamicField` until it finds the `NameExpr` underneath
(`Resolver.cpp:215`).

### 3.3 Resolution: `NameExpr → Binding*`

`Resolver::resolveExpr` (`Resolver.cpp:571`) is the workhorse. For a
`NameExpr` it just calls `S->lookup(N.Name)`:

- If found, set `N.Ref = B` and mark `B->ReadFrom = true`.
- If not found and we're in REPL mode, auto-declare a `Var` (the
  REPL feeds it from the runtime workspace).
- Otherwise diagnose `undefined name 'X'`.

LHS positions go through `resolveLValue` (`Resolver.cpp:443`), which
is similar but rejects assignment to a function/builtin/class.

### 3.4 Call vs. Index — the same syntax, two meanings

`f(2)` could be a function call or a subscript into a variable
called `f`. The parser doesn't try to decide; it always emits a
`CallOrIndex` node. Sema decides in `Resolver::resolveCallee`
(`Resolver.cpp:497`):

- If the callee is a `NameExpr` whose binding is a Var/Param/Output/
  Global/Persistent → `CallKind::Index`.
- If it's a Function/Builtin/Class → `CallKind::Call`.
- If it's an unresolved bareword → tentatively `Call`, let later
  passes complain.
- If it's a `FieldAccess` whose base is *pinned* to a class with a
  matching method → `Call` (instance method). Same for static
  methods on a `Class` binding.
- If it's a non-name expression with a `FuncHandle` type → `Call`.
- Otherwise → `Index`.

**Class pinning** is the small piece of inference the resolver does
itself. After resolving an `AssignStmt`, the resolver inspects the
RHS:

- A direct `ClassName(args)` call → pin the LHS to that class.
- A binary op where either operand is pinned → pin the LHS to the
  same class (assumes operator overloads return the same class).

The "pinning" is really a `ClassDef *` stored on the binding. It
lets `obj.method(args)` dispatch statically without runtime type
inspection. Without it, every dot-call would have to be a
matlab.subscript. Read `Resolver.cpp:339-388` for the exact
heuristic; it's deliberately conservative.

### 3.5 Class methods

After resolving plain functions, the resolver walks each
`ClassDef::Methods` and resolves them as functions, with one twist
(`Resolver.cpp:149-177`):

- For a constructor `function obj = ClassName(args)`, pin the
  *output* binding to the class.
- For non-constructor methods, pin the *first parameter* (`obj`).
- For binary-op overloads (`plus`, `minus`, `lt`, …), also pin the
  second parameter.

These pins propagate through the body, so `obj.x` reads route via
the class's property table.

## 4. TypeInference, briefly

`lib/Sema/TypeInference.cpp` is the largest file in Sema (~1150
lines). Its job: visit every `Expr`, set `Expr::Ty`, and update
`Binding::InferredType`. The type lattice lives in
`include/matlab/Sema/Type.h`:

- `Dtype` — element type (Double, Single, Complex, IntN, UIntN,
  Logical, Char, Fixed).
- `Shape` — rank + per-dim extents, `-1` = dynamic.
- `Type::Kind` — Any, Array, StringArray, Cell, Struct, Object,
  FuncHandle.
- `FixedSpec` — wordlength / fraction length / signedness / overflow
  / rounding for `fi` types.

The pass is structural: an integer literal becomes
`Array(Double, scalar)`, a `BinaryOp` promotes via
`promoteDtype` + `broadcastShape`, a builtin like `zeros(3,4)` has a
hard-coded shape rule, etc. Builtins with type-specific behavior
(`size`, `cast`, the `fi` family) are special-cased here. When the
analysis can't be sure, it emits `Type::Any` rather than failing —
runtime generic dispatch handles the rest.

Type inference is also where `load`'s return type would be set to
`Struct` (see `docs/save_load_compat.md` §1.1) — it's the natural
place for "this builtin returns a known kind".

## 5. SemaDumper — `--dump-sema`

If you want to see what Sema produced, run the driver with
`--dump-sema`. The output is printed by `lib/Sema/SemaDumper.cpp`:
the AST in tree form, with each `NameExpr` annotated with its
binding kind and each `Expr` annotated with its inferred type:

```
AssignStmt
  Name A (var) [Array<Double>(2,3)]
  MatrixLit [Array<Double>(2,3)]
    IntLit 1 [Array<Double>(scalar)]
    ...
ExprStmt
  Call [void]
    Name save (builtin) [?]
    StrLit "out.mat" [String]
    Name A (var) [Array<Double>(2,3)]
```

This is the easiest way to debug a resolver / type-inference bug:
diff the dump against the AST you expect, find the first node that's
wrong, work back.

## 6. Adding a new builtin — the 90% recipe

The most common Sema-side change is adding a new builtin. The
checklist:

1. Add the name to `Resolver::registerBuiltins`
   (`lib/Sema/Resolver.cpp:22`). At this point `f(args)` parses and
   resolves cleanly but the lowerer rejects it.
2. If the result type is non-obvious (matrix shape, dtype change,
   struct return), add a case to `TypeInference` so dependents see
   the right type. Keep it conservative — `Any` is a valid answer.
3. Lower it in `lib/MLIR/Passes/LowerTensorOps.cpp` (or a more
   specific pass) by matching on the builtin name and emitting the
   runtime call.
4. Add the runtime function to `runtime/matlab_runtime.{c,h,hpp}`.
5. Write a `test/Run/<feature>.m` that exercises the round-trip.

For builtins whose argument *names* matter (like `save` / `load`
where strings are variable names, not values), step 1 alone isn't
enough — you need a small Resolver pass that re-interprets those
arguments. See the worked example in `docs/save_load_compat.md`
§1.1.

## 7. Where Sema *isn't*

A few things you might expect to be in Sema, but aren't:

- **Constant folding.** Lives in MLIR passes, not Sema. Sema's job
  is structural — it doesn't evaluate expressions.
- **Borrow / lifetime analysis.** Not applicable; MATLAB has
  reference semantics for handles and value semantics elsewhere,
  enforced by the runtime.
- **Overload resolution for operators.** Limited: the resolver pins
  classes so `obj + obj` can lower to `plus(obj, obj)`, but it
  doesn't pick between multiple `plus` definitions. There is at
  most one `plus` per class.
- **Macros / preprocessor.** None. MATLAB has none either.

## 8. Reading order if you're new

1. `include/matlab/Sema/Scope.h` — the data model. ~100 lines.
2. `lib/Sema/Scope.cpp` — the trivial implementations. ~80 lines.
3. `lib/Sema/Resolver.cpp:92-178` — top-level `resolve()`.
4. `lib/Sema/Resolver.cpp:184-309` — pre-pass.
5. `lib/Sema/Resolver.cpp:497-569` — `resolveCallee`, the most
   subtle part.
6. `include/matlab/Sema/Type.h` — the type lattice.
7. `lib/Sema/TypeInference.cpp` — skim the dispatch on `Expr::Kind`,
   then look up specific builtins as you encounter them.
8. `lib/Sema/SemaDumper.cpp` — useful when you start debugging.

That's all of Sema. It's small on purpose: it does just enough to
make lowering deterministic, and pushes everything else to MLIR or
the runtime.
