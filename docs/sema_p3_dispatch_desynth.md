# #191 P3 — De-synthesize class operator/method dispatch

**Status:** design (implementation incremental, one class per PR, flag-gated).
**Risk:** HIGH — control-system classdef operators (`tf`/`ss`/`zpk`/`pid`/`frd`)
and `OptimizationExpression` are pervasive; a regression here breaks whole
toolboxes. Every increment runs the full gate and merges only on CI-green.

## Problem

Operator overloads, scalar-mixing, and some method calls are **synthesized
during lowering**, so Sema never sees them as real calls. That defeats
inter-procedural inference (P2) and the monomorphizer (P5): a `G + H` on `tf`
objects is, at the AST level, a plain `BinaryOpExpr` — the `tf.plus` call only
exists after lowering rewrites it.

### The three synthesis sites (current `lib/MLIR/Lowering.cpp`)

1. **Operator overloads — `~5819–5915`.** A `BinaryOpExpr` where either operand
   is class-pinned (`pinnedFromExpr`) is rewritten to
   `matlab.call @Owner__<plus|minus|mtimes|mrdivide|mldivide|mpower|times|rdivide|ldivide|eq|ne|lt|le|gt|ge>(LHS, RHS)`.
   `Owner` is the first class up the `Super` chain that defines the method.
2. **Scalar-mixing — `~5874–5895`.** When one operand is a class instance and
   the other is a non-class scalar/matrix, the non-class operand is boxed into a
   one-arg `Owner(value)` constructor call (`G + 2` → `G + tf(2)`). Restricted to
   `tf`/`ss`/`zpk`/`pid`/`frd`/`OptimizationExpression` (classes with a 1-arg
   constant constructor).
3. **Instance-method dispatch — single-return `~9033`/multi-return `~3130`.**
   `obj.method(args)` and `[a,b]=meth(obj,…)` route to `Class__method` with the
   object passed as the first parameter, walking `Super` for inheritance.

All three key off `Binding::PinnedClass` / `pinnedFromExpr`, which the Resolver
sets and Sema sees — so the rewrite can be done at Sema time.

## Design — a Sema-time AST rewrite

Add `lib/Sema/DispatchDesynth.{h,cpp}`: an AST pass that rewrites, in place,
operator/method dispatch on class instances into explicit `CallOrIndex` nodes
to the method (and ctor-boxing for scalar mixing), so every class call is a
real AST node visible to TypeInference and the monomorphizer.

- **When:** after Resolver + a first TypeInference pass (so `object<Class>` /
  `PinnedClass` are known), then re-run TypeInference on the rewritten AST. The
  rewrite uses the same `Owner`-up-the-`Super`-chain method lookup the lowering
  does today.
- **Rewrites:**
  - `BinaryOpExpr(a op b)` with a class-pinned operand → `CallOrIndex(NameExpr("<opmethod>"), [a, b])` resolved to the `Owner` method (the call carries the resolved method binding so lowering emits `Owner__opmethod` without re-deriving).
  - scalar-mixing → wrap the non-class operand as `CallOrIndex(NameExpr(Owner), [value])`.
  - `obj.method(args)` is already a `CallOrIndex` with a `FieldAccess` callee; normalize it to the function-style method call node so it flows through one path.
- **Flag:** `MATLAB_LLVM_SEMA_DISPATCH` (env) gates the pass per class via an
  allow-list, default empty (no behavior change). The lowering synthesis stays
  as the fallback: once the AST holds a real `CallOrIndex`, the lowering
  `BinaryOpExpr`-on-object path no longer matches, so there is no double-emit.
- **Acceptance (per the issue):** with `MATLAB_LLVM_PROBE_LATE_MONO`, no
  synthesized `matlab.call` to a class ctor/method remains for the migrated
  class; all lanes green.

## Incremental rollout (one PR each, full-gate + CI-green between)

1. Pass scaffolding + operator rewrite, allow-list = `{OptimizationExpression}`
   (simplest: arithmetic only, no inheritance, well-covered by `gads` examples).
2. `tf`, then `ss`, then `zpk`/`pid`/`frd` — one class per PR, adding
   scalar-mixing + method dispatch coverage as needed.
3. When the allow-list covers all synthesized classes and the
   `MATLAB_LLVM_PROBE_LATE_MONO` probe is clean, remove the lowering synthesis
   sites and the flag.
4. P5 (retire the late monomorphiser) becomes possible once dispatch is a
   first-class AST node everywhere.

## Feasibility check (verified)

`a = s2 * s1` and `b = mtimes(s2, s1)` on `ss` objects **both lower to
`matlab.call @ss__mtimes(ptr, ptr)`** — so the rewrite (`op` → method
`CallOrIndex`) reuses the existing function-style method dispatch; no new
lowering is needed.

**Caveat (must fix in PR 1):** the operator-synthesis path sets the result type
to `ptr` (object), but the function-style method call lowers to result type
`none` — i.e. it loses the `object<Class>` result type, which breaks a
downstream typed field access (`b.a`). So the rewrite must be paired with
**function-style method-call result typing**: when a `CallOrIndex` with a
NameExpr callee resolves to a method via a class-pinned `arg0`, return that
method's `OutputRefs[0]->InferredType` (the analog of P1.1, which covered the
`obj.method()` FieldAccess-callee form). Without this the rewrite is a
regression, not a no-op.

## Step-2 implementation status (validated)

`DispatchDesynth` is implemented (`lib/Sema/DispatchDesynth.{h,cpp}`, compiles,
behavior-neutral with an empty allow-list). End-to-end validation with the pass
temporarily wired into the `-repl`/JIT Sema site, allow-list `{OptimizationExpression}`:

- **Single-unit (AOT-shape) input** `x=optimvar(); y=optimvar(); e=x+2*y;` →
  the pass **fires** (`rewrote 2 ops`) and the result is correct. ✓ The rewrite
  works when operand object types are present on `Expr->Ty` in the same unit.
- **Cross-turn `-repl`** (`x`/`y` from earlier turns) → the pass is a **no-op**
  (`rewrote 0 ops`): per-turn compilation doesn't carry `object<Class>` on the
  operand `Expr->Ty` (only the binding's `PinnedClass` is re-pinned cross-turn).
  Synthesis handles it as before — `examples/optim/problem_based_lp.m` produces
  **identical** output (LP x=3,y=1 · QP a=1,b=1 · MILP i=3,j=2) either way.

So the rewrite is **safe everywhere**: it fires where operand types are known
(whole-program / AOT — exactly where P5 monomorphization runs) and falls back to
the identical synthesis where they aren't. A later enhancement can key the pass
off `PinnedClass` too (matching `pinnedFromExpr`) to also fire cross-turn.

### Remaining for the step-2 PR
- Wire `desynthDispatch` + a re-`TypeInference::run` after the first
  Resolver+TypeInference at each compile entry point (5 `R.resolve(*TU)` /
  `Inf.run(*TU)` sites in tools/matlabc/main.cpp — `runReplInput` @1166 and
  @4284, the breakpoint-merge @12934, and the monomorphize re-Sema lambdas
  @12974/@13056). A small shared helper avoids duplication. Re-running
  TypeInference (not Resolver) suffices — synthesized `NameExpr` Refs are set
  manually (`CD->Self`, a `BindingKind::Class` binding) and `Resolved=Call`.
- `test/Sema` golden of the rewritten AST for an OptimizationExpression op.
- Full gate; then add `tf`/`ss`/… one class per PR.

## Validation per increment

Full local gate (Run / golden / Repl / DAP / repl_sweep / emit-c/cpp/python/ts)
+ the class's own toolbox examples, then CI Full ctest gate green before merge.
A `test/Sema` golden per migrated class asserts the rewritten AST (operator →
explicit method `CallOrIndex`).

## `MATLAB_LLVM_PROBE_LATE_MONO` (implemented) + what it revealed

The probe is wired at the operator-synthesis site in `lib/MLIR/Lowering.cpp`:
when the env var is set, every class operator that reaches the lowering
synthesis fallback emits `[late-mono-probe] op-synth <Class>::<method>
lhs_obj=<0|1> rhs_obj=<0|1>` to stderr (gated off the var, zero behaviour
change otherwise). For a fully-migrated class this site must never fire with
the object on the LHS; a fire names a gap. `test/DesynthProbe/` locks the
contract in (asserts no migrated-class `lhs_obj=1` fire).

Sweeping the probe over `examples/` + `test/Run/` established:

- **The 6 migrated classes are first-class.** After the `PinnedClass` fix
  below, `tf`/`ss`/`zpk`/`pid`/`frd`/`Vec2` produce **zero** object-on-LHS
  synthesis fires. The only residual migrated-class fires are the **scalar-LHS**
  `k * G` form (`lhs_obj=0`), the separately-tracked deferred gap that needs a
  constructor-call method base in the lowering.
- **Full removal of the synthesis sites is NOT reachable by the allow-list
  rollout alone.** `dlarray` (the autodiff workhorse, ~500 operator sites),
  `OptimizationExpression` (whose only gate path is per-turn `-repl`, where
  desynth is a structural no-op), and arbitrary user classdefs all legitimately
  reach synthesis and are not migrated. The lowering synthesis therefore stays
  as the general fallback; deleting it would require generalising desynth to
  *every* class (incl. the `dlarray` autodiff path) plus cross-turn `-repl`
  firing — large and high blast radius, deferred.

### PinnedClass coverage fix (implemented)

`objClassOf` (the desynth operand-class test) originally keyed only off an
`object<Class>` `Expr->Ty`. A value carrying the class only via the binding's
`PinnedClass` — e.g. `Cz = c2d(G, Ts)` is typed `any` but pinned `tf`, or a
cross-turn `-repl` re-pin — was missed, so `Cz * Cz` fell through to synthesis
even though the synthesis path (`pinnedFromExpr`) keys off exactly that pin.
`objClassOf` now mirrors `pinnedFromExpr`: a `NameExpr` whose `Ref->PinnedClass`
is set resolves to that class. Because such an operand's result type isn't known
until the rewritten method call is re-typed, `p3DesynthDispatch` now iterates
desynth + `TypeInference::run` to a fixpoint, so a chained `a*b + c` on
pinned-only operands rewrites fully across passes.
