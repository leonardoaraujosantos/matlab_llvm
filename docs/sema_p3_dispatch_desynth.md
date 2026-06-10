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

## Validation per increment

Full local gate (Run / golden / Repl / DAP / repl_sweep / emit-c/cpp/python/ts)
+ the class's own toolbox examples, then CI Full ctest gate green before merge.
A `test/Sema` golden per migrated class asserts the rewritten AST (operator →
explicit method `CallOrIndex`).
