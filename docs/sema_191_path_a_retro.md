# #191 Path A — retrospective: what we tried, what shipped, what's a dead end

Tracking epic [#191](https://github.com/leonardoaraujosantos/matlab_llvm/issues/191)
("Sema completeness — toward precise whole-program type inference") was worked
end-to-end as **Path A (full)**: finish P2.2 → complete P3 desynth → land P5
(retire the late MLIR monomorphiser). Closed 2026-06-12. This is the honest
record of the attempt so the dead end isn't re-walked.

Branch: `sema/p5-scaffolding-191` — **12 commits, every one gated against all 5
lanes** (Run / golden / emit-c / emit-python / emit-ts), which never moved off
**763/0 · 91/0 · 328/0 · 268/0 · 233/0**.

## Method

Incremental, measurement-first. Each phase re-added a probe before touching
logic, every change ran the full gate before commit, and the riskiest piece
(P5) was built behind `MATLAB_LLVM_NO_LATE_MONO` so the default build stayed
inert by construction and progress was measured by the *bypass* failure count.

## What shipped (permanent)

### P2.2 — computed-arg / builtin-result typing (comprehensive)
The `MATLAB_LLVM_PROBE_ANYBUILTIN` probe ranked the builtins that returned
`Any` despite typed args (the "wall (a)" arg-poisoning). Typed:
- **Reductions** `sum/prod/mean/median/std/var`: matrix → `1×N` row; vector/scalar
  → **`1×1` matrix** (NOT a bare scalar — see the box/unbox trap below).
- **Shape-preserving** `sign/log10/log2/sinh/cosh/tanh` (float elementwise group,
  inherits the dlarray class-pinned back-off), `bitshift` (type-preserving),
  `reshape(x,m,n)` (foldable dims → `M×N`).
- **`norm`** → `1×1` matrix.

Single-arg `min/max` stays `Any` on purpose (it feeds the `[v,i]=min(x)`
multi-output index path).

**Trap learned the hard way:** reduction results lower to **boxed matrix
pointers**. Typing a scalar result as an *unboxed* `TC.scalar` makes the
function-return / slot type `f64` while the body value is a `ptr` → broke
`gpu_func_axpy` / `multiret_tilde` / `bode_margin_tf`. Typing it as a `1×1`
*matrix* (Matrix rank, ptr-shaped) matches the lowering and is a pure-Sema win.

### P3 — de-synthesize class operator dispatch (complete for the control classes)
`tf` / `ss` / `zpk` / `pid` / `frd` are now fully de-synthesized at Sema —
operator overloads, scalar-mixing, **scalar-LHS `X op obj`**, and **unary
`-obj`**. The `test/DesynthProbe` contract (in ctest) enforces zero
migrated-class synthesis fires, with a `liveness_unmigrated.m` fixture so the
probe can't pass vacuously.

**Stale blocker debunked:** the design doc claimed scalar-LHS `2*G` needed a
constructor-call method base in the lowering that "segfaults." Not true anymore
— the dispatch lowering already recovers the class from the base's `object<>`
type, so scalar-LHS desynth works and the migrated-class scalar-LHS synthesis
fires dropped 6 → 0.

### P5 prerequisites (landed, inert, independently useful)
`walkClassCallsWithCaller` (surfaces ctor + method calls the function-mono
walker skips), `ArgTypes` population for class ctor/method calls (a real
correctness fix), `MATLAB_LLVM_PROBE_CLASSMONO`, and the
`CallOrIndex::MonoSpec` / `Function::EmitSymbol` AST fields.

## What's a dead end — P5 via Sema-time class-mono

The full Sema-time class monomorphizer was built (`runClassMonomorphize`:
bucket class calls by `(method, signature)` → clone per signature → append to
`ClassDef::Methods` → `EmitSymbol`/`MonoSpec` → stamp → fixpoint; plus a lowerer
chokepoint dispatching `MonoSpec` calls to the clone symbol). **It works
mechanically** — clones are created and dispatched — but:

- bypass failure set **69 → 194** unscoped (miscompiles dlarray's autodiff tape
  and every runtime-backed class),
- **69 → 70** even scoped to `{tf,ss,zpk,pid,frd}` (recovers **zero**,
  regresses one).

**Root cause — the tensor/ptr boundary.** `tf`/`ss` methods operate on
polynomial-coefficient *matrices*. Matrix-parameter functions must be
monomorphized at the **post-`LowerTensorOps` ptr level** — which is exactly
what the late MLIR `runMonomorphiseUserCalls` does. An AST-level clone keeps
tensor-typed params the pipeline can't bridge to the body's ptr operands (the
same reason `stampSignatureTypes` already *defers* non-scalar Array params to
the late pass), and nested ctor/method calls inside clone bodies carry
non-concrete args that never specialise. Scalar-signature class calls do
monomorphize; the matrix-heavy control/mpc/dsp fixtures — the bulk of the 69 —
cannot.

**Conclusion:** P5's literal goal ("delete `runMonomorphiseUserCalls`, lock the
bypass set at 0 via Sema-time mono") is **not reachable through AST-level class
cloning**. The late MLIR mono is fundamentally required for matrix-param class
dispatch. Replacing it would mean reimplementing class-mono at the MLIR level
(post-`LowerTensorOps`) — i.e. duplicating the existing pass for ~no gain. P5
was closed as **superseded**. Full write-up: `docs/sema_p3_dispatch_desynth.md`
(P5 section). Engine code was reverted; the prerequisite infra above remains.

### Bug found while building the engine (for future reference)
Do **not** `push_back` to `ClassDef::Methods` *during* `walkClassCallsWithCaller`
— the walker iterates `Methods`, so appending mid-walk invalidates the
iteration and silently drops most clones. Collect sites during the walk, create
clones after.

## Net assessment

The Sema layer is meaningfully more precise (P2.2 removed hundreds of `Any`
arg-sites), the control-class dispatch is first-class and CI-protected (P3), and
the one architecturally-blocked sub-goal (P5) is now *mapped and documented*
rather than an open trap. The reverted engine wasn't wasted — it converted "we
should be able to delete the late mono" from an assumption into a proven fact.
