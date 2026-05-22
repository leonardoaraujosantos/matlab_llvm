# Arbitrary-Shape (N-D) Array Support — Roadmap

Scoped plan for letting `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug)
construct, index, and operate on arrays of **arbitrary shape** — first
arbitrary-depth 3-D (`M×N×P`, e.g. `300×200×4`), then true N-D
(`H×W×C×N` and beyond) — together with a concrete strategy for proving
that nothing already shipped regresses.

Status legend: ✅ shipped · 🟡 partial · 🔵 not started.

---

## 0. TL;DR

- **Arbitrary-depth 3-D already works** (`zeros(300,200,4)`, `A(:,:,4)`,
  `A(i,j,4)`, `size`/`numel`/`ndims`) — RAM-bound, no artificial cap.
- The cheap, high-value win is **Tier A**: make that arbitrary-depth 3-D
  *fully first-class* (`cat(3,…)` of N planes, 3-D-ness that survives
  expressions, depth-aware ops). ~½–1 day, low blast radius. **Tier A
  COMPLETE — A1–A5 shipped 2026-05-22 (467/467).** **Tier B COMPLETE too
  (B1–B6, 468/468)** — reshape/permute/squeeze/cat-dim/reductions/repmat.
- **True N-D (rank ≥ 4)** is **Tier C**: a new descriptor + pervasive
  plumbing. ~1–2 weeks, high blast radius. Only worth it with a concrete
  4-D use case (DL `NCHW`, RGB video, hyperspectral cubes).
- The **"nothing broke" guarantee** is a *diff-against-`main`* discipline:
  baseline the full suite (**468/468** after Tiers A+B), keep it green, add a
  depth/shape gating sweep, and cross-validate values against NumPy/PIL.

---

## 1. Current state (measured, not assumed)

`matlab_llvm` has exactly three dense descriptors (no rank-N type):

| Descriptor | Shape | Tag |
|---|---|---|
| `matlab_mat`   | 2-D real `M×N` (row-major) | *(none — leading `double*`)* |
| `matlab_mat_c` | 2-D complex `M×N` | `MATLAB_MAT_C_MAGIC` |
| `matlab_mat3`  | 3-D real `M×N×P` (slice-major) | `MATLAB_MAT3_MAGIC` |

(`runtime/runtime_internal.h`.)

**What works today** — verified empirically (all green):

| Probe | Result |
|---|---|
| `zeros(300,200,4)` → `size` | `300 200 4` |
| `zeros(4,4,5)` (depth 5) → `size(,3)` | `5` — *arbitrary depth* |
| `A(2,3,4)=77; A(2,3,4)` | `77` (element r/w, plane 4) |
| `A(:,:,4)=9; P=A(:,:,4)` | `8×8`, `9` (slice r/w, plane 4) |
| `zeros(1000,1000,3)` set `(999,999,3)` | `numel=3000000`, `42` (24 MB) |
| `zeros(3,3,4)` set planes 1–4 | reads `1 2 3 4` |

So **`300×200×4` is supported now** for construction (`zeros`/`ones(m,n,p)`),
element + whole-plane indexing, and `size`/`numel`/`ndims` — for *any*
depth, courtesy of the 3-D-indexing work (see
[`docs/image_toolbox_roadmap.md`](image_toolbox_roadmap.md) and the
`array3d_indexing` gating test).

**What does NOT work / is fragile** (also measured):

1. **Rank ≥ 4.** `zeros(2,2,2,2)` → *"zeros: 4 arguments unsupported"*.
   No descriptor above `matlab_mat3`.
2. ~~**`cat(3,…)` caps at 3 planes.**~~ **FIXED (A1, 2026-05-22).**
   `cat(3,a,b,c,d,…)` now builds depth N via `matlab_cat3_append`.
3. ~~**3-D tracking is syntactic, not type-based.**~~ **FIXED for
   expressions (A2, 2026-05-22).** `A = ones(5,5,4)*3; A(1,1,4)` compiles
   and reads `3`: `ThreeDBindings` population is now expression-aware
   (`exprIsThreeD` — arithmetic, unary, aliasing) and `BINARY_MS`/`_SM`
   preserve depth. *Still open — and blocked by a deeper gap:* 3-D returned
   from a **user function** (`A = makeVol(); A(i,j,k)`) loses its 3-D-ness.
   **Investigated 2026-05-22:** the analysis side is straightforward
   (interprocedural `funcReturns3D` over the callee body + Sema
   return-type propagation, prototyped and reverted), but it is blocked by
   a **broader pre-existing limitation**: a matrix-returning *user function*
   yields a `matlab.call` whose result stays `tensor<…>` (matching the func
   body) and is **never lowered to `ptr`** before `LowerTensorOps` — so the
   result can't feed *any* ptr-requiring op (`A(i,j)`, `A+1`, …), **2-D
   included**. `LowerUserCalls` keeps tensor sigs on purpose (bodies depend
   on per-tensor-shape inference, see its ~line 506), so the real fix is a
   func-boundary tensor→ptr lowering for matrix returns — a large, risky
   change deferred until prioritised. (Builtins dodge this: they return
   `ptr` from the runtime directly.)
4. ~~**Most elementwise/image ops are 2-D.**~~ **FIXED (A3, 2026-05-22).**
   Scalar/elementwise/unary binops (`+ - .* ./`, comparisons, `-A`) and the
   image arithmetic (`imadd`/`imsubtract`/`immultiply`/`imdivide`/
   `imabsdiff`/`imcomplement`/`imlincomb`) now loop over `depth` and return
   a mat3. Colour conversions are inherently depth-3 (RGB). *Still open:*
   `.^` (manual `epow_*`), matrix-mult/transpose/reductions on 3-D (Tier B).
5. ~~**`imread` of RGBA drops alpha.**~~ **FIXED (A4, 2026-05-22).** RGBA
   PNGs now decode to depth-4 (alpha kept); `png_encode` writes RGBA too.
   (Diverges from MATLAB's RGB + separate-alpha output — documented choice.)

**Size limits.** `mat3_alloc` does `calloc(rows*cols*depth, 8)` with all
dims `int64` — **no artificial cap; RAM-bound**. Latent (not hit in
practice): the element-count multiply is `int64` (overflow only past
~9.2×10¹⁸ elements) and `mat3_alloc`/`mat_alloc` don't check `calloc` for
`NULL`, so a genuine OOM segfaults rather than erroring.

---

## 2. Target: MATLAB's array model (what "faithful" means)

Real MATLAB has **one unified N-D array**; a matrix is just `ndims==2`.

- **Minimum 2 dims**, arbitrary maximum (memory-bound, not a small cap).
- **Trailing singleton dims are dropped**: `zeros(3,4,1)` is stored `3×4`
  (`ndims==2`); `size(A,3)` on a 2-D array returns `1`.
- **Max elements ≈ 2^48−1** on 64-bit (documented), RAM-bound in practice.
- `sparse` is **2-D only** (we already match this).
- Full N-D for numeric/complex/logical/char/cell/struct/`string`/`gpuArray`.

Fidelity items we must respect when extending:
- **Trailing-singleton drop** — `zeros(m,n,1)` should behave as 2-D
  (`ndims==2`, `size(,3)==1`). We currently keep a depth-1 `mat3` in some
  paths; tighten this.
- `size`/`numel`/`ndims`/`length`/`isrow`/`iscolumn`/`isvector`/`ismatrix`
  semantics on the new ranks.
- Error (not silently truncate) on shape mismatches in `cat`/`reshape`.

---

## 3. Tier A — arbitrary-depth 3-D, first-class ✅ *(A1–A5 all shipped 2026-05-22)*

Goal: make `300×200×4` (and any `M×N×P`) behave exactly like depth-3 RGB
everywhere it already half-works. **No new descriptor** — pure polish on
the shipped `matlab_mat3`.

| # | Surface | Change | Files |
|---|---|---|---|
| A1 ✅ | **N-ary `cat(3,…)`** | Shipped: `matlab_cat3_append(mat3, plane)` appends one plane; the cat arm folds `cat3_2(p1,p2)` then `cat3_append` per remaining plane, so `cat(3,p1,…,pN)` works for any N (no arity cap). | `runtime/matlab_runtime.cpp`, `lib/MLIR/Lowering.cpp` (cat arm), `lib/MLIR/Passes/LowerTensorOps.cpp` |
| A2 ✅ | **Type-based 3-D-ness** | Shipped option (a)-lite: `ThreeDBindings` population is now expression-aware via a recursive `exprIsThreeD` — 3-D-ness flows through elementwise/scalar arithmetic (`ones(5,5,4)*3`), unary ops, and aliasing (`B=A`). All routing gates stay keyed on `ThreeDBindings`, so complex/cell/struct never misroute. **Follow-on (blocked):** 3-D returned from a user function needs interprocedural propagation (prototyped) **and** a func-boundary tensor→ptr lowering for matrix returns (the real blocker — matrix-returning user fns can't be used/indexed even in 2-D today; see §1 item 3). | `lib/MLIR/Lowering.cpp` |
| A3 ✅ | **Depth-aware ops** | Shipped: scalar binops (`BINARY_MS`/`BINARY_SM`), elementwise `mat3⊙mat3` (`BINARY_MM` + `CMP_MM`), and unary (`UNARY_M` → `-A`, `exp`/`log`/`sin`/…) all get a `mat_is_3d` branch (flat-buffer alias → fresh mat3). Image arithmetic `imadd`/`imsubtract`/`immultiply`/`imdivide`/`imabsdiff`/`imcomplement`/`imlincomb` loop over depth (`img_binop` + the two bespoke ops). `exprIsThreeD` tracks these depth-preserving image ops (3-D iff their image arg is 3-D). Colour conversions were already 3-D (RGB). | `runtime/matlab_runtime.cpp`, `runtime/toolbox/images/runtime_images.cpp`, `lib/MLIR/Lowering.cpp` |
| A4 ✅ | **`imread` keep-alpha** | Shipped: the PNG decoder returns depth-4 for RGBA (colour type 6) instead of dropping alpha; `png_encode` writes depth-4/2 (RGBA/gray+alpha) too, so the roundtrip is self-contained. JPEG has no alpha. **Note:** real MATLAB returns RGB + a *separate* alpha output (`[X,~,a]=imread`); depth-4 is this project's documented arbitrary-depth choice (divergence). | `runtime/toolbox/images/runtime_images.cpp` |
| A5 ✅ | **Trailing-singleton drop** | Shipped: `matlab_zeros3`/`matlab_ones3` return a genuine 2-D `matlab_mat` when `p==1` (interops with all 2-D ops — transpose etc.); the cat arm returns the single operand for `cat(dim, a)`; `exprIsThreeD` treats `zeros(m,n,1)`-literal and 1-plane `cat(3,…)` as 2-D. | `runtime/matlab_runtime.cpp`, `lib/MLIR/Lowering.cpp` |

**Headline tests (shipped):** `test/Run/array_anyshape.m` — depth sweep
(`zeros(2,3,4)` per-plane store + read; `zeros(2,2,8)`), N-ary
`cat(3,p1..p4)`/`cat(3,p1..p5)` (depth 4/5), 3-D surviving expressions
(`ones(5,5,4)*3`, `10+ones(3,3,4)`, `B=A`), depth-aware ops (`A+B`, `-A`,
`A>B`, `imadd`, `imcomplement`), and trailing-singleton drop
(`zeros(3,4,1)'`, `cat(3,a)`). `test/Run/image_rgba_roundtrip.m` —
lossless RGBA PNG roundtrip (depth-4, alpha preserved). 467/467 regression.

Risk: low. The descriptor and indexing already support it; this removes
the arity cap + tracking fragility. **Tier A complete; next is Tier B
(reshape/permute/squeeze/cat-dim/reductions on 3-D).**

---

## 4. Tier B — 3-D manipulation ✅ *(B1–B6 all shipped 2026-05-22)*

Goal: the reshape/rearrange verbs, still on `matlab_mat3`.

| # | Surface | Status |
|---|---|---|
| B1 ✅ | `reshape(A, m, n, p)` | `matlab_reshape3` (flat row-major/slice-major reinterpret; p==1→2-D); `matlab_reshape` made mat3-aware for 3-D→2-D. Table `pfff`. |
| B2 ✅ | `permute` / `ipermute` | `matlab_permute` 3-D branch (general `out(o)=in` index map; result depth-1→2-D); new `matlab_ipermute` (inverse perm). `ipermute` registered as a builtin. |
| B3 ✅ | `squeeze` | `matlab_squeeze` 3-D branch: `m×n×1→m×n`, `1×n×p→n×p`, `m×1×p→m×p`, else copy. |
| B4 ✅ | `cat(1,…)` / `cat(2,…)` of 3-D | `matlab_vertcat`/`matlab_horzcat` 3-D branches (stack planes along rows/cols); cat arm folds them; `exprIsThreeD` marks `cat(1|2,…)` 3-D when an operand is 3-D. |
| B5 ✅ | `sum`/`mean`/`prod`/`max`/`min(…,3)` | `DIM_REDUCE` macro gains a `dim==3` mat3 branch (→ M×N). `max`/`min(A,[],dim)` newly wired via `matlab_max_dim3`/`min_dim3` (`ppf`) — also gives 2-D along-dim max/min. |
| B6 ✅ | `repmat(A, r, c, p)` | `matlab_repmat3` tiles into the 3rd dim (2-D or 3-D source; depth-1→2-D). Table `pfff`. |

**Element order:** the project's documented row-major (2-D) / slice-major
(3-D) convention, **not** MATLAB's column-major (consistent with the
existing 2-D `reshape`). **Test:** `test/Run/array_tierb.m` (468/468).

Risk: medium — touched reductions + reshape order. All new behaviour is
gated on `mat_is_3d`, so 2-D paths are byte-for-byte unchanged.

---

## 5. Tier C — true N-D arrays (rank ≥ 4) 🔵 *(large, ~1–2 weeks, pervasive)*

Goal: first-class rank-N arrays — `H×W×C×N` batches, RGB video, cubes.
This is the only path to MATLAB-equivalent N-D. **High blast radius.**

### Design

A new descriptor (generalising — not replacing — `matlab_mat`/`mat3`):

```c
#define MATLAB_MATN_MAGIC 0xC0FFEE04u
struct matlab_matN {
    uint32_t magic;        /* MATLAB_MATN_MAGIC */
    uint32_t ndims;        /* >= 2; trailing singletons dropped */
    int64_t *dims;         /* length ndims */
    int64_t *strides;      /* element strides, column-major-faithful */
    double  *data;         /* prod(dims) doubles */
};
```

Keep `matlab_mat` (2-D) and `matlab_mat3` (3-D) as the fast common path;
`matlab_matN` covers rank ≥ 4 (or unify everything behind `matN` later).
Add `mat_is_nd(p)` alongside `mat_is_3d`/`mat_is_complex`.

### Work breakdown

| Layer | Work |
|---|---|
| **Runtime** | `matN_alloc`/free; linear-offset from an index vector; `matlab_zerosN`/`onesN`/`randN`; `size`/`numel`/`ndims`/`length` made `matN`-aware (extend the existing magic-tag dispatch); generic N-D elementwise apply; `reshape`/`permute`/`squeeze`/`cat(dim,…)`/reductions over N-D. |
| **Lowering** | `zeros(d1,…,dn)` for n>3 → `matlab_zerosN` (variadic); N-D subscript read/store `A(i1,…,in)` and slice forms; carry an N-D type through `matlab.call_builtin`. |
| **Sema / TypeInference** | An N-D array type (rank + per-dim sizes where static); `ndims`/`size` constant-fold; subscript arity checks. |
| **REPL / Debug (DAP)** | Workspace mirror + variable view must render an N-D shape (currently 2-D/3-D aware). |
| **Consumers** | Audit every op that special-cases `mat_is_3d`; decide per-op whether to support N-D, project to 2-D, or error cleanly. |

### Sequencing
C1 descriptor + alloc + `size`/`numel`/`ndims` → C2 `zeros(…n)` +
N-D subscript → C3 `reshape`/`permute`/`squeeze`/`cat` → C4 reductions +
generic elementwise → C5 REPL/DAP rendering → C6 op audit.

Each step gated by the full regression staying green (see §6).

---

## 6. Testing strategy — proving "nothing broke"

The repo already has the harness; the rule is **diff against `main`**
(per the project's verification discipline).

### 6.1 Regression baseline + gate
1. On `main`: run `test/Run/run_tests.sh` (or the fast `/tmp/fastrun.sh`) →
   record the count (**`PASS=468 FAIL=0`** after Tiers A+B; was 465).
2. After every change: re-run, **require the same or higher PASS, zero
   FAIL**. A drop is a regression — investigate before proceeding.
3. Also keep the **strict no-C-cast lane** green: any modernised TU
   (`runtime_images.cpp`, etc.) must still compile with
   `-Werror=old-style-cast`.
4. Full `cmake --build build` clean (all toolboxes link together).

### 6.2 Existing guards that MUST stay green
These already exercise 2-D and 3-D paths and are the front line:
`array3d_indexing`, `array_anyshape` (Tier A), `array_tierb` (Tier B),
`image_rgba_roundtrip` (A4), `images_t1_io … images_t6_transforms`,
`channel_split`, `image_png_roundtrip`, plus every numeric/linalg test
that builds matrices. A shape change that breaks any of these is rejected.

### 6.3 New gating tests for the change (write-to-`/tmp`-then-read style)
- **Depth sweep (Tier A):** `zeros(m,n,N)` for `N = 1,2,4,8` → check
  `size(,3)`, `numel`, `ndims`, and read-back of *every* plane.
- **N-ary cat:** `cat(3,a,b,c,d)` → `size(,3)==4` + per-plane values;
  `cat(3,a)` → 2-D (trailing-singleton drop).
- **Tracking survives expressions:** `B = ones(5,5,4)*3; B(1,1,4)` compiles
  and reads `3`.
- **2-D fallback unchanged:** `zeros(m,n)` and depth-3 RGB pipelines behave
  exactly as before (guards the `mat_is_3d` dispatch).
- **Tier C:** `zeros(2,3,4,5)` → `ndims==4`, `numel==120`; N-D subscript
  round-trip; `reshape`/`permute`/`squeeze` identities; `cat(4,…)`.
- **Error paths:** shape-mismatched `cat`/`reshape` error rather than
  silently truncate.

### 6.4 Ground-truth value checks
Don't just check shapes — check **values** against an oracle, as done for
the PNG/JPEG decoders:
- Generate reference arrays with **NumPy** (`np.zeros`, `reshape`,
  `transpose`, `concatenate`) and compare element-by-element.
- For anything image-shaped, cross-check with **PIL**.
- Where feasible, compare against **real MATLAB** output for the same
  snippet (shape + a few sampled elements).

### 6.5 Memory / robustness
- Add a `NULL`-check after `calloc` in `mat_alloc`/`mat3_alloc`/`matN_alloc`
  (and any new allocator) so OOM errors cleanly instead of segfaulting —
  test with a deliberately huge `zeros(...)` that should fail gracefully.
- Sanity-check a large-but-reasonable allocation (e.g. `1000×1000×3`,
  already verified) to confirm no `int64`→`size_t` surprises.

### 6.6 REPL + Debug lanes (for Tier C)
- REPL: `A = zeros(2,2,2,2); size(A)` displays the right shape line-by-line
  (cross-line type pinning is lost in the REPL, so prefer runtime dispatch).
- DAP: the workspace variable view renders the N-D shape without crashing.

---

## 7. Carve-outs / non-goals

- **Sparse N-D** — MATLAB itself is 2-D-only here; not in scope.
- **N-D for every toolbox op** — Tier C ships the *array machinery*;
  individual ops opt in (or project-to-2-D / error) per the §5 audit. No
  promise that, say, `regionprops` understands a 5-D input.
- **GPU / distributed / tall arrays.**
- **Non-`double` N-D** (int/uint/logical N-D) beyond what the existing
  typed-int lanes already do — follow-on.
- Matching MATLAB's exact memory-layout (column-major) at the bit level —
  we are row-major (2-D) / slice-major (3-D); document the chosen N-D
  linearisation rather than guaranteeing `.mat`-identical strides.

---

## 8. Dependency / sequencing summary

```
Tier A (arbitrary-depth 3-D first-class)   ── ✅ COMPLETE (A1–A5, 2026-05-22)
   └─ N-ary cat · type-based 3-D · depth-aware ops · singleton-drop · RGBA
Tier B (3-D reshape/permute/squeeze/cat-dim/reductions)  ── ✅ COMPLETE (B1–B6, 2026-05-22)
   └─ reshape3 · permute/ipermute · squeeze · cat(1|2) · dim-3 reductions · repmat3
Tier C (true rank-N)                        ── large, gated on a real use case
   └─ new matN descriptor → zeros(…n)+subscript → reshape/permute/cat
      → reductions/elementwise → REPL/DAP → per-op audit
```

**Status:** **Tiers A and B are complete** — arbitrary-depth 3-D is now
first-class (construction, indexing, depth-aware ops, RGBA I/O) and has the
full set of manipulation verbs (reshape/permute/squeeze/cat/reductions/
repmat). Defer **Tier C** (true rank-N, `matlab_matN`) until a concrete
4-D workload (deep-learning batches, video, hyperspectral) justifies the
pervasive change. Every tier is gated by the §6 regression discipline:
**baseline `main`, stay at 468/468, add the shape sweep, validate values
against NumPy/MATLAB.**
