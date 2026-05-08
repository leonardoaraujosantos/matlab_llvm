# Runtime — `matlab_runtime.{c,h}`

This document inventories every feature the runtime currently exposes to
generated MATLAB code. The runtime is a single C translation unit
(`runtime/matlab_runtime.c`) plus a header (`runtime/matlab_runtime.h`).
The static `-emit-llvm` / `-emit-c` / `-emit-cpp` paths link against the
C runtime; the JIT (REPL + DAP) runs the same compiled symbols via
`mlir::ExecutionEngine`. Two stub mirrors —
[`runtime/matlab_runtime.py`](../runtime/matlab_runtime.py) and
[`runtime/matlab_runtime.ts`](../runtime/matlab_runtime.ts) — exist for
the Python and TypeScript transpile backends; they are kept in lockstep
with the C runtime for the operations the goldens exercise.

> **Source of truth.** The C header is the authoritative ABI. If the
> Python / TS mirrors disagree, the C version wins. The matrix below
> states which surface a feature is currently exposed on.

## Conventions

- All matrices are row-major `double` (`matlab_mat`) or row-major
  `double` real + imag arrays (`matlab_mat_c`). Both descriptors are
  reference-typed — the compiler passes pointers and never copies the
  payload unless an op semantically requires it.
- Reductions over a vector return a 1×1 matrix; over a matrix they
  return a 1×N row (one entry per column). `sum`, `mean`, `min`, `max`,
  `prod`, `std`, `var`, `median`, `any`, `all` all follow this rule.
- Polymorphic complex helpers (`fft`, `fftshift`, `conj`, `real`,
  `imag`, `angle`, `abs`) accept either a `matlab_mat *` or a
  `matlab_mat_c *` and dispatch on a magic word in the first 4 bytes
  of the descriptor (`MATLAB_MAT_C_MAGIC = 0xC0FFEE01`).
- Empty/invalid inputs return a 0×0 matrix rather than raising — the
  generated code can `isempty()`-check the result.
- Errors that can't be encoded structurally (singular matrix, non-SPD
  Cholesky, etc.) set the runtime error flag (`matlab_set_error`)
  and return a sentinel value. Generated code reads it with
  `matlab_check_error()` at the next try/catch boundary.

## Type system

| Descriptor | Purpose | Magic | Notes |
|---|---|---|---|
| `matlab_mat`     | Real row-major matrix.        | (untagged)            | Default for all double-typed values. |
| `matlab_mat_c`   | Complex (separate re/im).     | `0xC0FFEE01`          | Returned by FFT/conj/spectral shift. |
| `matlab_mat_i64` | `fi`-typed signed integer matrix.   | (descriptor tag)      | 64-bit lane only today. |
| `matlab_mat_u64` | `fi`-typed unsigned integer matrix. | (descriptor tag)      | 64-bit lane only today. |
| `matlab_struct`  | Named-field heterogeneous record.   | (descriptor tag)      | Children: f64, mat, struct. |
| `matlab_cell`    | 1-D heterogeneous container.        | (descriptor tag)      | Indexed via `{i}` / `(i)`. |
| `matlab_string`  | Heap-owned UTF-8 string descriptor. | (descriptor tag)      | Returned by `bin/hex/dec/sprintf/num2str`. |

3-D tensors have a magic (`MATLAB_MAT3_MAGIC = 0xC0FFEE03`) and a
`matlab_zeros3 / matlab_ones3` constructor pair, but the operator surface
on 3-D is currently sparse — most ops will reject 3-D inputs.

---

## Feature matrix

Legend: ✓ supported, — not implemented, ◇ partial (see notes).

| Feature                  | C runtime | Python mirror | TS mirror | MATLAB builtin name |
|---|---|---|---|---|
| **Constructors**         |           |               |           |                     |
| zeros / ones / eye       | ✓         | ✓             | ✓         | `zeros` / `ones` / `eye` |
| magic                    | ✓         | ✓             | ✓         | `magic` |
| rand / randn             | ✓         | ✓             | ✓         | `rand` / `randn` |
| range (`a:s:b`)          | ✓         | ✓             | ✓         | colon op |
| linspace                 | ✓         | ✓             | ✓         | `linspace` |
| repmat                   | ✓         | ✓             | ✓         | `repmat` |
| meshgrid / ndgrid        | ✓         | ✓             | —         | `meshgrid` / `ndgrid` (multi-return splitter) |
| **Elementwise binary**   |           |               |           |                     |
| `+ - .* ./ .^` (mm/ms/sm) | ✓        | ✓             | ✓         | operators |
| `<` `<=` `>` `>=` `==` `~=` | ✓     | ✓             | ✓         | operators |
| **Elementwise unary**    |           |               |           |                     |
| `-`, `exp`, `log`, `sqrt`, `abs` | ✓ | ✓             | ✓         | operators / builtins |
| `sin`/`cos`/`tan` + arc/hyperbolic | ✓ | ✓          | ✓         | `sin`, `cos`, `tan`, `asin`, ... |
| `log2`, `log10`, `sign`  | ✓         | ✓             | ✓         | builtins |
| `floor`/`ceil`/`round`/`fix` | ✓     | ✓             | ✓         | builtins |
| `mod`, `rem`, `atan2`    | ✓         | ✓             | ✓         | builtins |
| **Linear algebra**       |           |               |           |                     |
| `*` (matmul)             | ✓         | ✓             | ✓         | `mtimes` |
| `\` / `/` (mldivide / mrdivide) | ✓  | ✓             | ✓         | operators |
| `inv`, `det`             | ✓         | ✓             | ✓         | builtins |
| `transpose`, `ctranspose` (`'`, `.'`) | ✓ | ✓        | ✓         | operators |
| `diag`, `reshape`, `kron` | ✓        | ✓             | ✓         | builtins |
| `tril`, `triu`           | ✓         | ✓             | —         | builtins |
| `svd`, `eig` (one- and two-return) | ✓ | ✓ stub     | ✓ stub    | builtins |
| `qr` (`[Q,R]`), `lu` (`[L,U]`) | ✓   | ✓ stub        | ✓ stub    | builtins |
| `chol`, `pinv`, `norm`, `trace` | ✓  | ✓             | ✓         | builtins |
| `rank`, `cond` (from SVD) | ✓        | —             | —         | builtins |
| `null`, `orth` (from eig / QR) | ✓   | —             | —         | builtins |
| `matpow` (`A^n`)         | ✓         | ✓             | ✓         | `^` op |
| **Reductions**           |           |               |           |                     |
| `sum`, `prod`, `mean`    | ✓ (+`_dim`) | ✓           | ✓         | builtins |
| `min`, `max` (1- and 2-arg) | ✓ (+`_mm`) | ✓          | ✓         | builtins |
| `cumsum`, `cumprod`      | ✓ (+`_dim`) | ✓           | ✓         | builtins |
| `std`, `var`, `median`   | ✓         | ✓             | —         | builtins |
| `any`, `all`             | ✓         | ✓             | —         | builtins |
| `diff` (1st-order)       | ✓         | ✓             | —         | builtin |
| **Sort / set**           |           |               |           |                     |
| `sort`, `sortrows`, `unique` | ✓     | ✓             | ✓         | builtins |
| `ismember`, `setdiff`, `intersect`, `union` | ✓ | ✓ | ✓         | builtins |
| **Shape / predicates**   |           |               |           |                     |
| `size` (one- and two-return), `length`, `numel`, `ndims` | ✓ | ✓ | ✓ | builtins |
| `isempty`, `isequal`     | ✓         | ✓             | ✓         | builtins |
| `permute`, `squeeze`, `flip*`, `rot90` | ✓ | ✓        | ✓         | builtins |
| `find`, `sub2ind`, `ind2sub` | ✓     | ✓             | ✓         | builtins |
| **Subscripting / slicing** |          |              |           |                     |
| `A(i)`, `A(i,j)`, `A(rows,cols)`, end keyword | ✓ | ✓  | ✓        | parser-driven |
| `A(idx) = v` slice store (vec or scalar) | ✓ | ✓     | ✓         | parser-driven |
| Logical-mask indexing    | ✓         | ✓             | ✓         | parser-driven |
| `erase_rows` / `erase_cols` (`A(i,:)=[]`) | ✓ | ✓     | ✓         | parser-driven |
| `horzcat`, `vertcat`     | ✓         | ✓             | ✓         | bracket-cat |
| **Signal / FFT**         |           |               |           |                     |
| `fft`, `ifft`, `fft2`, `ifft2` (real or complex in) | ✓ | ✓ | ✓ stub | builtins |
| `fftshift`, `ifftshift`  | ✓         | ✓             | —         | builtins |
| `conv` (1-D), `conv2` (2-D) | ✓      | ✓             | ✓         | builtins |
| `xcorr` (full lag axis)  | ✓         | —             | —         | builtin |
| `filter` (DF-II-T IIR/FIR) | ✓       | ✓             | —         | builtin |
| Windows: `hamming`, `hann`, `blackman` | ✓ | —        | —         | builtins |
| **Polynomial**           |           |               |           |                     |
| `polyval` (Horner, elementwise) | ✓  | —             | —         | builtin |
| `polyfit` (least squares via normal eqs) | ✓ | —     | —         | builtin |
| `roots` (Durand-Kerner, complex out) | ✓ | —         | —         | builtin |
| **Numerical calculus**   |           |               |           |                     |
| `interp1` (linear, NaN out-of-range) | ✓ | —         | —         | builtin |
| `interp2` (bilinear, NaN out-of-range) | ✓ | —       | —         | builtin |
| `trapz(y)`, `trapz(x,y)` | ✓         | —             | —         | builtin |
| `cumtrapz` (running, leading 0) | ✓  | —             | —         | builtin |
| `gradient` (central diff + one-sided ends) | ✓ | —   | —         | builtin |
| **ODE / IVP solvers**    |           |               |           |                     |
| `ode45` (Dormand–Prince 5(4)) — scalar `y` | ✓ | ✓        | ✓         | `matlab_ode45_t`/`_y` (+ `_opts`, `_stats`) |
| `ode45` — **vector `y`** (system of ODEs) | ✓ | ✓         | ✓         | `matlab_ode45_v_t`/`_y` (+ `_opts`, `_stats`) |
| `ode23` (Bogacki–Shampine 3(2)) — scalar / vector | ✓ | ✓ | ✓        | `matlab_ode23_*` family |
| `odeset` fields: `RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, `Stats` | ✓ | ✓ | ✓ | accepted via `_opts` runtime entries; see [`ode.md`](ode.md) |
| User-time grid `tspan = [t0 t1 … tN]`; backward `[t1 t0]`; 3-return `[t, y, stats]` | ✓ | ✓ | ✓ | |
| Stiff (`ode15s` etc.), `Events`, `OutputFcn`, `pdepe` | — | — | —     | tracked in [`feature_status.md`](feature_status.md) |
| **Image processing**     |           |               |           |                     |
| `imfilter` (same-size conv2 wrapper) | ✓ | —         | —         | builtin |
| `padarray` (zero pad, [pre_r pre_c]) | ✓ | —         | —         | builtin |
| **Multirate signal**     |           |               |           |                     |
| `upsample`, `downsample` | ✓         | —             | —         | builtins |
| **Complex**              |           |               |           |                     |
| `complex_scalar`, `mat_c_from_real`, `mat_c_from_buf` | ✓ | ✓ stub | ✓ stub | constructors |
| `conj`, `real`, `imag`, `angle`, `abs` (polymorphic) | ✓ | ✓ | ✓ stub | builtins |
| `+ - .* ./` (`cc`), matmul (`cc`), transpose, ctranspose | ✓ | ✓ | ✓ stub | operators |
| **Strings**              |           |               |           |                     |
| `sprintf`, `num2str`, `str2double` | ✓ | ✓           | ✓         | builtins |
| `upper`, `lower`, `strtrim`, `strrep`, `strcat` | ✓ | ✓ | ✓     | builtins |
| `startsWith`, `endsWith`, `contains` | ✓ | ✓         | ✓         | builtins |
| `strlen`, `isstring`     | ✓         | ✓             | ✓         | builtins |
| **I/O**                  |           |               |           |                     |
| `disp`, `fprintf` (str, 1–4 f64 args) | ✓ | ✓        | ✓         | builtins |
| `input`                  | ✓         | ✓             | —         | builtin |
| `fopen`, `fclose`, `fgetl`, `feof`, `fread`, `fwrite` | ✓ | ◇ | — | builtins |
| `save`, `load`           | ✓         | ◇             | —         | builtins; see `save_load_compat.md` |
| **Structs / cells**      |           |               |           |                     |
| `struct(...)`, field set/get (f64, mat, child struct) | ✓ | ✓ | ✓ | parser-driven |
| `fieldnames`, `isstruct`, `isfield`, `rmfield` | ✓ | ✓ | ✓        | builtins |
| `cell(...)`, `{i}`, `(i)`, `numel`, `iscell`     | ✓ | ✓ | ✓        | parser-driven |
| **Globals / persistent** |           |               |           |                     |
| `global x` (f64)         | ✓         | ✓             | —         | declarations |
| `persistent x` (f64 or ptr)  | ✓     | ✓             | —         | declarations |
| **Try / catch**          |           |               |           |                     |
| Error flag set/get/clear | ✓         | —             | —         | `try`/`catch` lowering |
| **Parallel**             |           |               |           |                     |
| `parfor` dispatch + `+=` reduction | ✓ | ◇            | —         | `parfor` lowering |
| **Fixed-Point Designer (`fi`)** |    |               |           | see `docs/emit_fixed_point.md` |
| `fi`, `numerictype`, `fimath`, `fipref` constructors | ✓ | ◇ | —    | builtins |
| `int`, `storedInteger`, `storedIntegerToDouble`, `reinterpretcast`, `removefimath`, `setfimath` | ✓ | ◇ | — | builtins |
| `matlab_mat_i64` / `matlab_mat_u64` descriptors + slice/store | ✓ | — | — | parser-driven |
| Saturation + 5 rounding modes (Floor / Nearest / Zero / Convergent / Ceiling) | ✓ | — | — | controlled by `numerictype` flags |
| `bin(n)`, `hex(n)`, `dec(n)` renderers | ✓ | —        | —         | builtins |
| **Bitwise**              |           |               |           |                     |
| `bitand`, `bitor`, `bitxor`, `bitcmp`, `bitshift` | ✓ | ✓ | ✓     | builtins (also map to SV operators in HDL emit) |
| **Debug / DAP mirrors**  |           |               |           |                     |
| `matlab_dbg_frame_*`, `matlab_ws_*` (workspace mirror) | ✓ | — | — | inserted by `LowerTensorOps` |
| **Function handles**     |           |               |           |                     |
| `@name`, `@(x) ...`, captures, `(handle)(x)` | ✓ | ✓  | ✓         | parser-driven |

---

## Tier-1 builtins (added together with `conv`/`conv2`)

These are the seven groups of features added in the most recent runtime
expansion. All of them follow the existing column-wise reduction shape
where applicable.

### `filter(b, a, x)`
Direct-form II transposed implementation of the standard difference
equation `a(1)*y[n] = sum b[k]*x[n-k] - sum a[k+1]*y[n-k-1]`.

- `b`, `a` are vectors. `a(1)` (i.e. `a->data[0]`) must be non-zero.
- `x` may be a vector (output mirrors orientation) or a matrix (filtered
  column-wise). 
- The state register is reset between columns.
- Coefficients are normalised by `a(1)` once at entry, so callers may
  pass an unnormalised `a`.

### `any(A)` / `all(A)`
Logical reductions matching MATLAB's vector-vs-matrix shape rule.
Treats any non-zero element as true. `all([])` is `1`.

### `tril(A)` / `triu(A)`
Lower / upper triangular extraction (no offset argument yet).
Returns a fresh matrix; the input is not modified.

### `fftshift(A)` / `ifftshift(A)`
Circular shift that moves DC to the centre (`fftshift`) or back
(`ifftshift`). Polymorphic — accepts a `matlab_mat *` or a
`matlab_mat_c *` and always returns a `matlab_mat_c *` so chained
spectra survive. Shift amount per axis: `floor((d+1)/2)` forward,
`floor(d/2)` inverse. Singleton dims are left alone.

### `std(A)` / `var(A)`
Sample dispersion with `N-1` normalisation (matches MATLAB's default).
`std = sqrt(var)`. Vector → 1×1, matrix → 1×N column-wise.

### `median(A)`
`qsort`-and-pick on a scratch copy. `O(n log n)` per column. The result
shape mirrors `std`/`var`.

### `diff(A)`
First-order discrete differences. Vector input of length `n` → vector
of length `n-1` (orientation preserved); matrix input → `(m-1) x n`
matrix with differences down each column. Length < 2 returns 0×0.

### `meshgrid(x[, y])` / `ndgrid(x[, y])`
Coordinate matrices via two single-output runtime calls, picked up by
the multi-return splitter in `LowerTensorOps`. The one-arg form
re-uses `x` for both axes. `meshgrid` uses image (xy) ordering;
`ndgrid` uses array (ij) ordering.

```
[X,Y] = meshgrid([10 20 30], [1 2]);
%  X = [10 20 30; 10 20 30],  Y = [1 1 1; 2 2 2]

[Xn,Yn] = ndgrid([10 20 30], [1 2]);
%  Xn = [10 10; 20 20; 30 30], Yn = [1 2; 1 2; 1 2]
```

---

## Tier-2 builtins (signal / polynomial / numerical calculus)

These were added on top of Tier 1. End-to-end demo lives in
[`examples/tier2_demo.m`](../examples/tier2_demo.m).

### `xcorr(u, v)`
Full cross-correlation as a row vector of length `2L − 1` with
`L = max(numel(u), numel(v))` and lag-zero at index `L` (1-based).
The shorter input is implicitly zero-padded so the lag axis runs
`-(L-1) .. (L-1)` from index 1 to `2L-1`.

```
xcorr([1 2 3], [1 1])   ->  [0 1 3 5 3]    % lags -2..+2
xcorr([1 1 1], [1 1 1]) ->  [1 2 3 2 1]    % triangular autocorr
```

### `polyval(p, x)`
Horner evaluation of `a_n x^n + a_{n-1} x^{n-1} + ... + a_0` where
`p = [a_n, a_{n-1}, ..., a_0]` (MATLAB's highest-power-first order).
Applied elementwise on `x`; output mirrors `x`'s shape.

### `polyfit(x, y, n)`
Least-squares polynomial fit of degree `n` via the normal equations
on the Vandermonde matrix. Returns a row vector of length `n+1` in
the same coefficient order as `polyval`. Uses Gaussian elimination
with partial pivoting on the `(n+1) × (n+1)` normal matrix —
appropriate for the small degrees the runtime targets (typically
`n ≤ 8`). For higher degrees prefer a QR-based fit (not yet wired).

### `roots(p)`
Polynomial roots via Durand-Kerner (Weierstrass) iteration. Returns
a complex column vector of length `deg(p)`. The iteration starts
from `n` initial guesses on a spiral (`(0.4 + 0.9i)^k`) and
simultaneously refines them — converges in ~10 iterations for
well-conditioned polynomials. Leading-zero coefficients drop the
effective degree; trailing zeros become explicit roots at the origin.

```
roots([1 -5 6])   ->  [2; 3]            % up to ~1e-40 noise on im
roots([1  0 1])   ->  [-i; +i]          % complex conjugate pair
```

The output uses the complex descriptor (`matlab_mat_c *`), so even
all-real-root polynomials carry a (vanishing) imaginary part. Compare
with `abs()` if you need to filter numerical noise.

### `interp1(x, y, xi)`
1-D linear interpolation. `x` must be sorted ascending and the same
length as `y`. The query points `xi` can be any shape; the result
mirrors `xi`. Out-of-range `xi` produces `NaN` (MATLAB's default
extrapolation behaviour). Uses binary search per query, so each
lookup is `O(log n)`.

### `trapz(y)`, `trapz(x, y)`
Trapezoidal integration. The unit-spacing form sums
`0.5*(y[0]+y[end]) + sum(y[1..end-1])`. The two-arg form uses
`0.5 * sum((x[i+1]-x[i]) * (y[i]+y[i+1]))`. Vector input → 1×1;
matrix input → 1×N row (one integral per column).

### `cumtrapz(y)`
Running trapezoidal integral with leading zero. Same shape as input,
unit spacing.

### `gradient(f)`
Numerical gradient. Central differences in the interior, one-sided
at the endpoints (`g[0] = f[1] - f[0]`, `g[end] = f[end] - f[end-1]`).
For matrices, takes the gradient down each column (matching MATLAB's
single-output form). Same shape as input.

### Windows: `hamming(n)`, `hann(n)`, `blackman(n)`
Standard symmetric (non-periodic) DSP windows. Each returns a column
vector of length `n`. Coefficients:

| Window     | Formula |
|---|---|
| `hamming`  | `0.54 - 0.46 cos(2πk/(n-1))` |
| `hann`     | `0.5 - 0.5 cos(2πk/(n-1))` |
| `blackman` | `0.42 - 0.5 cos(2πk/(n-1)) + 0.08 cos(4πk/(n-1))` |

---

## Tier-3 builtins (SVD-derived linalg + image-processing wrappers)

These build on the existing SVD / EIG / QR / `conv2` primitives — none
implement new core numeric kernels. End-to-end demo lives in
[`examples/tier3_demo.m`](../examples/tier3_demo.m).

### `rank(A)`
Counts singular values larger than `max(m,n) * σ_max * eps`. Same
tolerance MATLAB uses by default. Returns a scalar.

### `cond(A)`
`σ_max / σ_min`. Returns `Inf` when the smallest singular value is
exactly zero (rank-deficient). For `m × n` matrices the singular
values are reported as a `min(m, n)`-long vector — `cond` consumes
the first and last entries directly.

### `null(A)`
Orthonormal basis for `ker(A)`, returned as an `n × (n - rank(A))`
matrix. Computed by eigendecomposing `A' * A` (which is symmetric and
PSD, fitting the symmetric Jacobi eig in the runtime) and selecting
the eigenvectors whose eigenvalues are below
`max-eig * n * eps`.

### `orth(A)`
Orthonormal basis for `col(A)`, returned as an `m × rank(A)` matrix.
For `m ≥ n`: columns 1…r of `qr(A).Q`. For `m < n`: eigenvectors of
`A * A'` with positive eigenvalues. **Caveat:** the QR path is
unpivoted Gram-Schmidt — for rank-deficient matrices where the first
`r` columns are *not* themselves linearly independent the result will
be wrong. Use SVD-based orthonormalisation for those cases (not yet
exposed as a separate builtin).

### `imfilter(A, h)`
Apply 2-D filter `h` to image `A` with output the same size as `A`.
Equivalent to `conv2(A, h)` cropped by `floor(size(h)/2)` on each
side. Boundary mode is implicit zero (no replicate / symmetric / etc.
options yet).

### `padarray(A, padsize)`
Zero-pad `A` by `padsize = [pre_rows pre_cols]` (or scalar applied to
both dims). Padding is symmetric — same amount before and after each
dim — so output is `(m + 2*pad_r) × (n + 2*pad_c)`.

### `interp2(X, Y, V, Xq, Yq)`
Bilinear interpolation. `X` is a sorted `1×N` row, `Y` is a sorted
`M×1` column, `V` is `M×N`. `Xq` and `Yq` must have the same shape;
output mirrors that shape. Out-of-range queries → `NaN`. Each query
costs `O(log M + log N)` via two binary searches.

### `upsample(x, n)` / `downsample(x, n)`
Multirate primitives. `upsample` inserts `n-1` zeros between samples
(output length `L * n`); `downsample` keeps every `n`-th sample
starting at index 1 (output length `ceil(L / n)`). 1-D vectors only —
matrix inputs are flattened. Output orientation mirrors the input.

---

## Scalar-arg overloads (auto-boxing)

Several Tier 1/2/3 builtins take ptr-typed (matrix) arguments where
MATLAB code idiomatically passes a scalar — `conv(u, gain)`,
`filter(b, 1, x)`, `polyval(p, 5)`, `interp1(x, y, 0.5)`, etc.
A 1×1 literal collapses to `f64` in the lowering pipeline, so these
calls used to miss the strict-typed dispatch slot.

The dispatcher now has a second-pass scalar-promotion fallback in
`LowerTensorOps.cpp`. After the strict match fails, it scans the
table again, accepting f64 operands in any `'p'` slot and recording
their indices. At call-site materialisation each f64 is wrapped via
`matlab_mat_from_scalar(f64) -> matlab_mat *`, then passed to the
runtime as a 1×1 matrix.

The fallback is gated by an explicit allowlist
(`AutoBoxNames` in `LowerTensorOps.cpp`) so calls like `mean(5.0)`
still flow through the scalar-math path (`matlab_mean_s`) instead of
becoming a 1×1 reduction. Current allowlist:

`conv`, `conv2`, `filter`, `xcorr`, `polyval`, `polyfit`, `interp1`,
`interp2`, `trapz`, `cumtrapz`, `imfilter`, `padarray`.

---

## How a builtin reaches the runtime

Adding a new MATLAB-visible builtin requires touching three places in
addition to writing the runtime function:

1. **`lib/Sema/Resolver.cpp`** — append the spelling to the
   `registerBuiltins()` initializer list so the name resolves at
   binding time instead of being treated as an undefined identifier.
2. **`lib/MLIR/Lowering.cpp`** — if the call returns a matrix descriptor,
   add the spelling to the `PtrRet` set so the `matlab.call_builtin`
   op carries `!llvm.ptr` instead of f64.
3. **`lib/MLIR/Passes/LowerTensorOps.cpp`** — add a `Spec` row to the
   dispatch table, e.g.
   `{"my_builtin", "matlab_my_builtin", 1, "pp"}` (1 = ptr return,
   `"pp"` = two ptr args). For overloaded forms (e.g. `mean(A)` vs
   `mean(A, dim)`) add one row per arity / type combination — the
   first matching row wins.
4. **Multi-return**: extend the splitter blocks at the top of
   `rewriteBuiltinCalls()` (eig/qr/lu/size/meshgrid/ndgrid pattern)
   and provide one runtime entry per output column.

The Python and TypeScript mirrors are best-effort — keep them in sync
when the goldens you care about run through those backends.

---

## Known gaps

- **`roots` numerical noise** — Durand-Kerner converges quickly but
  leaves ~1e-40-magnitude imaginary parts on real roots. Filter with
  `abs(imag(r)) < tol` or take `real(r)` if you need a clean
  real-valued result. `polyfit` uses normal equations rather than QR,
  so numerical conditioning degrades for high-degree fits (preferred
  upgrade: route through `qr_Q` / `qr_R`).
- **Eigendecomposition** symmetrises its input — `matlab_eig` only
  computes eigenvalues of `(A + A')/2`. `null`/`orth` therefore work
  via `A'A` / `AA'` (always symmetric). For non-symmetric problems
  (e.g. classical companion-matrix `roots`) use the dedicated builtin
  (`matlab_roots`) rather than building the companion and calling
  `eig`.
- **`orth` rank-deficient leading columns** — the `m ≥ n` path uses
  unpivoted Gram-Schmidt QR. If the first `r` columns of `A` are not
  themselves linearly independent, the truncation drops the wrong
  ones. Robust orth needs SVD-with-V (not yet wired) or column-pivoted
  QR (also not yet wired).
- **3-D tensor surface** is sparse — most ops reject inputs with the
  3-D magic. The `matlab_mat3` descriptor exists with `zeros3` /
  `ones3` / 3-D subscripting, but the elementwise / reduction surface
  hasn't been extended. Expanding 3-D is a structural change rather
  than a per-op addition.
- **Scalar promotion is allowlisted** — only the builtins in
  `AutoBoxNames` accept scalar args via auto-box. Adding a new
  ptr-only builtin? You'll need to add it to that list (or rely on
  callers writing `[v]`).
- **No `interp1` / `interp2` 'spline'** — only linear / bilinear is
  implemented. Cubic / spline methods aren't here yet.
- **`save` / `load`** support only a subset of the `.mat` format —
  see [`docs/save_load_compat.md`](save_load_compat.md).
