# Complex numbers and FFT

A tour of the complex-number runtime and the pure-C FFT — both
shipping together so DSP-style workflows compile and run end-to-end
without an external dependency like FFTW.

## Surface

```matlab
>> a = 2i                   % imaginary literal
a = 2i

>> b = 3 + 4i               % mixed real + complex: b becomes complex
b = 3 + 4i

>> abs(b)                   % magnitude (real result)
ans = 5

>> angle(b)                 % argument in radians
ans = 0.927295

>> conj(b)
ans = 3 - 4i

>> real(b), imag(b)
ans = 3
ans = 4

>> fft([1 2 3 4])
ans = 10 + 0i    -2 + 2i    -2 + 0i    -2 - 2i

>> ifft(fft([1 2 3 4]))     % round-trip recovers the input
ans = 1    2    3    4
```

All of the above work from both the REPL and from AOT-compiled `.m`
scripts (via `matlabc -emit-llvm`, `-emit-c`, `-emit-cpp`).

## Representation

At runtime, complex values are `matlab_mat_c *` pointers:

```c
struct matlab_mat_c {
    uint32_t magic;   // MATLAB_MAT_C_MAGIC = 0xC0FFEE01
    uint32_t _pad;
    double  *re;      // row-major, rows*cols doubles
    double  *im;      // row-major, rows*cols doubles
    int64_t  rows;
    int64_t  cols;
};
```

Scalars are 1×1 matrices — same auto-boxing trick the real runtime
uses so the compiler's MLIR only has to plumb one SSA type
(`!llvm.ptr`) for every complex value.

The **magic marker at byte 0** discriminates at runtime between
`matlab_mat *` (real, starts with a heap pointer) and `matlab_mat_c *`
(complex, starts with `0xC0FFEE01`). Polymorphic runtime entries
(`matlab_disp_mat`, `matlab_add_mm`, `matlab_fft_c`, …) check the
marker and route accordingly.

Separate real/imag planes (rather than interleaved `{re, im}` pairs)
keep the SIMD-friendly stride-1 loop shape on the real-only fast path
and let arithmetic kernels share the scalar math loops between real
and complex matrices.

## Mixed real + complex arithmetic

Binary ops (`+ - .* ./`) dispatch at runtime: if either operand carries
the magic marker, both sides are promoted to complex via
`matlab_mat_c_from_real` and the `_cc` variant runs:

```matlab
>> 3 + 4i                 % f64 + matlab_mat_c* scalar
ans = 3 + 4i

>> [1 2 3] + 2i           % real vector + complex scalar (broadcast)
ans = 1 + 2i    2 + 2i    3 + 2i
```

The real fast path stays branch-free in the hot loop — it just has
an extra one-word check at the call entry. `epow` (`.^`) is the one
hole in the complex coverage; real `.^` still works, complex `.^` is
left for a follow-up since the runtime doesn't implement
`matlab_epow_cc` (complex exponentiation wants a principal-branch
convention the other ops don't need).

## FFT — pure-C Cooley-Tukey

Two paths, both operating on a `matlab_mat_c *`:

- **Power-of-two `N`** — iterative radix-2 DIT with bit-reversal
  permutation. `O(N log N)`. Exact modulo FP rounding for any
  power-of-two `N`; MATLAB's built-in FFT (backed by FFTW) will
  differ only in the last few ULPs.

- **General `N`** — Bluestein's algorithm. Expresses `DFT(x)` as a
  convolution multiplied by chirp factors, and convolves via a
  radix-2 FFT at the next power of two ≥ `2N-1`. `O(N log N)`
  asymptotically. A few× slower than direct radix-2 and
  noticeably less accurate (rounding accumulates through the
  chirp construction); matches MATLAB to ~1e-12 on typical inputs.

```c
matlab_mat_c *matlab_fft_c(void *A);   // polymorphic — real or complex
matlab_mat_c *matlab_ifft_c(void *A);
matlab_mat_c *matlab_fft2_c(void *A);
matlab_mat_c *matlab_ifft2_c(void *A);
```

Vector inputs (1×N or N×1) transform along their non-singleton
dim; matrices transform along columns (MATLAB's default). 2-D
variants apply the 1-D transform along rows then columns — order
doesn't matter for a separable transform.

No external library. No `-lfftw3`. Compiles with `cc -std=c99`.

## Compiler side

- `2i` / `3.5j` lex as `imag_literal`, parse as `ImagLiteral`, lower
  to `matlab.const_complex` with the source text as a string attribute.
  `LowerTensorOps.cpp:rewriteComplexLiterals` parses the imag
  magnitude and emits `llvm.call @matlab_complex_scalar(0.0, imag)`
  returning `!llvm.ptr`.

- `Dtype::Complex` maps to `!llvm.ptr` in `TypeMapper.cpp` (not
  `ComplexType<F64>`). Every complex-touching slot / load / store /
  call becomes ptr-typed at MLIR level, which keeps verification
  clean through the full conversion pipeline.

- The Sema resolver registers `fft` / `ifft` / `fft2` / `ifft2` /
  `conj` / `real` / `imag` / `angle` as builtins so bare references
  from user code resolve instead of erroring with "undefined name".
  `abs` routes to the polymorphic `matlab_abs_c` (works on either
  real or complex input).

- The generic builtin-return-type override in `Lowering.cpp` adds
  these names to the `PtrRet` set so implicit display in the REPL
  (`fft(x)` at the prompt) sees a non-None result and emits `disp`.

## What's deferred

- **Complex linalg tail**: `inv(A)`, `det(A)`, `svd(A)`, `eig(A)`,
  `chol(A)`, `qr(A)`, `lu(A)` — all still take a real matrix only.
  Each needs a complex-aware kernel in the runtime; the Lowerer
  dispatch layer is ready.
- **`complex(re, im)` builder**: MATLAB's explicit constructor.
  Workaround: `re + im*1i` works end-to-end.
- **`.^` (element-wise power) on complex** — see above.
- **Format `%gi` variants** for `disp` on real matrices with
  nontrivial imaginary content — today `disp` of a scalar 1×1
  complex prints `a + bi`, but reductions over complex matrices
  (`sum`, `mean`, …) still flow through the real path and will
  drop the imag component.

## See also

- [`docs/feature_status.md`](feature_status.md) — full feature
  matrix, including the complex and FFT rows this page anchors.
- [`docs/repl.md`](repl.md) — the JIT REPL picks up complex ops
  automatically (same lowering pipeline as AOT).
