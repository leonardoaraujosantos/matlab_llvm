# LAPACK Acceleration Benchmark

Tracks the wall-clock impact of the LAPACK acceleration epic
([#45](../../docs/lapack_roadmap.md)) on dense linear algebra. Three
implementations of each kernel — pure Python (triple-loop), NumPy
(BLAS / LAPACK), and matlab_llvm — at three sizes (small / medium /
large). The driver writes timings to `results/<tag>.json`; the
report script renders them as a Markdown table.

Phases captured:

| Tag | When | matlab_llvm config |
|---|---|---|
| `baseline_pre_lapack` | before Phase 1 | naive O(N³), no BLAS / LAPACK |
| `phase1` | after Phase 1 | BLAS gemm only |
| `phase2` | after Phase 2 | + LAPACK solve / LU / QR / chol |
| `phase3` | after Phase 3 | + LAPACK eig / SVD / Schur |

## Running

```bash
# From repo root, after `cmake --build build --target matlabc`:
bash bench/lapack/driver.sh <tag>

# Generate the comparison report once two tags exist:
python3 bench/lapack/report.py baseline_pre_lapack phase3
```

## Kernels covered

| Kernel | Phase | Sizes | Story |
|---|---|---|---|
| `matmul` (`A * B`) | 1 | 100, 300, 1000 | dense linalg — LAPACK closes the gap to NumPy |
| `solve` (`A \ b`) | 2 | 100, 300, 1000 | dense linalg |
| `lu` (`[L,U] = lu(A)`) | 2 | 100, 300, 1000 | dense linalg |
| `qr` (`[Q,R] = qr(A)`) | 2 | 100, 300, 1000 | dense linalg |
| `chol` (`R = chol(A'*A + I)`) | 2 | 100, 300, 1000 | dense linalg |
| `inv` (`inv(A)`) | 2 | 100, 300, 1000 | dense linalg |
| `eig` (`eig(A+A')` symmetric) | 3 | 100, 300, 1000 | dense linalg |
| `svd` (`svd(A)`) | 3 | 100, 300, 1000 | dense linalg |
| `mandelbrot` (escape-time, max_iter=100) | — | 100, 300, 1000 | scalar inner loop — matlab_llvm beats NumPy 4-6× and pure Python 11× |

Pure-Python is intentionally limited to the kernels where a scalar
algorithm is the natural comparison: `matmul` (triple loop) and
`mandelbrot`. Pure-Python at N=1000 is skipped for both — the runs
would take minutes and add nothing to the story. The dense-linalg
kernels (`lu` / `qr` / `svd` / etc.) skip pure-Python entirely; NumPy
already dispatches to LAPACK there, so a pure-Python version would
just measure CPython interpreter overhead.

## Reproducibility

- Input matrices are fixed via a deterministic seed (no `rand` time-of-day variance).
- 3 trials per (kernel, size, impl); we report the minimum (most stable measure of inherent cost, less skewed by interrupt jitter).
- Apple M-series users: avoid running the bench under Rosetta. Driver checks `arch` and prints a warning if not native.
- BLAS-pool / parfor-pool contention: driver pins single-threaded BLAS via `OPENBLAS_NUM_THREADS=1` / `VECLIB_MAXIMUM_THREADS=1` so the comparison is fair across implementations.
