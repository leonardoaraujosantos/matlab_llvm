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

| Kernel | Phase | Sizes |
|---|---|---|
| `matmul` (`A * B`) | 1 | 100, 300, 1000 |
| `solve` (`A \ b`) | 2 | 100, 300, 1000 |
| `lu` (`[L,U] = lu(A)`) | 2 | 100, 300, 1000 |
| `qr` (`[Q,R] = qr(A)`) | 2 | 100, 300, 1000 |
| `chol` (`R = chol(A'*A + I)`) | 2 | 100, 300, 1000 |
| `inv` (`inv(A)`) | 2 | 100, 300, 1000 |
| `eig` (`eig(A+A')` symmetric) | 3 | 100, 300, 1000 |
| `svd` (`svd(A)`) | 3 | 100, 300, 1000 |

Pure-Python at N=1000 is skipped — the triple loop would take minutes.

## Reproducibility

- Input matrices are fixed via a deterministic seed (no `rand` time-of-day variance).
- 3 trials per (kernel, size, impl); we report the minimum (most stable measure of inherent cost, less skewed by interrupt jitter).
- Apple M-series users: avoid running the bench under Rosetta. Driver checks `arch` and prints a warning if not native.
- BLAS-pool / parfor-pool contention: driver pins single-threaded BLAS via `OPENBLAS_NUM_THREADS=1` / `VECLIB_MAXIMUM_THREADS=1` so the comparison is fair across implementations.
