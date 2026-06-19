# Numeric Runtime Spec

## Purpose
Documents the observed behavior of the dense/sparse numeric core shared by every backend: matrix construction, indexing/slicing, broadcasting, reductions, linear-algebra decompositions, FFT, complex numbers, sparse matrices, and the size-threshold dispatch to LAPACK/BLAS. This is the foundation the language core and all emitters build on (src: runtime/matlab_runtime.cpp, runtime/runtime_internal.h).

## Requirements

### Requirement: Dense matrix construction and arithmetic
The system SHALL represent dense matrices as a row-major descriptor holding `double` data plus `rows`/`cols`, and SHALL provide constructors and element-wise/matrix arithmetic operators over them.

#### Scenario: Construct and add matrices
- **WHEN** a program calls `zeros`, `ones`, `eye`, `rand`, `magic`, `repmat`, or `matlab_mat_from_buf`/`matlab_mat_from_scalar` and then applies `+`, `-`, `.*`, `./`, `.^`, or `*` (matmul)
- **THEN** the system SHALL return a matrix descriptor with the correct shape and values (src: runtime/matlab_runtime.h matlab_zeros/matlab_ones/matlab_eye/matlab_repmat; runtime/matlab_runtime.cpp matlab_add_mm/matlab_emul_mm/matlab_matmul_mm)

#### Scenario: Element type is double
- **WHEN** any dense numeric operation runs
- **THEN** the system SHALL operate on IEEE-754 `double` storage (the dense numeric path does not carry single/int32/int64 element types) (src: runtime/runtime_internal.h matlab_mat)

### Requirement: Indexing and slicing
The system SHALL support 1-based linear indexing, 2-D subscripting, colon ranges, the `end` keyword, and logical/`find`-style indexing, walking elements in MATLAB column-major order.

#### Scenario: Linear and 2-D access
- **WHEN** a program reads `A(k)`, `A(i,j)`, a range `start:step:stop`, or assigns through `A(idx) = V`
- **THEN** the system SHALL resolve the index in column-major order and return/store the addressed elements (src: runtime/matlab_runtime.cpp matlab_subscript1_s/matlab_subscript2_s/matlab_slice1/matlab_slice2/matlab_slice_store1/matlab_range)

#### Scenario: Logical indexing and end
- **WHEN** a program uses `find(A)` or the `end` keyword in a subscript
- **THEN** the system SHALL return column-major 1-based indices of nonzero elements (`find`) or the size along the indexed dimension (`end`) (src: runtime/matlab_runtime.cpp matlab_find/matlab_end_of_dim)

### Requirement: Scalar broadcasting
The system SHALL broadcast a scalar operand against a matrix operand for element-wise binary operators via dedicated matrix-scalar and scalar-matrix entry points.

#### Scenario: Scalar plus matrix
- **WHEN** a program evaluates `s + A`, `A .* s`, or similar scalar/matrix combinations
- **THEN** the system SHALL apply the scalar to every element and return a matrix of the same shape as the matrix operand (src: runtime/matlab_runtime.h matlab_add_ms/matlab_add_sm/matlab_emul_ms/matlab_ediv_sm)

### Requirement: Reductions
The system SHALL provide reductions (sum, prod, mean, min, max) and cumulative reductions (cumsum, cumprod) that default to column-wise behavior and accept an explicit dimension argument.

#### Scenario: Column-wise and dimensioned reduction
- **WHEN** a program calls `sum(A)`, `mean(A)`, `max(A)`, etc., with or without a trailing dimension argument
- **THEN** the system SHALL reduce along columns by default (M×N → 1×N) and along the requested dimension when a `_dim` variant is invoked (src: runtime/matlab_runtime.cpp MULTIDIM_REDUCE sum/prod/mean/min/max, matlab_sum_dim, matlab_cumsum)

### Requirement: String-mode builtin call shapes
The system SHALL accept the standard string-flagged forms of construction/reduction builtins, lowering them to dedicated runtime entries where the semantics differ from the no-flag form: `sum(X, 'all')` (whole-array sum → scalar, distinct from the column-wise `sum(X)`), `norm(X, 'fro')` (Frobenius norm over all elements), and `zeros(sz, 'like', A)` / `ones`/`rand`/… (drop the `'like', A` prototype pair on the double-only CPU lane, keeping the numeric dims). These SHALL hold across the LLVM, C, C++, Python, and TypeScript lanes.

#### Scenario: Whole-array sum and Frobenius norm
- **WHEN** a program calls `sum(X, 'all')` or `norm(X, 'fro')`
- **THEN** the system SHALL return the scalar whole-array sum / Frobenius norm (src: runtime/matlab_runtime.cpp matlab_sum_all/matlab_norm_fro; lowering: lib/MLIR/Passes/LowerTensorOps.cpp string-mode intercept; test: test/Run/string_mode_builtins.m)

#### Scenario: Prototype-typed construction
- **WHEN** a program calls `zeros(sz, 'like', A)` (or `ones`/`rand`/`eye`/`randn` with `'like'`)
- **THEN** the system SHALL drop the `'like', A` pair and construct the array from the numeric dims (double-typed on the CPU lane)

### Requirement: Decompositions and FFT
The system SHALL provide eigen, singular-value, QR, LU, Cholesky, Schur, and Hessenberg decompositions plus FFT/IFFT (1-D and 2-D) with shift helpers.

#### Scenario: Compute a decomposition
- **WHEN** a program calls `eig`, `svd`, `qr`, `lu`, `chol`, `schur`, or `hess`
- **THEN** the system SHALL return the corresponding factor(s), using a hand-coded algorithm below the LAPACK threshold (src: runtime/matlab_runtime.cpp matlab_eig/matlab_svd/matlab_qr_Q/matlab_lu_L/matlab_chol/matlab_schur/matlab_hess)

#### Scenario: Transform with FFT
- **WHEN** a program calls `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, or `ifftshift`
- **THEN** the system SHALL compute the transform with a built-in Cooley-Tukey (radix-2) plus Bluestein (general N) implementation and return a complex matrix (src: runtime/matlab_runtime.cpp matlab_fft_c/matlab_ifft_c/matlab_fft2_c/matlab_fftshift_c; doc: docs/complex.md)

### Requirement: Complex number support
The system SHALL represent complex matrices with separate real/imaginary planes and SHALL provide complex arithmetic and component-extraction operators that polymorphically accept real or complex inputs.

#### Scenario: Complex arithmetic and extraction
- **WHEN** a program builds a complex value (e.g. `complex(re,im)`) and applies `+`, `-`, `.*`, `./`, or calls `conj`, `real`, `imag`, `abs`, `angle`
- **THEN** the system SHALL dispatch on the descriptor magic tag, promoting real operands to complex as needed, and return the correct complex or real result (src: runtime/runtime_complex.cpp matlab_complex_scalar/matlab_add_cc/matlab_conj_c/matlab_real_c; runtime/runtime_internal.h matlab_mat_c)

### Requirement: Sparse matrices
The system SHALL provide a CSR-format sparse matrix type built from triplets, with shape queries, sparse matrix-vector product, and a preconditioned conjugate-gradient solver.

#### Scenario: Build and solve sparse
- **WHEN** a program calls `sparse(I,J,V,m,n)` and then `nnz`, a sparse matrix-vector product, or `pcg`
- **THEN** the system SHALL store data in compressed-sparse-row layout (1-based MATLAB surface, 0-based internally), summing duplicate triplets, and SHALL route polymorphic ops via the sparse magic tag `0xC0FFEE05` (src: runtime/runtime_sparse.cpp matlab_sparse_from_triplets/matlab_sparse_nnz/matlab_sparse_matvec/matlab_sparse_pcg)

### Requirement: LAPACK/BLAS size-threshold dispatch
The system SHALL route dense matmul and decompositions to BLAS/LAPACK above a configurable size threshold when built with BLAS support, and SHALL fall back to built-in implementations otherwise.

#### Scenario: Threshold-based dispatch
- **WHEN** a dense decomposition runs with matrix dimension `n` >= the LAPACK threshold (default 64, override `MATLAB_LAPACK_MIN`), or a matmul exceeds the GEMM threshold (default 64³ ≈ 262144, override `MATLAB_BLAS_GEMM_MIN`), and the build defines `MATLAB_LLVM_WITH_BLAS`
- **THEN** the system SHALL call the corresponding `cblas_*`/LAPACK routine, otherwise it SHALL use the built-in path (src: runtime/matlab_runtime.cpp lapack_threshold line 110, blas_gemm_threshold line 612; doc: docs/lapack_roadmap.md)
