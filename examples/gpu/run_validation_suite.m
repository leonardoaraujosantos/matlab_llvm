% run_validation_suite.m — orchestrate the 7 GPU-lane validation tests
% in the canonical order from the project's GPU Coder validation rubric:
%
%   1. AXPY (element-wise + gather)
%   2. arrayfun (scalar kernel from anon)
%   3. GEMM (BLAS-like backend dispatch)
%   4. Stencil 2-D (indexing + neighbourhood)
%   5. Benchmark (CPU vs GPU sweep)
%   6. parfor batches (task parallelism)
%
% Each test reports its own pass/fail line; this driver runs them in
% sequence and is the canonical entry point for the GPU validation
% suite (analogous to `runtests('gpu_validation')` in MathWorks).

fprintf('=== 1. AXPY ===\n');
test_gpuarray_axpy(1024);

fprintf('\n=== 2. arrayfun ===\n');
test_gpuarray_arrayfun(1024);

fprintf('\n=== 3. GEMM ===\n');
C = test_gpuarray_gemm(64);
fprintf('GEMM result (1,1) = %g\n', C(1, 1));

fprintf('\n=== 4. Stencil 2-D ===\n');
B = test_gpuarray_stencil2d(32, 3);
fprintf('stencil result (16,16) = %g\n', B(16, 16));

fprintf('\n=== 5. Benchmark sweep ===\n');
benchmark_gpu_backend();

fprintf('\n=== 6. parfor batches ===\n');
r = test_parfor_gpu_batches(4, 32);
fprintf('parfor batch sum-norms = '); disp(r');
