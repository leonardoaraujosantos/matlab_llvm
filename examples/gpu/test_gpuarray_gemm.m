% test_gpuarray_gemm.m — matrix-matrix multiply on the GPU lane.
%
% Validates: BLAS-like backend dispatch.  On a real device:
%   - CUDA   → cuBLAS  Sgemm
%   - Metal  → MPSMatrixMultiplication
%   - OpenCL → clBlast Sgemm
% Today (CPU-debug lane): host matlab_matmul_mm.
function C = test_gpuarray_gemm(n)
    A = gpuArray.rand(n, n, 'single');
    B = gpuArray.rand(n, n, 'single');
    tic;
    Cgpu = A * B;
    gpuTime = toc;
    C = gather(Cgpu);
    fprintf('GPU matrix multiply time = %.4f s\n', gpuTime);
end
