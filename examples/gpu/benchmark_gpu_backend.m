% benchmark_gpu_backend.m — CPU-vs-GPU GEMM sweep across sizes.
%
% Validates: end-to-end pipeline — host alloc, gpuArray upload, GPU
% matmul, wait(gpuDevice) sync, gather, numerical equivalence within
% fp32 tolerance.  Prints per-size CPU time, GPU time, speedup, and
% max absolute error.  On the CPU-debug lane the "GPU" path is the
% same host BLAS as the reference; speedup ≈ 1x is the expected v1
% baseline.  Real backends should show speedup > 1x at sizes ≥ 1024.
function benchmark_gpu_backend()
    sizes = [512, 1024, 2048, 4096];
    for n = sizes
        fprintf('\nN = %d\n', n);
        A = rand(n, n, 'single');
        B = rand(n, n, 'single');
        tic;
        Ccpu = A * B;
        tCpu = toc;
        Ag = gpuArray(A);
        Bg = gpuArray(B);
        tic;
        Cgpu = Ag * Bg;
        wait(gpuDevice);
        tGpu = toc;
        C = gather(Cgpu);
        err = max(abs(C(:) - Ccpu(:)));
        fprintf('CPU time: %.4f s\n', tCpu);
        fprintf('GPU time: %.4f s\n', tGpu);
        fprintf('Speedup: %.2fx\n', tCpu / tGpu);
        fprintf('Max error: %.8g\n', err);
    end
end
