% test_parfor_gpu_batches.m — parfor batches each running a GPU GEMM.
%
% Validates: parfor + GPU dispatch composability.  Each iteration
% allocates two gpuArrays, multiplies, sums, and gathers the scalar.
% On a single-GPU box this serialises on the device (contention is
% expected; the test exercises the absence of races / deadlocks).
% On multi-GPU, see test_parfor_multigpu.m.
function results = test_parfor_gpu_batches(numBatches, n)
    results = zeros(numBatches, 1);
    parfor i = 1:numBatches
        A = gpuArray.rand(n, n, 'single');
        B = gpuArray.rand(n, n, 'single');
        C = A * B;
        results(i) = gather(sum(C, 'all'));
    end
end
