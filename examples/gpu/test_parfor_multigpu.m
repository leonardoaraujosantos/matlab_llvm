% test_parfor_multigpu.m — round-robin GPU selection across parfor workers.
%
% Validates: multi-GPU scheduling.  Each iteration picks a device by
% modular hashing of the loop counter against gpuDeviceCount(), then
% runs a GEMM and gathers the Frobenius norm.  On a single-GPU box,
% gpuDeviceCount() = 1 and all workers fall back to the same device.
function results = test_parfor_multigpu(numBatches, n)
    gcount = gpuDeviceCount;
    results = zeros(numBatches, 1);
    parfor i = 1:numBatches
        gpuId = mod(i - 1, gcount) + 1;
        gpuDevice(gpuId);
        A = gpuArray.rand(n, n, 'single');
        B = gpuArray.rand(n, n, 'single');
        C = A * B;
        results(i) = gather(norm(C, 'fro'));
    end
end
