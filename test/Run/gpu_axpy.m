% gpuArray PCT-surface AXPY validation.  Exercises:
%   - gpuArray.rand(n, m, 'single')  — static factory + dtype-tag drop
%   - element-wise `a .* x + b` on gpuArrays
%   - gather(g) round-trip
% CPU-debug lane v1 is numerical-equivalent to host doubles; backends
% (Metal/CUDA/OpenCL) will route through real device buffers.
%
% Validates the GPU-Tier-8 PCT-surface scaffolding wired in
% LowerTensorOps + runtime_gpu_helpers.cpp.
n = 8;
a = 2.5;
x = gpuArray.rand(n, 1, 'single');
b = gpuArray.rand(n, 1, 'single');
y_gpu = a .* x + b;
y = gather(y_gpu);
y_ref = a .* gather(x) + gather(b);
diff = y - y_ref;
fprintf('axpy err = %.0f\n', sum(sum(diff .* diff)));
