% test_gpuarray_axpy.m — AXPY (a*x + b) on the GPU lane.
%
% Validates: element-wise op on gpuArrays, gpuArray.rand allocator,
% gather() round-trip, single-precision storage.  Per the GPU Coder
% validation rubric: numerical correctness within fp32 tolerance.
function err = test_gpuarray_axpy(n)
    a = single(2.5);
    x = gpuArray.rand(n, 1, 'single');
    b = gpuArray.rand(n, 1, 'single');
    y_gpu = a .* x + b;
    y = gather(y_gpu);
    y_ref = a .* gather(x) + gather(b);
    err = max(abs(y - y_ref));
    fprintf('AXPY error = %.8g\n', err);
end
