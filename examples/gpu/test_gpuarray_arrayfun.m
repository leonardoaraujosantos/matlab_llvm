% test_gpuarray_arrayfun.m — scalar kernel applied element-wise on the GPU lane.
%
% Validates: arrayfun-driven kernel generation from a scalar function
% applied to a gpuArray.  Per the GPU Coder pattern: the @kernel body
% gets lowered to MSL / CUDA-C / OpenCL-C at codegen time and dispatched
% as a per-element grid launch.  Today (CPU-debug lane): the function-
% handle ABI calls the anon body sequentially on the host.
function y = test_gpuarray_arrayfun(n)
    x = gpuArray.linspace(single(-10), single(10), n);
    y_gpu = arrayfun(@activation_kernel, x);
    y = gather(y_gpu);
end

function y = activation_kernel(x)
    % Sigmoid — a standard activation choice that exercises exp() in
    % the inner kernel.  Backends will need exp() support in their
    % per-target math library binding.
    y = 1 ./ (1 + exp(-x));
end
