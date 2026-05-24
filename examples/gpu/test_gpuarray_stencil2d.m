% test_gpuarray_stencil2d.m — 5-point 2-D stencil on the GPU lane.
%
% Validates: 2-D indexing on gpuArrays, slice assignment, neighbourhood
% access (the canonical Sobel/heat-diffusion stencil pattern).  Per the
% GPU Coder UG p. 2-91 "Stencil Processing" reference design.
function B = test_gpuarray_stencil2d(n, steps)
    A = gpuArray.rand(n, n, 'single');
    for k = 1:steps
        B = zeros(size(A), 'like', A);
        B(2:end-1, 2:end-1) = ...
            0.25 * (A(1:end-2, 2:end-1) + ...
                    A(3:end,   2:end-1) + ...
                    A(2:end-1, 1:end-2) + ...
                    A(2:end-1, 3:end));
        A = B;
    end
    B = gather(A);
end
