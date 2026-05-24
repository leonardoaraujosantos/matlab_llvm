% Function-form AXPY — exercises:
%   - PromoteNoneParams pass (n: none → f64)
%   - PromoteBinopTypes pass (a .* x + b propagates ptr through binop chain)
%   - RefineSlotTypes propagating from binop results to local slots
% Without these passes the body's intermediate slots stay `none` and
% gather() fails to dispatch.
err = test_gpuarray_axpy(8);
fprintf('axpy err = %.0f\n', err);

function err = test_gpuarray_axpy(n)
    a = single(2.5);
    x = gpuArray.rand(n, 1, 'single');
    b = gpuArray.rand(n, 1, 'single');
    y_gpu = a .* x + b;
    y = gather(y_gpu);
    y_ref = a .* gather(x) + gather(b);
    err = max(abs(y - y_ref));
end
