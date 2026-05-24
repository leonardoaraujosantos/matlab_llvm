% Validates the Sema-level none→f64 promotion for top-level entry-point
% function parameters used numerically (lib/MLIR/Passes/PromoteNoneParams.cpp).
% Without the pass, `function y = produce(n)` would have `n` typed `none`
% and `gpuArray.rand(n, ...)` would fail at LowerTensorOps dispatch.
y = produce(8);
disp(y(1) >= 0);
fprintf('sz = %.0f %.0f\n', size(y, 1), size(y, 2));

function y = produce(n)
    g = gpuArray.rand(n, 1, 'single');
    y = gather(g);
end
