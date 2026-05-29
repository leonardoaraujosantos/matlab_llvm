% tierc_unrealized_cast.m — diagnostic-pinning fixture for the emit-c
% lane's gap on Tier-C 4-D tensor ops.
%
% conv2d_batch's runtime ABI (matlab_conv2d_batch_full) takes a void*
% argument that LowerTensorOps bridges from a tensor result via
% `builtin.unrealized_conversion_cast`.  The C emitter
% (lib/MLIR/Passes/EmitC.cpp) has no case for that op, so it bails out
% with a clear diagnostic instead of silently emitting broken C.
%
% This test asserts the diagnostic surface stays exactly:
%   "unsupported op in emitter: builtin.unrealized_conversion_cast"
%
% Pairs with the ~27 .skip-emit-c markers in test/Run/ that document the
% same capability gap.  When emit-c gains a tensor->ptr coercion lowering
% (proper handling of unrealized_conversion_cast), this test will start
% failing and we'll remove it alongside the skip markers.
X = zeros(2, 2, 1, 1);
W = zeros(2, 2, 1, 1);
Y = conv2d_batch(X, W);
fprintf('done\n');
