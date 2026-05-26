% Issue #20 Phase 3 (matrix-read capture) — a matrix allocated outside
% the parfor body, read element-by-element via subscript inside.  Today
% the body's `matlab.load(slot_A) : tensor<10x10xf64>` is forwarded to
% an LLVM-level ptr load from the state[] array (PtrTy slot tail), and
% the cloned `matlab.subscript` against that ptr lowers normally via
% LowerTensorOps's PtrTy-keyed dispatch.
%
% Pre-fix, this errored with:
%   parfor: body captures value of unsupported defining op 'matlab.alloc'
% because OutlineParfor's capture analysis only accepted f64 scalars
% and (in the rejection arm) matlab.alloc itself wasn't on the
% cloneable-external allowlist.

A = magic(10);
total = 0;
parfor j = 1:10
    total = total + A(j, j);
end
% Diagonal of magic(10): 92 + 99 + 1 + ... = 505.
fprintf('total = %d\n', total);

% Two captured matrices — one read symmetrically, one read by row.
B = magic(5);
C = magic(5);
combined = 0;
parfor i = 1:5
    combined = combined + B(i, i) * C(i, i);
end
% diag(magic(5)) = [17 5 13 21 9].  diag^2 sum = 1005.
fprintf('combined = %d\n', combined);
