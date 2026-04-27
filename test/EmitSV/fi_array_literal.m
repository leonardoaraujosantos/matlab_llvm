% Phase 5.6 Stage C SV — static fi-array literal init.
%
% Mirrors `fi_array_sum4` (which uses `fi(zeros(1, 4), ...)`) but
% initializes the static array with a literal `[1, 2, 3, 4]` so the
% Stage C lowering shortcut fires. The args overwrite the literal
% constants and the final sum reads the array — same SV shape as
% sum4 (lint-clean), confirming that literal-init produces the same
% downstream IR as zeros-init.
%
% Lowering folds `fi([1, 2, 3, 4], 1, 16, 0)` at compile time into
% a `matlab_mat_i64_zeros(1, 4)` followed by four
% `__subscript_store` calls with the (already-integral) constants
% as the stored values. The existing `LowerStaticFiArrays` pass
% then collapses the chain into an `llvm.alloca [4 x i16]` with
% constant-init stores, just like the zeros-init case.
T = numerictype(1, 16, 0);
y = fi_array_literal(fi(10, T), fi(20, T), fi(30, T), fi(40, T));
disp(y);

function r = fi_array_literal(a, b, c, d)
    %#codegen
    v = fi([1, 2, 3, 4], 1, 16, 0);
    v(1) = a;
    v(2) = b;
    v(3) = c;
    v(4) = d;
    r = v(1) + v(2) + v(3) + v(4);
end
