% Phase 4.5.5 (workaround) — scalar-args alternative to a literal
% vector function argument. The user's natural form
% `vector_processor(vec_a, vec_b)` with vec_a/vec_b as 3-element fi
% vectors does NOT work today: the user-call refinement collapses
% the !llvm.ptr call-site type to a scalar element type at the
% function signature, so `vec_a(1)` becomes a malformed
% `subscript(scalar, 1)`. Supporting vector args end-to-end
% requires changes throughout Sema / MIR-to-MLIR lowering / user-
% call refinement / LLVM tensor-op lowering — a multi-week effort.
%
% The recommended workaround is to pass the elements individually
% (the script-side caller "unrolls" the vector into N scalars).
% This emits clean SV with N input ports per vector.
T = numerictype(1, 16, 8);
[m, d] = vector_proc3(fi(1, T), fi(2, T), fi(3, T), fi(4, T), fi(5, T), fi(6, T));
disp(m);

function [mag_sq, dot_prod] = vector_proc3(a1, a2, a3, b1, b2, b3)
    %#codegen
    % Dot product
    p1 = a1 * b1;
    p2 = a2 * b2;
    p3 = a3 * b3;
    dot_prod = p1 + p2 + p3;
    % Magnitude squared
    sq1 = a1 * a1;
    sq2 = a2 * a2;
    sq3 = a3 * a3;
    mag_sq = sq1 + sq2 + sq3;
end
