% Phase 4.5.4 SV — static fi-array via `fi(zeros(1, N), ...)`.
% The user writes a 4-element local array, fills it from scalar
% inputs at constant indices, then sums the elements at constant
% indices. The lowering pipeline:
%
%   1. LowerStaticFiArrays recognizes `fi(zeros(1, 4), 1, 16, 0)`
%      and rewrites the runtime-call chain (matlab_mat_i64_zeros +
%      __subscript_store + matlab_mat_i64_subscript1_s) into
%      `llvm.alloca <[4 x i16]>` with GEP + load/store access.
%   2. The pass also DCEs surviving `matlab_mat_from_scalar`
%      wrappers, replaces `matlab_fi_sat_s64` with passthrough
%      (Wrap-mode arithmetic is the natural SV semantic), and
%      collapses `extsi/trunci` chains the saturate-replacement
%      step leaves behind.
%   3. HWLegalize accepts `llvm.alloca` of static integer-element
%      array type alongside scalar primitives.
%   4. The SV emitter renders the alloca as
%      `logic signed [15:0] arr [4];` with `arr[i] = v;` /
%      `v = arr[i];` access via a side-table that maps GEP results
%      to indexed-access expressions.
T = numerictype(1, 16, 0);
y = fi_array_sum4(fi(1, T), fi(2, T), fi(3, T), fi(4, T));
disp(y);

function r = fi_array_sum4(a, b, c, d)
    %#codegen
    v = fi(zeros(1, 4), 1, 16, 0);
    v(1) = a;
    v(2) = b;
    v(3) = c;
    v(4) = d;
    r = v(1) + v(2) + v(3) + v(4);
end
