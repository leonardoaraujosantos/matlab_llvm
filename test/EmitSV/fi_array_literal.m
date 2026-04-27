% Phase 5.6 Stage C SV — static fi-array literal init.
%
% `h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15)` is the FIR / IIR / CIC
% coefficient-table shape. Lowering folds it at compile time into a
% `matlab_mat_i64_zeros(1, 4)` followed by four `__subscript_store`
% calls with the round-to-nearest-quantized stored integers. The
% existing `LowerStaticFiArrays` pass then collapses both runtime
% calls into an `llvm.alloca [4 x i16]` with constant-init stores,
% so the emitted SV contains a static array driven by integer
% literals — synthesis-friendly and zero runtime cost.
%
% Coefficients quantized at WL=16 / FL=15 (Q1.15):
%   0.1 → round(0.1 * 32768) = 3276
%   0.2 → round(0.2 * 32768) = 6554 (banker / round-to-nearest
%                              floor-half = 6553)
%   0.3 → round(0.3 * 32768) = 9830
%   0.4 → round(0.4 * 32768) = 13107
T = numerictype(1, 16, 15);
y = fi_array_literal(fi(0.5, T));
disp(y);

function y = fi_array_literal(x)
    %#codegen
    h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15);
    y = h(1) + h(2) + h(3) + h(4) + x;
end
