% Phase 5.6 Stage E SV — vector concat with static shapes.
%
% `[b, a(1:3)]` (concat of a literal-init array + slice) is the
% shift-register fragment in fir / sequential_processor (the full
% pattern wraps it in a persistent slot, which is Stage F). Stage E
% recognizes the concat operands as statically-shaped sources:
%
%   - `matlab_mat_i64_zeros(1, N)` (full literal-init array, after
%     Stage C lowering) contributes N elements.
%   - `matlab_mat_i64_slice1(src, range(start, 1, end))` with
%     constant range bounds contributes (end - start + 1) elements
%     read from `src` at indices [start..end].
%   - `matlab_mat_i64_from_scalar(val)` contributes 1 element.
%
% The concat is rewritten to a fresh `matlab_mat_i64_zeros(1, N) +
% N __subscript_store` chain that the existing zeros-folding path
% picks up. The final SV emits per-element store assigns
% (`v_concat[k] = v_src[k']`) with the concat result rendered as a
% static array port.
T = numerictype(1, 16, 0);
y = concat_static(fi(99, T));
disp(y);

function r = concat_static(c)
    %#codegen
    % Build two literal-init arrays. The first one's first element
    % is overwritten by `c` so the input port is observed without
    % participating in the wider sum (which would trip Verilator's
    % WIDTHEXPAND lint with mixed 16-bit + 32-bit operands).
    a = fi([1, 2, 3, 4], 1, 16, 0);
    a(1) = c;
    b = fi([10, 20], 1, 16, 0);
    % `[b, a(1:3)]`: 2 + 3 = 5 elements. Maps to:
    %   v_concat[0] = b[0],  v_concat[1] = b[1],
    %   v_concat[2] = a[0] (= c),
    %   v_concat[3] = a[1],
    %   v_concat[4] = a[2].
    z = [b, a(1:3)];
    r = z(1) + z(2) + z(3) + z(4) + z(5);
end
