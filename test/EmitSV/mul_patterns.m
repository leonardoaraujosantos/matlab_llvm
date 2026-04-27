% Phase 5.4 — constant-coefficient multiplier rewrite. Each
% `x * <const>` matches one of the simple-CSD patterns:
%
%   ×0    → 0           (folded)
%   ×1    → x           (passthrough)
%   ×2/×8 → x << k       (power of 2 shift)
%   ×7/×15 → (x << k) - x  (2^k - 1)
%   ×9/×17 → (x << k) + x  (2^k + 1)
%   ×11    → no rewrite (kept as `*`); v1 leaves arbitrary
%            constants alone, full Booth/CSD recoding is a v2
%            follow-up
%
% The synth tool's downstream view is far smaller and avoids the
% `*` operator on most patterns — useful both for ASIC flows that
% prefer explicit shift/add chains and for visibility (a code
% reviewer can read the resource shape directly from the SV).
T = numerictype(1, 16, 0);
y = mul_patterns(fi(5, T));
disp(y);

function y = mul_patterns(x)
    a = x * fi(0, numerictype(1, 16, 0));
    b = x * fi(1, numerictype(1, 16, 0));
    c = x * fi(2, numerictype(1, 16, 0));
    d = x * fi(8, numerictype(1, 16, 0));
    e = x * fi(7, numerictype(1, 16, 0));
    f = x * fi(15, numerictype(1, 16, 0));
    g = x * fi(9, numerictype(1, 16, 0));
    h = x * fi(17, numerictype(1, 16, 0));
    i = x * fi(11, numerictype(1, 16, 0));
    y = a + b + c + d + e + f + g + h + i;
end
