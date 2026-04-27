% Phase 1 SV — pure dataflow chain (mul, mul, add).
% Exercises straight-line `always_comb` with no temps written twice.
T = numerictype(1, 32, 0);
y = sumsq(fi(3, T), fi(4, T));
disp(y);

function y = sumsq(a, b)
    y = a*a + b*b;
end
