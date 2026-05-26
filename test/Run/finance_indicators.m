% Financial Toolbox Tier-1 §5 — function-form indicators on matlab_mat.

p = [100; 102; 101; 103; 105; 104; 106];

% tick2ret: per-period simple returns.
r = tick2ret(p);
fprintf('tick2ret length = %.0f\n', length(r));
fprintf('  r(1) = %.4f  (102/100 - 1)\n', r(1));
fprintf('  r(6) = %.4f  (106/104 - 1)\n', r(6));

% ret2tick: round-trip from returns -> normalized prices.
t = ret2tick(r);
fprintf('ret2tick length = %.0f\n', length(t));
fprintf('  t(1) = %.4f (start)\n', t(1));
fprintf('  t(7) = %.4f (==1.06, since p(7)/p(1)=1.06)\n', t(7));

% sma(p, 3): rolling 3-period mean.
s = sma(p, 3);
% s(3) = mean(100,102,101) = 101; s(7) = mean(105,104,106) = 105
fprintf('sma(p,3): s(3)=%.4f s(7)=%.4f\n', s(3), s(7));

% Bollinger bands on a 5-period window with K=2.
b = bolling(p, 5, 2);
% Last row: window = [101,103,105,104,106], mean = 103.8
% var = mean of (d^2) = mean((101-103.8)^2, (103-103.8)^2, ...) ≈ 3.36
% sd ≈ 1.833. Upper = 103.8 + 2*1.833 = 107.47
fprintf('bolling last: mid=%.2f upper=%.2f lower=%.2f\n', ...
        b(7,1), b(7,2), b(7,3));

% RSI(p, 3): based on 3 most recent diffs.
% Gains: 2, 2, 2; Losses: 1, 0, 0 across windows...
ri = rsindex(p, 3);
fprintf('rsindex(p,3) last = %.2f\n', ri(7));
