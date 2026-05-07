% Tier-2 (Signal Processing Toolbox roadmap §3.2): linear prediction
% — Levinson-Durbin recursion + LPC + Yule-Walker AR + Burg AR.
% Single-output form (just the AR coefficient row).

% Levinson-Durbin from a known autocorrelation (geometric, r[k] = α^|k|
% with α = 0.5). Order-2 fit recovers the exact AR(1) shape.
r = [1 0.5 0.25];
a = levinson(r, 2);
disp(a);             % [1, -0.5, 0] (order-2 fit on AR(1) data)

% LPC of a stationary signal — should match aryule by definition for
% the biased-autocorrelation case.
x = [1 2 3 4 5 6 7 8 9 10 11 12];
a1 = lpc(x, 3);
a2 = aryule(x, 3);
disp(a1);
disp(max(abs(a1 - a2)));    % 0 — same algorithm

% Burg's method on the same x. Different recursion → slightly different
% coefficients, but a[0] = 1 always.
ab = arburg(x, 3);
disp(ab(1));         % 1
disp(size(ab, 2));   % p+1 = 4
