% Tier 3 — `pole(A)` (alias for eig) and `getPeakGain_ss(A,B,C,D)`
% (rough H∞ approximation via log-spaced freq grid).

% --- 1. pole(A) — closed-loop pole list. Same shape as eig(A).
A = [0-1, 0.5; 0, 0-2];
disp('pole(A) — should be {-2, -1} (sorted ascending):');
disp(pole(A));

% --- 2. getPeakGain_ss on a 1st-order plant. H(s) = 1/(s+1) → peak = 1.
B = [1; 0];
C = [1, 0];
D = [0];
fprintf('1st-order peak (closed form 1.0): %.4f\n', getPeakGain_ss(A, B, C, D));

% --- 3. Resonant 2nd-order. wn=5, zeta=0.05. Closed form peak ≈ 10
% but the 200-point log grid misses the sharp peak; expect ~8.95.
wn = 5; zeta = 0.05;
A2 = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B2 = [0; wn*wn];
C2 = [1, 0];
D2 = [0];
fprintf('Resonant peak grid-approx (closed form ~10): %.4f\n', ...
        getPeakGain_ss(A2, B2, C2, D2));

% --- 4. Higher-damping: peak == DC gain (no resonance).
zeta3 = 0.7;
A3 = [0, 1; 0-wn*wn, 0-2*zeta3*wn];
fprintf('Damped peak (closed form 1.0 = DC gain): %.4f\n', ...
        getPeakGain_ss(A3, B2, C2, D2));
