% Tier-4 follow-ons — model-reduction + time-delay surface.
%
% Backs matrix-arg runtime entries (matlab_pade_num / _den for
% Padé approximation of e^{-τs}, matlab_minreal_tf_num / _den for
% transfer-function pole-zero cancellation) and the model-object
% short forms hsvd(sys) / balreal_T(sys).
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as the existing
% model-object tests.

% --- §5.3 [num, den] = pade(τ, n) — [n/n] Padé of e^{-τs}.
%   [1/1]:  e^{-τs} ≈ (1 − τs/2) / (1 + τs/2)
%   [2/2]:  e^{-τs} ≈ (τ²s²/12 − τs/2 + 1) / (τ²s²/12 + τs/2 + 1)
[num1, den1] = pade(1, 1);
disp(num1);
disp(den1);

[num2, den2] = pade(1, 2);
disp(num2);
disp(den2);

% --- §5.1 [num_r, den_r] = minreal(num, den, tol) — tf-form
% pole-zero cancellation. Common factor (s + 1) drops both sides
% so (s + 1) / ((s + 1)(s + 2)) reduces to 1 / (s + 2).
n = [1 1];
d = [1 3 2];
[nr, dr] = minreal(n, d, 1e-6);
disp(nr);
disp(dr);

% --- §5.1 hsvd(sys) / balreal_T(sys) on a balanced-style 2-state
% plant. Numbers reproduce the existing ctrl_balreal regression.
A = [-1 0; 0 -2];
B = [1; 1];
C = [1 1];
D = [0];
sys = ss(A, B, C, D);

disp(hsvd(sys));
disp(balreal_T(sys));
