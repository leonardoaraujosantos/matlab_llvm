% H2 system norm via observability gramian — Tier 3 demo.
%
% For a stable LTI plant  G(s) = C (sI - A)^{-1} B + D  with D = 0,
% the H2 norm satisfies
%       ||G||_2^2 = trace(B' Wo B) = trace(C Wc C')
% where Wo = gram_o(A, C) and Wc = gram_c(A, B). The H2 norm measures
% the energy of the impulse response (or equivalently, the steady-state
% RMS output under unit-power white noise input). It's the cost
% functional that LQR minimises (with R^{-1/2} as the input weighting).
%
% Tier-3.4 of the CST roadmap. Sits cleanly on Tier-1.4 lyap.

% --- 1. Mass-spring-damper plant.
%   xdot = [0 1; -k/m -c/m] x + [0; 1/m] u,  y = [1 0] x.
m = 1.0;  k = 9.0;  c = 0.6;
A = [0 1;  0-k/m,  0-c/m];
B = [0;  1/m];
C = [1, 0];

% Compute both gramians.
Wc = gram_c(A, B);
Wo = gram_o(A, C);
disp('controllability gramian Wc:');
disp(Wc);
disp('observability gramian Wo:');
disp(Wo);

% H2 norm via Wo.  ||G||_2 = sqrt(B' Wo B).
H2sq = B' * Wo * B;
fprintf('H2 norm (via Wo) = %.6f\n', sqrt(H2sq(1, 1)));

% Cross-check via Wc.  ||G||_2 = sqrt(C Wc C').
H2sq2 = C * Wc * C';
fprintf('H2 norm (via Wc) = %.6f\n', sqrt(H2sq2(1, 1)));
% Both expressions must give the same value.

% --- 2. Step response — visualise via fprintf at a few checkpoints.
%   At sample N, the position should approach 1/k (DC gain).
D = [0];
y = step_ss(A, B, C, D, 0.05, 200);
disp('step response checkpoints (must converge to 1/k = 0.111):');
fprintf('y(t = 0)    = %.6f\n', y(1, 1));
fprintf('y(t = 1.0)  = %.6f\n', y(21, 1));
fprintf('y(t = 5.0)  = %.6f\n', y(101, 1));
fprintf('y(t = 10.0) = %.6f\n', y(200, 1));
% DC gain = -C A^{-1} B = 1/k:
DCgain = (0 - C) * inv(A) * B;
fprintf('DC gain = %.6f\n', DCgain(1, 1));
