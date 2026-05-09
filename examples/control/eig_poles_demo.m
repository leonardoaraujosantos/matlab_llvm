% eig(A) for non-symmetric state matrices — Tier 1.1 (CST roadmap §2.1).
%
% Until model-object constructors land in Tier 2 we don't have
% pole(sys), but the underlying primitive is just `eig` of the state
% matrix A.  This example shows the closed-loop pole computation that
% pole(feedback(...)) will eventually wrap.

% --- 1. Mass-spring-damper plant.
%   xdot = [0 1; -k/m  -c/m] x + [0; 1/m] u
%   m = 1, k = 9 (wn = 3 rad/s), c = 0.6 (zeta = 0.1).
m = 1; k = 9; c = 0.6;
A = [0 1; 0-k/m, 0-c/m];
disp('plant poles (expect lightly-damped pair near +- 3j):');
disp(real(eig(A)));        % approx -0.3
disp(imag(eig(A)));        % approx -+ 2.985

% --- 2. Open-loop poles of an inverted pendulum.
%   linearised around the upright equilibrium.
%   xdot = [0 1; g/L 0] x  (no damping, no input).
%   g = 9.81, L = 1 m.  Eigenvalues = +- sqrt(g/L) ~ +- 3.132.
g = 9.81; L = 1.0;
P = [0 1; g/L 0];
disp('inverted-pendulum poles (expect +- 3.13, both real):');
disp(eig(P));

% --- 3. Discrete-time system stability check.
%   x_{k+1} = A_d x_k.  Stable iff |eig(A_d)| < 1 for all eigenvalues.
%   Build A_d = expm(A * Ts) for the mass-spring-damper at Ts = 0.05 s.
Ts = 0.05;
Ad = expm(A * Ts);
disp('discrete-time poles (expect inside unit circle):');
e = eig(Ad);
disp(real(e));
disp(imag(e));
% Stability margin: max |e_k| should be < 1 for stable plant.
mag2 = real(e) .* real(e) + imag(e) .* imag(e);
disp('|e_k|^2 (must be <= 1):');
disp(mag2);
