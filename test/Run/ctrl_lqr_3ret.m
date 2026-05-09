% Tier 2 (CST roadmap §4.1) — full 3-return [K, S, e] = lqr(A, B, Q, R).
% K is the gain (m × n). S is the Riccati solution X (n × n). e is the
% closed-loop spectrum eig(A − B·K) (n × 1, possibly complex).
%
% Routes the 2-/3-return shape through a dedicated splitter in
% LowerTensorOps.cpp:
%   K = matlab_lqr(A, B, Q, R)
%   S = matlab_care(A, B, Q, R)
%   e = matlab_lqr_e(A, B, Q, R)
% The 1-return form `K = lqr(A, B, Q, R)` continues to use the existing
% direct dispatch.

% --- 1. Double integrator. Closed form: K = [1, sqrt(3)],
% S = [sqrt(3), 1; 1, sqrt(3)], e = -sqrt(3)/2 ± j*0.5.
A = [0 1; 0 0];
B = [0; 1];
Q = [1 0; 0 1];
R = [1];

[K, S, e] = lqr(A, B, Q, R);
fprintf('K = [%.6f, %.6f]\n', K(1, 1), K(1, 2));
disp('Riccati S (closed form [sqrt(3), 1; 1, sqrt(3)]):');
disp(S);

% Closed-loop pole real and imag parts.
disp('eig real parts (-sqrt(3)/2):');
disp(real(e));
disp('eig imag parts (±0.5):');
disp(imag(e));

% --- 2. 2-return shape — drop e.
[K2, S2] = lqr(A, B, Q, R);
fprintf('K2 = [%.6f, %.6f]\n', K2(1, 1), K2(1, 2));
fprintf('S2(1,1) = %.6f\n', S2(1, 1));

% --- 3. Discrete dlqr — full 3-return shape.
Ad = [0.5 0; 0 0.6];
Bd = [1; 1];
Qd = [1 0; 0 1];
Rd = [1];
[Kd, Sd, ed] = dlqr(Ad, Bd, Qd, Rd);
disp('dlqr K:');
disp(Kd);
disp('dlqr eig magnitudes squared (must be < 1):');
mag2 = real(ed) .* real(ed) + imag(ed) .* imag(ed);
disp(mag2);
