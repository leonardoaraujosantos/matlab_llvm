% [X, K, L] = care(A, B, Q, R) — full continuous-time Riccati shape.
%   X: stabilising Riccati solution (n × n)
%   K: LQ gain  K = R⁻¹ B' X
%   L: closed-loop poles eig(A − B·K)  (matlab_mat_c when complex)
%
% Double-integrator with Q = I, R = 1: closed form
%   X = [√3 1; 1 √3]
%   K = [1, √3]

A = [0 1; 0 0];
B = [0; 1];
Q = eye(2);
R = 1;

[X, K, L] = care(A, B, Q, R);

% X ≈ [√3 1; 1 √3]
disp(X(1,1));
disp(X(1,2));
disp(X(2,1));
disp(X(2,2));

% K ≈ [1, √3]
disp(K(1,1));
disp(K(1,2));

% 2-return shape — same X/K as the 3-return form.
[X2, K2] = care(A, B, Q, R);
disp(X2(1,1) - X(1,1));
disp(K2(1,1) - K(1,1));

% Discrete: [X, K] = dare(Ad, Bd, Q, R) on a diagonal Schur-stable plant.
Ad = [0.5 0; 0 0.7];
Bd = eye(2);
Qd = eye(2);
Rd = eye(2);
[Xd, Kd] = dare(Ad, Bd, Qd, Rd);
disp(Xd(1,1));
disp(Xd(2,2));
disp(Kd(1,1));
disp(Kd(2,2));
