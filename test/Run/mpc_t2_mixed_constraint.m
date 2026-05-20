% MPC Tier-2 §3.1 — mixed input/output constraint.
% E·u + F·y ≤ G  evaluated at the first prediction step.  Here we
% impose `0.5·u + y ≤ 0.8` on a 2-state plant.  With y converging to
% 1 (and u driving it there), the constraint should clip the
% steady-state behavior.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
obj.umax = [10];
obj.umin = [-10];

% E (1 × 1 = nE × nu), F (1 × 1 = nE × ny), G (1 × 1).
obj.E = [0.5];
obj.F = [1.0];
obj.G = [0.8];

T = 20;
r = [1];
y = sim(obj, T, r);

fprintf('y(5)  = %.4f\n', y(5, 1));
fprintf('y(10) = %.4f\n', y(10, 1));
fprintf('y(20) = %.4f\n', y(20, 1));
