% MPC Tier-1 §2.7 — mpcmove with an active MV bound.
% The umax = 0.5 limit kicks in immediately; KWIK should saturate
% the first move at the bound.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
obj.umax = [0.5];
obj.umin = [-0.5];
st  = mpcstate(2, 1);

r  = [1];
ym = [0];
u  = mpcmove(obj, st, ym, r);

% u should saturate at 0.5 (the unconstrained move was ~1.97 from
% the previous test).
fprintf('u = %.6f\n', u(1));
