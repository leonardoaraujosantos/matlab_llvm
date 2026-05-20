% MPC Tier-6 §7.3 — MV-tracking term Wu‖u - u_target‖².
% With a non-zero Wu and u_target = 0.5, the solver should bias the
% move toward 0.5 rather than the output-error-only optimum 0.987.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
obj.umax = [10]; obj.umin = [-10];
obj.Wu = [5.0];          % strong MV penalty
obj.u_target = [0.5];

st = mpcstate(2, 1, 1);
u = mpcmove(obj, st, [0], [1]);
fprintf('u MV-tracked (target=0.5): %.4f\n', u(1, 1));
% The drag toward 0.5 should pull the move below 0.987.
