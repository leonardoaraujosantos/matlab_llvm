% MPC Tier-6 §7.2 — rate bounds on Δu(j).
% Default unconstrained-on-rate behaviour vs. a tight dumax = 0.3
% rate limit that clips the move.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

% Reference move from the Tier-1 test: u ≈ 0.987 with default bounds.
obj1 = mpc(sys_d, 5, 2);
obj1.umax = [10]; obj1.umin = [-10];
st1 = mpcstate(2, 1, 1);
u_ref = mpcmove(obj1, st1, [0], [1]);
fprintf('u no rate bound: %.4f\n', u_ref(1, 1));

% With rate bound dumax = 0.3, the first move (which is Δu(0) since
% u_prev = 0) saturates at 0.3.
obj2 = mpc(sys_d, 5, 2);
obj2.umax = [10]; obj2.umin = [-10];
obj2.dumax = [0.3]; obj2.dumin = [-0.3];
st2 = mpcstate(2, 1, 1);
u_clip = mpcmove(obj2, st2, [0], [1]);
fprintf('u rate-bounded (dumax=0.3): %.4f\n', u_clip(1, 1));

% Continuous-plant auto-c2d (Tier-6 §7.1): build mpc on a continuous
% ss (no Ts) and verify it works.
A_c = [0, 1; -1, -1];
B_c = [0; 1];
C_c = [1, 0];
D_c = [0];
sys_c = ss(A_c, B_c, C_c, D_c);     % continuous (Ts = 0)
obj3 = mpc(sys_c, 5, 2);            % auto-c2d at default Ts = 0.1
fprintf('auto-c2d Ts: %.4f (must be 0.1)\n', obj3.Ts);
