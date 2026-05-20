% MPC Tier-3 §4.2 — time-varying MPC with stacked per-step plants.
% Two-plant horizon: the first half uses plant A_lo, the second half
% A_hi.  Verifies the TV matrix builders produce a different MV than
% steady plant A_lo would yield.

% Plant A_lo (slow).
A_lo = [0.9, 0.0; 0.0, 0.9];
B_lo = [1; 0.5];
C_lo = [1, 0];
D    = [0];

% Plant A_hi (fast).
A_hi = [0.5, 0.0; 0.0, 0.5];
B_hi = [1; 0.5];
C_hi = [1, 0];

% Build a baseline mpc with A_lo, then drive one tick with TV stack.
sys_lo = ss(A_lo, B_lo, C_lo, D, 0.1);
obj = mpc(sys_lo, 4, 2);
obj.umax = [10];
obj.umin = [-10];

% Stack: p = 4 steps.  Use A_lo for steps 0/1, A_hi for steps 2/3.
% A_stack is (p*nx × nx) = (8 × 2); B_stack is (8 × 1); C_stack
% is (p*ny × nx) = (4 × 2).
A_top = vertcat(A_lo, A_lo);
A_bot = vertcat(A_hi, A_hi);
A_stack = vertcat(A_top, A_bot);             % 8 × 2
B_top = vertcat(B_lo, B_lo);
B_bot = vertcat(B_hi, B_hi);
B_stack = vertcat(B_top, B_bot);             % 8 × 1
C_top = vertcat(C_lo, C_lo);
C_bot = vertcat(C_hi, C_hi);
C_stack = vertcat(C_top, C_bot);             % 4 × 2

st = mpcstate(2, 1, 1);
ym = [0];
r  = [1];
u_tv = mpcmoveTV(obj, st, A_stack, B_stack, C_stack, ym, r);
fprintf('u (TV with mixed lo/hi plants): %.4f\n', u_tv(1, 1));

% Compare to all-A_lo TV (should match the standard mpcmove on A_lo).
A_top2 = vertcat(A_lo, A_lo);
A_stack2 = vertcat(A_top2, A_top2);
B_top2 = vertcat(B_lo, B_lo);
B_stack2 = vertcat(B_top2, B_top2);
C_top2 = vertcat(C_lo, C_lo);
C_stack2 = vertcat(C_top2, C_top2);
st2 = mpcstate(2, 1, 1);
u_tv_lo = mpcmoveTV(obj, st2, A_stack2, B_stack2, C_stack2, ym, r);
fprintf('u (TV all-lo, must match steady): %.4f\n', u_tv_lo(1, 1));

% Verify the steady-state mpcmove on A_lo.
obj2 = mpc(sys_lo, 4, 2);
obj2.umax = [10]; obj2.umin = [-10];
st3 = mpcstate(2, 1, 1);
u_steady = mpcmove(obj2, st3, ym, r);
fprintf('u (steady on A_lo):              %.4f\n', u_steady(1, 1));
