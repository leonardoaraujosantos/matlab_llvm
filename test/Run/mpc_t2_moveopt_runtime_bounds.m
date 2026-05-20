% MPC Tier-2 §3.7 — mpcmoveopt for run-time MVMax override.
% Default umax = 10 (loose).  An mpcmoveopt with Use_MVMax=1 and
% MVMax=0.3 tightens the limit for one tick.  Compare the unbounded
% u vs. the overridden u.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
obj.umax = [10];
obj.umin = [-10];

% First tick, no override.
st1 = mpcstate(2, 1);
ym = [0];
r  = [1];
u_open = mpcmove(obj, st1, ym, r);
fprintf('u (no override): %.4f\n', u_open(1));

% First tick, with MVMax override = 0.3.
st2 = mpcstate(2, 1);
opt = mpcmoveopt();
opt.MVMax = [0.3];
opt.Use_MVMax = 1;
u_clip = mpcmove(obj, st2, ym, r, opt);
fprintf('u (MVMax=0.3 override): %.4f\n', u_clip(1));
