% MPC Tier-3 §4.3 — gain-scheduled MPC via user-level controller bank.
% Two pre-built mpc objects covering two operating regimes; a
% scheduling variable picks between them.  Each controller has its
% own mpcstate; the user-visible idiom is just `if sched > t; u =
% mpcmove(mpc_hi, st_hi, ...); else; u = mpcmove(mpc_lo, st_lo, ...);
% end`.  A dedicated `mpcmoveMultiple` wrapper is a follow-up.

% Low-regime plant (slow dynamics).
A_lo = [0.9, 0.0; 0.0, 0.9];
B_lo = [1; 0.5];
C    = [1, 0];
D    = [0];
sys_lo = ss(A_lo, B_lo, C, D, 0.1);
mpc_lo = mpc(sys_lo, 5, 2);
mpc_lo.umax = [10];
mpc_lo.umin = [-10];

% High-regime plant (fast dynamics).
A_hi = [0.5, 0.0; 0.0, 0.5];
B_hi = [1; 0.5];
sys_hi = ss(A_hi, B_hi, C, D, 0.1);
mpc_hi = mpc(sys_hi, 5, 2);
mpc_hi.umax = [10];
mpc_hi.umin = [-10];

st_lo = mpcstate(2, 1, 1);
st_hi = mpcstate(2, 1, 1);
ym = [0];
r  = [1];

% Schedule on "operating-point" variable.
sched = 0.3;
if sched < 0.5
    u = mpcmove(mpc_lo, st_lo, ym, r);
    fprintf('sched=%.2f → low regime, u = %.4f\n', sched, u(1, 1));
else
    u = mpcmove(mpc_hi, st_hi, ym, r);
    fprintf('sched=%.2f → high regime, u = %.4f\n', sched, u(1, 1));
end

% Switch schedule.
sched = 0.8;
if sched < 0.5
    u = mpcmove(mpc_lo, st_lo, ym, r);
    fprintf('sched=%.2f → low regime, u = %.4f\n', sched, u(1, 1));
else
    u = mpcmove(mpc_hi, st_hi, ym, r);
    fprintf('sched=%.2f → high regime, u = %.4f\n', sched, u(1, 1));
end
