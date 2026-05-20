% MPC Tier-6 §7.7 — reference previewing.
% Passing r as a (p × ny) matrix makes mpcmove see a per-step
% setpoint trajectory.  With a setpoint ramp 0.2 → 0.4 → ... → 1.0,
% the first move should be smaller than the broadcast-1 baseline.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

% Broadcast reference (single ny×1 vector): same r=1 across horizon.
obj1 = mpc(sys_d, 5, 2);
obj1.umax = [10]; obj1.umin = [-10];
st1 = mpcstate(2, 1, 1);
u1 = mpcmove(obj1, st1, [0], [1]);
fprintf('u broadcast r=1: %.4f\n', u1(1, 1));

% Preview reference (p×ny matrix): r ramps 0.2 → 1.0 over the horizon.
obj2 = mpc(sys_d, 5, 2);
obj2.umax = [10]; obj2.umin = [-10];
st2 = mpcstate(2, 1, 1);
r_prev = [0.2; 0.4; 0.6; 0.8; 1.0];   % (p=5) × (ny=1)
u2 = mpcmove(obj2, st2, [0], r_prev);
fprintf('u preview ramp: %.4f\n', u2(1, 1));
% The previewed first move should be lower than the broadcast u1.
