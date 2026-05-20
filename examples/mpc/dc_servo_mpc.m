% examples/mpc/dc_servo_mpc.m — Tier-1 headline.
%
% DC-servomechanism position tracking with a Model Predictive
% Controller.  Adapted from MPC Toolbox User's Guide §2 "Design MPC
% Controller for Position Servomechanism".
%
% Plant: critically-damped 2-state servo (motor angle + velocity),
%   continuous-time state-space:
%       θ̇ = ω
%       ω̇ = (V - 2ω) / 0.5
%   so A_c = [0 1; 0 -4], B_c = [0; 2], C = [1 0], D = [0].
%
% Controller: linear MPC, p = 10 prediction steps, m = 2 control
% moves, ±220 V actuator bound.  The MPC discretises the plant
% internally via the stamped Ts = 0.1 on the c2d return, builds the
% prediction matrices Sx/Su/Su1/Hessian once at construction, and
% solves a hard-bounded QP via the KWIK active-set solver at each
% tick.
%
% The closed-loop step response should track a unit step in θ
% within ~1.5 s with a small overshoot.
%
% Run via the regular test harness:
%   test/Run/mpc_t1_dc_servo.m is the gating copy of this script.

A_c = [0, 1; 0, 0-4];
B_c = [0; 2];
C_c = [1, 0];
D_c = [0];

sys_c = ss(A_c, B_c, C_c, D_c);
sys_d = c2d(sys_c, 0.1);

obj = mpc(sys_d, 10, 2);
obj.umax = [220];
obj.umin = [-220];

T = 30;
r = [1];
y = sim(obj, T, r);

fprintf('DC servo closed-loop step response:\n');
fprintf('  t = 0.1s: theta = %.4f\n', y(1, 1));
fprintf('  t = 0.5s: theta = %.4f\n', y(5, 1));
fprintf('  t = 1.0s: theta = %.4f\n', y(10, 1));
fprintf('  t = 3.0s: theta = %.4f\n', y(30, 1));
