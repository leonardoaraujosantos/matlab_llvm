% examples/mpc/paper_machine.m — Tier-2 headline.
%
% Paper Machine Process MPC (MPC Toolbox User's Guide §2.116),
% simplified to a 2-input 2-output discrete-time plant with
% cross-coupling and asymmetric MV bounds:
%
%   - 2 manipulated variables: stock flow (bidirectional) and white
%     water flow (one-sided)
%   - 2 measured outputs: mass concentration and level
%   - Coupled plant: A diagonal, B has off-diagonal entries → both
%     MVs affect both outputs
%   - Output-disturbance estimator (obj.outdist = 1) gives zero
%     steady-state error under unmodelled disturbances
%
% Demonstrates Tier-2 functionality:
%   - Multivariable QP (4-element decision Δu vector when m=3)
%   - Asymmetric MV bounds (umax/umin differ per channel)
%   - MV blocking (m=3 < p=8)
%   - Output disturbance integrator
%
% Run via the regular test harness:
%   test/Run/mpc_t2_paper_machine.m is the gating copy.

A = [0.7, 0.0; 0.0, 0.5];
B = [0.5, 0.1; 0.1, 0.4];
C = [1.0, 0.0; 0.0, 1.0];
D = [0, 0; 0, 0];
sys_d = ss(A, B, C, D, 0.5);

obj = mpc(sys_d, 8, 3);
obj.umax = [5; 3];
obj.umin = [-5; 0];
obj.outdist = 1;

T = 40;
r = [1.0; 0.5];
y = sim(obj, T, r);

fprintf('Paper machine closed-loop tracking:\n');
fprintf('  y1 (mass conc, target 1.0): %.4f at t=20s\n', y(40, 1));
fprintf('  y2 (level,    target 0.5): %.4f at t=20s\n', y(40, 2));
