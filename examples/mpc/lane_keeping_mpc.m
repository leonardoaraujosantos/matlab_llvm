% examples/mpc/lane_keeping_mpc.m — Tier-3 headline (MATLAB form).
%
% Lane-keeping assist (MPC Toolbox User's Guide §12.10), simplified.
% 2-state lateral-dynamics plant tracking the lane-center reference
% under a tight ±2 m/s² lateral-acceleration bound, with an
% integrating output disturbance for offset-free steady-state under
% wind / bank disturbances.
%
% This is the MATLAB-side headline.  The roadmap's full
% lane_keeping_mpc_sil.mflow (Tier-3 §4.5–4.7) is a carve-down —
% emitting MPC as an mflow MpcMove block + cocotb SystemVerilog SIL
% needs matrix-state extensions to the signal-width inference pass
% and is deferred to a Tier-3b follow-up.

A_d = [1.00, 0.05;
       0.00, 0.90];
B_d = [0.00;
       0.05];
C_d = [1, 0];
D_d = [0];
sys_d = ss(A_d, B_d, C_d, D_d, 0.05);

obj = mpc(sys_d, 15, 3);
obj.umax = [2.0];
obj.umin = [-2.0];
obj.outdist = 1;
obj.Wy  = [5.0];
obj.Wdu = [0.2];

T = 60;
r = [1.0];
y = sim(obj, T, r);

fprintf('Lane-keeping closed-loop step response:\n');
fprintf('  t=0.25s  y_lat = %.4f m\n', y(5, 1));
fprintf('  t=0.50s  y_lat = %.4f m\n', y(10, 1));
fprintf('  t=1.00s  y_lat = %.4f m\n', y(20, 1));
fprintf('  t=3.00s  y_lat = %.4f m\n', y(60, 1));
