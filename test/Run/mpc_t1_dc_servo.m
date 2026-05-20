% MPC Tier-1 headline — DC servomechanism (MPC User's Guide §2.93).
% Classic example: 4-state continuous DC servo plant, c2d at Ts=0.1,
% tracking shaft position (output 1) with limits on the actuator
% voltage.  Output 2 (motor torque) is also predicted but unused
% as a constraint here (Tier-2 will add output bounds).
%
% Plant (User's Guide values):
%   states: motor inertia, motor velocity, load inertia, load velocity
%   inputs: armature voltage V (MV, ±220 V)
%   outputs: load position θ (rad), motor torque T (Nm)
% Continuous-time matrices (scaled):
%   A_c = [0 1 0 0; -kt/Jm -bt/Jm kt/(Jm*N) bt/(Jm*N);
%          0 0 0 1;  kt/(Jl*N) bt/(Jl*N) -kt/(Jl*N^2) -bt/(Jl*N^2)]
% For a smoke test we simplify to a 2-state critically-damped servo:
%   plant: angle θ̇ = ω, ω̇ = (V - 2·ω)/0.5
%   continuous A = [0 1; 0 -4], B = [0; 2], C = [1, 0], D = [0].
A_c = [0, 1; 0, 0-4];
B_c = [0; 2];
C_c = [1, 0];
D_c = [0];

% c2d to Ts = 0.1.
sys_c = ss(A_c, B_c, C_c, D_c);
sys_d = c2d(sys_c, 0.1);

% MPC with PredictionHorizon p = 10, ControlHorizon m = 2.
obj = mpc(sys_d, 10, 2);
obj.umax = [220];
obj.umin = [-220];

% Step-response sim: 30 ticks (3 s sim time), reference θ = 1 rad.
T = 30;
r = [1];
y = sim(obj, T, r);

fprintf('y(1)  = %.4f\n', y(1, 1));
fprintf('y(5)  = %.4f\n', y(5, 1));
fprintf('y(10) = %.4f\n', y(10, 1));
fprintf('y(30) = %.4f\n', y(30, 1));
