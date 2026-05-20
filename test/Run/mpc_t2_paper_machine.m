% MPC Tier-2 headline — Paper Machine Process (User's Guide §2.116).
% 2-input 2-output multivariable plant with mass-flow + dilution
% disturbances.  We simplify the User's Guide model to a 2×2 stable
% discrete plant with cross-coupling, set per-output references and
% per-MV bounds, and verify the controller reaches both setpoints.
%
% Plant: 2 states (slow + fast), 2 inputs (stock flow, white water
% flow), 2 outputs (mass concentration, level).  Cross-coupling
% between channels means the QP has to coordinate moves to track
% both references simultaneously.

% Discrete 2×2 plant at Ts = 0.5.
A = [0.7, 0.0; 0.0, 0.5];
B = [0.5, 0.1; 0.1, 0.4];
C = [1.0, 0.0; 0.0, 1.0];
D = [0, 0; 0, 0];
sys_d = ss(A, B, C, D, 0.5);

% MPC with p = 8, m = 3 (control horizon < prediction horizon to
% exercise the blocking / J_M selector).
obj = mpc(sys_d, 8, 3);

% Per-MV bounds (asymmetric — stock pump can go full forward / back,
% white water pump is one-sided).
obj.umax = [5; 3];
obj.umin = [-5; 0];

% Output disturbance estimator on — typical paper-machine setup.
obj.outdist = 1;

% Step both outputs from 0 to a 2-vector setpoint.
T = 40;
r = [1.0; 0.5];
y = sim(obj, T, r);

fprintf('y1(5)   = %.4f\n', y(5, 1));
fprintf('y2(5)   = %.4f\n', y(5, 2));
fprintf('y1(20)  = %.4f\n', y(20, 1));
fprintf('y2(20)  = %.4f\n', y(20, 2));
fprintf('y1(40)  = %.4f\n', y(40, 1));
fprintf('y2(40)  = %.4f\n', y(40, 2));
