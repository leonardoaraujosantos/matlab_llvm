% MPC Tier-0 — sys-form kalman that picks the continuous- or
% discrete-time kernel based on sys.Ts.
%   kalman(sys, Qn, Rn) extracts A, B, C off the ss and dispatches:
%     sys.Ts == 0  → matlab_kalman_L (continuous)
%     sys.Ts >  0  → matlab_kalmd_L  (discrete)
% B serves as the noise input matrix G (the MPC §1.4 canonical
% input-channel-noise assumption).

% --- 1. Continuous SISO 1×1. a = -1, b = 1, c = 1, Qn = Rn = 1.
% Same closed form as the matrix-lane ctrl_kalman test:
%   dual ARE: -2P - P² + 1 = 0 → P = sqrt(2) - 1, L = P ≈ 0.4142.
A = [0-1];
B = [1];
C = [1];
D = [0];
sys_c = ss(A, B, C, D);
Qn = [1];
Rn = [1];
disp('continuous 1×1 Kalman gain L (sqrt(2) - 1):');
disp(kalman(sys_c, Qn, Rn));

% --- 2. Continuous 2-state — must agree with the matrix form.
A2 = [1, 1; 0, 0-2];
B2 = [1, 0; 0, 1];
C2 = [1, 0];
D2 = [0, 0];
sys_c2 = ss(A2, B2, C2, D2);
Q2 = [1, 0; 0, 1];
R2 = [1];
disp('continuous 2×1 Kalman gain L (sys-form):');
disp(kalman(sys_c2, Q2, R2));
disp('continuous 2×1 Kalman gain L (matrix-form, must match):');
disp(kalman_L(A2, B2, C2, Q2, R2));

% --- 3. Discrete dispatch — same A2 but tagged with Ts > 0.
% The dispatcher must pick matlab_kalmd_L (Schur-stable plant).
Ad = [0.7, 0.1; 0.0, 0.4];
Bd = [1, 0; 0, 1];
Cd = [1, 0];
Dd = [0, 0];
sys_d = ss(Ad, Bd, Cd, Dd, 0.1);
Qd = [1, 0; 0, 1];
Rd = [0.5];
disp('discrete sys.Ts (must be 0.1):');
disp(sys_d.Ts);
disp('discrete 2×1 Kalman gain L (sys-form, must match kalmd_L):');
disp(kalman(sys_d, Qd, Rd));
disp('discrete 2×1 Kalman gain L (matrix-form):');
disp(kalmd_L(Ad, Bd, Cd, Qd, Rd));
