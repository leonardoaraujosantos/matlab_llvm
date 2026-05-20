% MPC Tier-1 §2.6 — sim closed-loop step response.
% Stable 2-state plant, ref tracking from y=0 to y=1.
% After ~50 ticks the output should be near 1.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
obj.umax = [10];
obj.umin = [-10];

T = 50;
r = [1];
y = sim(obj, T, r);

% Final output should approach the unit setpoint.
fprintf('y(1)  = %.4f\n', y(1, 1));
fprintf('y(5)  = %.4f\n', y(5, 1));
fprintf('y(20) = %.4f\n', y(20, 1));
fprintf('y(50) = %.4f\n', y(50, 1));
