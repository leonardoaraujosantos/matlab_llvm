% MPC Tier-2 §3.4 — output-bound saturation.
% Use a 2-state plant with an aggressive setpoint that would
% normally drive y past the upper bound; verify the controller
% holds y at the limit.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
obj.umax = [10];
obj.umin = [-10];
obj.ymax = [0.5];                 % cap the output at 0.5
obj.ymin = [-10];

T = 20;
r = [1];                          % ask for 1 but the limit clips
y = sim(obj, T, r);

fprintf('y(5)  = %.4f\n', y(5, 1));
fprintf('y(10) = %.4f\n', y(10, 1));
fprintf('y(20) = %.4f\n', y(20, 1));
