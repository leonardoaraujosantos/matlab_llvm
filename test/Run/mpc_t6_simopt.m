% MPC Tier-6 §7.6 — mpcsimopt PlantInitialState override.
% Default sim starts from x0 = 0 in the controller's internal model.
% mpcsimopt lets the caller hand in a different plant initial state.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);
obj = mpc(sys_d, 5, 2);
obj.umax = [10]; obj.umin = [-10];

% Default sim_opt (no override).
opt1 = mpcsimopt();
y1 = sim(obj, 3, [1], opt1);
fprintf('y default x0: %.4f %.4f %.4f\n', y1(1, 1), y1(2, 1), y1(3, 1));

% Override plant initial state to a non-zero x0 — the controller
% should still drive toward r=1.
opt2 = mpcsimopt();
opt2.PlantInitialState = [0.5; 0.5];
opt2.Use_PlantInitialState = 1;
y2 = sim(obj, 3, [1], opt2);
fprintf('y init x=[0.5;0.5]: %.4f %.4f %.4f\n', y2(1, 1), y2(2, 1), y2(3, 1));
