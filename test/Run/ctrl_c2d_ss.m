% MPC Tier-0 — sys-form c2d that stamps Ts on the returned ss.
% Goal: sys_d = c2d(sys_c, Ts) must give a fresh ss whose Ts is set
% to the requested sample period. The matrix-form ctrl_c2d test
% covers the discretization math; this test covers the class wiring
% needed for MPC (`mpc(plant, Ts, …)` reads sys.Ts to decide whether
% to discretize the plant internally).

% --- 1. Continuous ss has Ts == 0 by default.
A = [0-1, 0; 0, 0-2];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_c = ss(A, B, C, D);
disp('continuous sys.Ts (default 0):');
disp(sys_c.Ts);

% --- 2. c2d(sys, Ts) returns an ss with sys_d.Ts == Ts.
Ts = 0.1;
sys_d = c2d(sys_c, Ts);
disp('discrete sys.Ts (must equal 0.1):');
disp(sys_d.Ts);

% --- 3. The A / B matrices match the matrix-form c2d.
[Ad, Bd] = c2d(A, B, Ts);
fprintf('%.6f\n', sys_d.A(1, 1));     % 0.904837
fprintf('%.6f\n', sys_d.A(2, 2));     % 0.818731
fprintf('%.6f\n', sys_d.B(1, 1));     % 0.095163
fprintf('%.6f\n', sys_d.B(2, 1));     % 0.045317

% --- 4. C and D pass through unchanged.
fprintf('%.6f\n', sys_d.C(1, 1));     % 1.000000
fprintf('%.6f\n', sys_d.C(1, 2));     % 0.000000
fprintf('%.6f\n', sys_d.D(1, 1));     % 0.000000

% --- 5. Explicit 5-arg ss constructor sets Ts directly.
sys_e = ss(A, B, C, D, 0.05);
disp('explicit-ctor sys.Ts (must equal 0.05):');
disp(sys_e.Ts);
