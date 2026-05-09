% Tier 4.2 — `[L, P] = kalman(A, G, C, Qn, Rn)` 2-return shape.
% L is the gain (n × p ptr), P is the dual-care steady-state covariance
% (n × n SPD ptr). Routes through a 2-return splitter to
% matlab_kalman_L + matlab_kalman_P.
%
% Sanity: for a single-output plant (p = 1) with Rn = 1,
%   L = P · C' / Rn = P(:, k)
% where k is the 1-based column corresponding to the C-row's nonzero.

% --- 1. Open-loop unstable plant — same as ctrl_kalman.m's case 2.
A = [1, 1; 0, 0-2];
G = [1, 0; 0, 1];
C = [1, 0];
Qn = [1, 0; 0, 1];
Rn = [1];

[L, P] = kalman(A, G, C, Qn, Rn);
disp('L:');
disp(L);
disp('P:');
disp(P);
disp('P(:,1) (must equal L for this plant):');
disp(P(1, 1));
disp(P(2, 1));

% --- 2. 1-return form defaults to L (same as the existing kalman_L).
L2 = kalman(A, G, C, Qn, Rn);
disp('1-return L (must equal L from 2-return form):');
disp(L2);

% --- 3. Discrete kalmd 2-return shape on a Schur-stable plant.
Ad = [0.7, 0.1; 0.0, 0.4];
Gd = [1, 0; 0, 1];
Cd = [1, 0];
Qd = [1, 0; 0, 1];
Rd = [0.5];
[Ld, Pd] = kalmd(Ad, Gd, Cd, Qd, Rd);
disp('discrete L:');
disp(Ld);
disp('discrete P (n × n SPD):');
disp(Pd);
