% Tier 1.3 (Control System Toolbox roadmap §2.3) — expm matrix
% exponential. Scaling-and-squaring with [13/13] Pade approximant.
% Gates Tier 2.2 (c2d ZOH), Tier 2.3 (lsim continuous, initial), and
% the closed-form Lyapunov / Riccati transcriptions.

% expm(zeros) = I — defining identity. Print just the (1,1) entry to
% sidestep the matrix-element printing format (which differs across
% lanes for 1x1 matrices).
E0 = expm(zeros(2, 2));
disp(E0(1, 1));        % 1
disp(E0(2, 2));        % 1
disp(E0(1, 2));        % 0

% expm of a diagonal matrix: diag([exp(d_i)]).
%   A = diag([-1, -2]) -> expm(A) = diag([exp(-1), exp(-2)]).
% Use fprintf with bounded precision so all lanes print bit-stable
% ASCII (1e-12 absolute tolerance is well above float64 rounding).
A = [-1 0; 0 -2];
E = expm(A);
fprintf('%.6f\n', E(1, 1));    % 0.367879
fprintf('%.6f\n', E(2, 2));    % 0.135335
fprintf('%.6f\n', E(1, 2));    % 0.000000

% Rotation matrix.  expm([0 t; -t 0]) = [cos(t) sin(t); -sin(t) cos(t)].
% At t = pi/2: result is [0 1; -1 0] (the input itself, conveniently).
piover2 = pi / 2;
mpiover2 = 0 - piover2;
R = expm([0 piover2; mpiover2 0]);
fprintf('%.6f\n', R(1, 1));    % 0.000000  (cos pi/2)
fprintf('%.6f\n', R(1, 2));    % 1.000000  (sin pi/2)
fprintf('%.6f\n', R(2, 1));    % -1.000000
fprintf('%.6f\n', R(2, 2));    % 0.000000

% Group identity expm(A) * expm(-A) = I — exercises both branches and
% the LU back-substitution at the end of the Pade evaluation.
A2 = [1 -2 0; 3 0 1; 0 1 -1];
P = expm(A2);
N = expm(-A2);
M = P * N;
fprintf('%.6f\n', M(1, 1));    % 1.000000
fprintf('%.6f\n', M(2, 2));    % 1.000000
fprintf('%.6f\n', M(3, 3));    % 1.000000
fprintf('%.6f\n', abs(M(1, 2)));    % 0.000000  (abs() avoids -0 sign noise)
fprintf('%.6f\n', abs(M(3, 1)));    % 0.000000

% Large-norm path — anrm > theta13 forces scaling-and-squaring.
% A = diag([10, -8]) -> exp(10) ≈ 22026.4658, exp(-8) ≈ 0.00033546.
B = [10 0; 0 -8];
EB = expm(B);
fprintf('%.4f\n', EB(1, 1));   % 22026.4658
fprintf('%.6f\n', EB(2, 2));   % 0.000335
