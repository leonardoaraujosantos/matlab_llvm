% Tier 2 (Control System Toolbox roadmap §4.1) — first user-facing
% wrapper. K = lqr(A, B, Q, R) is a 1-line wrapper over Tier-1.5 care
% that returns the optimal LQ feedback gain for
%      xdot = A x + B u,  cost J = integral (x' Q x + u' R u) dt.
%
% Closed-form for the canonical double integrator (A = [0 1; 0 0],
% B = [0; 1], Q = I, R = 1):  K = [1, sqrt(3)].

% --- 1. Double integrator.
A = [0 1; 0 0];
B = [0; 1];
Q = [1 0; 0 1];
R = [1];
K = lqr(A, B, Q, R);
fprintf('%.6f\n', K(1, 1));     % 1.000000
fprintf('%.6f\n', K(1, 2));     % 1.732051

% --- 2. The closed-loop poles eig(A - B K) — symmetric root locus
% places them at -sqrt(3)/2 +- j 0.5.
Acl = A - B * K;
disp(real(eig(Acl)));      % both -0.866025
disp(imag(eig(Acl)));      % imaginary conjugate pair (sorted ascending)

% --- 3. Marginally unstable plant: A = [1 1; 0 -2].  LQR stabilises.
A2 = [1 1; 0 0-2];
B2 = [1; 0];
Q2 = [1 0; 0 1];
R2 = [1];
K2 = lqr(A2, B2, Q2, R2);
% Closed-loop must be Hurwitz.
Acl2 = A2 - B2 * K2;
e2 = eig(Acl2);
% Both eigenvalues should be real and negative for this plant.
disp(e2);

% --- 4. SISO LQR cost from initial condition x0 = [1; 0]:
%       J* = x0' X x0  where  X = care(A, B, Q, R).
% For double integrator this is sqrt(3).
X = care(A, B, Q, R);
x0 = [1; 0];
J = x0' * X * x0;
fprintf('%.6f\n', J(1, 1));    % 1.732051
