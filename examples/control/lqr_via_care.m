% LQR via care — Tier-1.5 demo. The infinite-horizon LQR on a continuous
% LTI plant
%      xdot = A x + B u,  cost J = integral_0^infty (x' Q x + u' R u) dt
% has the closed-form solution
%      X = care(A, B, Q, R)
%      K = R^{-1} B' X
% with the closed-loop dynamics A_cl = A - B K stable. Tier 2 will wrap
% these three lines in a top-level `lqr(A, B, Q, R)` MATLAB function.

% --- 1. Double integrator. Q = I, R = 1.
A = [0 1; 0 0];
B = [0; 1];
Q = [1 0; 0 1];
R = [1];

% Solve the Riccati equation.
X = care(A, B, Q, R);
disp('Riccati X (closed form: [sqrt(3) 1; 1 sqrt(3)]):');
disp(X);

% LQR gain.
K = inv(R) * (B' * X);
disp('LQR gain K (closed form: [1 sqrt(3)]):');
disp(K);

% Closed-loop poles. Should land at -sqrt(3)/2 +- j 0.5.
Acl = A - B * K;
disp('closed-loop poles (real and imag separately):');
disp(real(eig(Acl)));
disp(imag(eig(Acl)));

% --- 2. Stable plant — verify residual after LQR.
% Random-ish 2x2 stable A with diagonal-dominant input matrix.
A2 = [0-2, 1; 0.5, 0-3];
B2 = [1; 0];
Q2 = [4 0; 0 1];
R2 = [0.5];
X2 = care(A2, B2, Q2, R2);
% Riccati residual should be zero.
Res = A2' * X2 + X2 * A2 - X2 * B2 * inv(R2) * B2' * X2 + Q2;
disp('Riccati residual (entries should be ~0):');
disp(Res);

% --- 3. The optimal cost from initial condition x0 = [1; 0]:
%       J* = x0' X x0.
x0 = [1; 0];
J = x0' * X * x0;
disp('optimal LQR cost from x0 = [1; 0] (= sqrt(3) for the integrator):');
disp(J);
