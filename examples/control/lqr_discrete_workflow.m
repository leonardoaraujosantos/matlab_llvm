% Tier 2 demo — full continuous-time LQR design + ZOH discretisation
% pipeline using lqr (Tier 2.4) and c2d (Tier 2.2). This is the
% standard workflow for designing a digital controller from a
% continuous plant model.

% --- 1. Continuous plant: simple double integrator (e.g. position
% control of a frictionless mass under force).
%   xdot = [0 1; 0 0] x + [0; 1] u,  output y = x(1) (position).
A = [0 1; 0 0];
B = [0; 1];

% --- 2. LQR design — closed-form K = [1, sqrt(3)] for Q = I, R = 1.
Q = [1 0; 0 1];
R = [1];
K = lqr(A, B, Q, R);
disp('continuous LQR gain K:');
disp(K);

% --- 3. Closed-loop continuous-time A.
Acl = A - B * K;
disp('closed-loop poles (real then imag):');
disp(real(eig(Acl)));    % -0.866 each (with non-zero imag)
disp(imag(eig(Acl)));    % +- 0.5

% --- 4. Discretise the closed-loop plant at Ts = 0.05 s for digital
% implementation. The same controller K applies on the discrete
% sample grid:  x[k+1] = Ad_cl x[k]  with  Ad_cl = c2d(Acl, B*0, Ts)
% but since closed-loop has no exogenous input, just discretise Acl
% with B = zeros — or use the open-loop discretisation and apply
% u[k] = -K x[k] in the digital control loop.
Ts = 0.05;
[Ad, Bd] = c2d(A, B, Ts);
disp('discrete plant Ad:');
disp(Ad);
disp('discrete plant Bd:');
disp(Bd);

% --- 5. Discrete closed-loop:  x[k+1] = (Ad - Bd K) x[k].
Ad_cl = Ad - Bd * K;
disp('discrete closed-loop poles (must be inside unit circle):');
e = eig(Ad_cl);
disp(real(e));
disp(imag(e));
% Squared magnitudes < 1.
disp('|e|^2 < 1:');
disp(real(e) .* real(e) + imag(e) .* imag(e));

% --- 6. Three-step rollout from x0 = [1; 0].  Position decays toward
% zero under LQR control. (Recompute Ad_cl from raw matrices so the
% Sema type lattice doesn't get crossed by the upstream complex-eig
% returns.)
Ad_cl2 = Ad - Bd * K;
x0 = [1; 0];
x1 = Ad_cl2 * x0;
x2 = Ad_cl2 * x1;
x3 = Ad_cl2 * x2;
fprintf('step 1 position = %.6f\n', x1(1, 1));
fprintf('step 2 position = %.6f\n', x2(1, 1));
fprintf('step 3 position = %.6f\n', x3(1, 1));
