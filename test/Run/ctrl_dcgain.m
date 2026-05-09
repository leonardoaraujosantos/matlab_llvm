% Tier 3 — state-space DC gain.
% dcgain_ss(A, B, C, D) = D - C inv(A) B
% Returns a p x m matrix; for SISO this is 1x1.

% --- 1. SISO 1st-order with feedthrough.
% xdot = -2 x + u, y = 3 x + 0.5 u → G(0) = 0.5 + 3/2 = 2.0.
A = [0-2];
B = [1];
C = [3];
D = [0.5];
disp('1st-order DC gain (closed form 2.0):');
disp(dcgain_ss(A, B, C, D));

% --- 2. Mass-spring-damper. xdot = [0 1; -k/m -c/m] x + [0; 1/m] u,
% y = [1 0] x. G(0) = 1/k. Pick m = 1, k = 4, c = 0.6 → G(0) = 0.25.
A2 = [0, 1; 0-4, 0-0.6];
B2 = [0; 1];
C2 = [1, 0];
D2 = [0];
disp('mass-spring-damper DC gain (1/k = 0.25):');
disp(dcgain_ss(A2, B2, C2, D2));

% --- 3. After LQR — closed-loop DC gain shifts because of the gain
% feedback term. Verify via direct computation.
Q = [1 0; 0 0.1];
R = [1];
K = lqr(A2, B2, Q, R);
Acl = A2 - B2 * K;
disp('closed-loop DC gain (LQR-feedback shifts the static response):');
disp(dcgain_ss(Acl, B2, C2, D2));

% --- 4. Singular A (integrator at the origin) — DC gain is unbounded.
% dcgain_ss returns 0x0 in this case (the inv() fails); user's job to
% check.
A3 = [0, 1; 0, 0];
B3 = [0; 1];
C3 = [1, 0];
D3 = [0];
out = dcgain_ss(A3, B3, C3, D3);
disp('numel of dcgain output for an integrator (0 → unbounded gain):');
disp(numel(out));
