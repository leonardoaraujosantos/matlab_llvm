% MPC Tier-6 §7.5 — review() sanity diagnostic.
% A correctly-built mpc obj should return 1; a deliberately broken
% one (zeros in the Hessian Cholesky's diagonal) returns 0.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
r = review(obj);
fprintf('review(obj) (must be 1): %.0f\n', r(1, 1));

% setEstimator round-trip.
L_new = [0.5; 0.0];
setEstimator(obj, L_new);
L_back = getEstimator(obj);
fprintf('setEstimator round-trip L(1,1) = %.4f\n', L_back(1, 1));
fprintf('setEstimator round-trip L(2,1) = %.4f\n', L_back(2, 1));
