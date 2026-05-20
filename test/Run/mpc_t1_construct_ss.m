% MPC Tier-1 §2.4 — mpc() constructor against a discrete ss.
% Verifies the classdef instantiates and caches the precomputed
% prediction matrices.

A = [0.8, 0.1; 0.0, 0.7];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);   % Ts = 0.1

obj = mpc(sys_d, 5, 2);

% Horizon round-trip.
fprintf('p = %.0f\n', obj.p);
fprintf('m = %.0f\n', obj.m);
fprintf('Ts = %.4f\n', obj.Ts);

% Sx(1, :) = C * A = [0.8, 0.1].
fprintf('Sx(1,1) = %.6f\n', obj.Sx(1, 1));
fprintf('Sx(1,2) = %.6f\n', obj.Sx(1, 2));

% Sx(2, :) = C * A^2 = [0.64, 0.15].
fprintf('Sx(2,1) = %.6f\n', obj.Sx(2, 1));
fprintf('Sx(2,2) = %.6f\n', obj.Sx(2, 2));

% Su1(1, :) = C * I * B = 1.
fprintf('Su1(1,1) = %.6f\n', obj.Su1(1, 1));

% Su(1, 1) = C * I * B = 1.
fprintf('Su(1,1) = %.6f\n', obj.Su(1, 1));

% Cached Hessian + Cholesky factor.
fprintf('H(1,1) = %.4f\n', obj.H(1, 1));
fprintf('R(1,1) = %.4f\n', obj.R(1, 1));

% Kalman gain L (continuous-form, both kernels would work for this Schur-stable
% discrete plant; nonzero is the structural sanity check).
fprintf('L(1,1) = %.4f\n', obj.L(1, 1));
