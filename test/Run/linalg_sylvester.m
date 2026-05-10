% 3-arg Sylvester equation — A·X + X·B + C = 0  (MATLAB's `lyap(A, B, C)`).
% Tier-1.4 follow-on of CST roadmap §2.4.

% --- 1. 1×1 closed form: a·x + x·b + c = 0  →  x = -c/(a + b).
A1 = [-2];
B1 = [-3];
C1 = [10];
X1 = lyap(A1, B1, C1);
disp(X1(1,1));         % -10 / (-5) = 2

% --- 2. 2×2 by 2×2 case. A diagonal, B diagonal — closed form
%   x_ij = -c_ij / (a_ii + b_jj).
A2 = [-1 0; 0 -2];
B2 = [-3 0; 0 -4];
C2 = [4 6; 8 12];
X2 = lyap(A2, B2, C2);
% Expected: x_11 = -4/(-4) = 1, x_12 = -6/(-5) = 1.2,
%           x_21 = -8/(-5) = 1.6, x_22 = -12/(-6) = 2.
disp(X2(1,1));         % 1
disp(X2(1,2));         % 1.2
disp(X2(2,1));         % 1.6
disp(X2(2,2));         % 2

% --- 3. Asymmetric shape: A is 2×2, B is 3×3, C is 2×3 → X is 2×3.
A3 = [-2 0; 0 -3];
B3 = [-1 0 0; 0 -4 0; 0 0 -5];
C3 = [3 6 7; 4 7 8];
X3 = lyap(A3, B3, C3);
% x_ij = -c_ij / (a_ii + b_jj).
%   x(1,1) = -3 / (-2 + -1) = 1
%   x(1,2) = -6 / (-2 + -4) = 1
%   x(1,3) = -7 / (-2 + -5) = 1
%   x(2,1) = -4 / (-3 + -1) = 1
%   x(2,2) = -7 / (-3 + -4) = 1
%   x(2,3) = -8 / (-3 + -5) = 1
disp(X3(1,1));
disp(X3(1,2));
disp(X3(1,3));
disp(X3(2,1));
disp(X3(2,2));
disp(X3(2,3));

% --- 4. Residual self-consistency on a non-diagonal case.
%   The Sylvester equation A·X + X·B + C = 0 must hold to within
%   round-off. Sum the absolute residuals — should be ~1e-15.
A4 = [-1 2; 0 -3];
B4 = [-1 -2; 0 -2];
C4 = [1 2; 3 4];
X4 = lyap(A4, B4, C4);
R = A4 * X4 + X4 * B4 + C4;
res = abs(R(1,1)) + abs(R(1,2)) + abs(R(2,1)) + abs(R(2,2));
% Print the rounded magnitude (always 0 because residual is below
% disp's default precision).
disp(round(res * 1e10));
