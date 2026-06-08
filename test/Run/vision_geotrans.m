% Computer Vision Toolbox Tier-2 — geometric transform estimation.
%   estgeotform2d (RANSAC affine fit, robust to outliers) recovers a known
%   transform; estimateFundamentalMatrix satisfies the epipolar constraint.
P1 = [10 20; 50 30; 30 70; 80 80; 15 60; 65 25; 40 45; 70 55];
P2 = P1 * 1.2;                 % scale 1.2 ...
P2(:,1) = P2(:,1) + 10;        % ... + tx = 10
P2(:,2) = P2(:,2) - 5;         % ... + ty = -5
P1o = [P1; 5 5; 90 10];        % + 2 gross outliers
P2o = [P2; 33 77; 12 88];

T = estgeotform2d(P1o, P2o, 'affine');
fprintf('recovered a=%.2f d=%.2f tx=%.1f ty=%.1f\n', T(1,1), T(2,2), T(3,1), T(3,2));
q = [30 40 1] * T;             % apply transform to a point
fprintf('map(30,40) -> (%.1f, %.1f)\n', q(1), q(2));   % (46, 43)

% Fundamental matrix: epipolar residual p2'*F*p1 ~ 0 (scalar form, point 1).
F = estimateFundamentalMatrix(P1, P2);
fprintf('F is 3x3: %.0f\n', size(F,1) * size(F,2));    % 9
x1 = P1(1,1); y1 = P1(1,2); x2 = P2(1,1); y2 = P2(1,2);
e = x2*(F(1,1)*x1 + F(1,2)*y1 + F(1,3)) ...
  + y2*(F(2,1)*x1 + F(2,2)*y1 + F(2,3)) ...
  +    (F(3,1)*x1 + F(3,2)*y1 + F(3,3));
fprintf('epipolar residual ~ 0: %.0f\n', round(abs(e) * 100));   % 0
