% Model-object CST multi-return splitters: kalman (gain + covariance) and
% margin (gain/phase margins + crossover frequencies).
A = [0 1; 0 0]; B = [0; 1]; C = [1 0]; D = 0;
sys = ss(A, B, C, D);
[kest, L, P] = kalman(sys, 0.01, 0.1);     % scalar noise covariances
fprintf('L %.4f %.4f\n', L(1), L(2));
fprintf('P %.4f %.4f\n', P(1,1), P(2,2));

% L(s) = 4 / (s^2 + 2s): phase margin ~51.8 deg at gain crossover ~1.57 rad/s.
G = ss([0 1; 0 -2], [0; 4], [1 0], 0);
[Gm, Pm, Wcg, Wcp] = margin(G);
fprintf('Pm %.1f Wcp %.3f\n', Pm, Wcp);
