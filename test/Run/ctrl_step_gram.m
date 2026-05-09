% Tier 2.3 / 3.4 — state-space step response (`step_ss`) and gramians
% (`gram_c` / `gram_o`). Both build on Tier-1.4 lyap and Tier-2.2 c2d.

% --- 1. First-order lowpass step response.
%   xdot = -(1/tau) x + (1/tau) u, y = x.
%   Closed form:  y(t) = 1 - exp(-t/tau).
tau = 0.5;
A = [0-1/tau];
B = [1/tau];
C = [1];
D = [0];
y = step_ss(A, B, C, D, 0.05, 6);
fprintf('%.6f\n', y(1, 1));     % 0.000000  (relaxed initial state)
fprintf('%.6f\n', y(2, 1));     % 0.095163  (1 - exp(-0.05/0.5))
fprintf('%.6f\n', y(3, 1));     % 0.181269  (1 - exp(-0.10/0.5))
fprintf('%.6f\n', y(6, 1));     % 0.393469  (1 - exp(-0.25/0.5))

% --- 2. Second-order plant — observe the rise toward steady state.
%   A = [-1 1; 0 -2], B = [0; 1], C = [1 0], D = 0.
%   Steady-state value y_ss = -C A^{-1} B = 0.5.
A2 = [0-1, 1; 0, 0-2];
B2 = [0; 1];
C2 = [1, 0];
D2 = [0];
y2 = step_ss(A2, B2, C2, D2, 0.1, 200);
% Print three checkpoints.
fprintf('%.6f\n', y2(1, 1));      % 0
fprintf('%.6f\n', y2(50, 1));     % rising toward 0.5
fprintf('%.6f\n', y2(200, 1));    % near 0.5

% --- 3. Controllability gramian via gram_c.
%   For A = diag(-1, -2), B = [1; 1]:
%     Wc[i,j] = (B B')[i,j] / (-A[i,i] - A[j,j])
%     Wc = [[0.5, 1/3], [1/3, 0.25]].
A3 = [0-1, 0; 0, 0-2];
B3 = [1; 1];
Wc = gram_c(A3, B3);
fprintf('%.6f\n', Wc(1, 1));    % 0.500000
fprintf('%.6f\n', Wc(1, 2));    % 0.333333
fprintf('%.6f\n', Wc(2, 1));    % 0.333333
fprintf('%.6f\n', Wc(2, 2));    % 0.250000

% --- 4. Observability gramian.
%   For A = diag(-1, -2), C = [1, 1]:
%     Wo same shape as Wc above (C' C = [1 1; 1 1]).
C3 = [1, 1];
Wo = gram_o(A3, C3);
fprintf('%.6f\n', Wo(1, 1));    % 0.500000
fprintf('%.6f\n', Wo(2, 2));    % 0.250000

% --- 5. H2 system norm via gramians.
%   ||G||_2 = sqrt(trace(B' Wo B)) = sqrt(trace(C Wc C')).
%   For (A, B, C) above and B' Wo B as a scalar:
%     B' Wo B = [1 1] * [[0.5 1/3]; [1/3 0.25]] * [1; 1]
%             = [1 1] * [0.5+1/3; 1/3+0.25] = (5/6) + (7/12) = 17/12.
%     ||G||_2 = sqrt(17/12) ~ 1.190.
H2_sq = B3' * Wo * B3;
fprintf('%.6f\n', sqrt(H2_sq(1, 1)));   % 1.190238
