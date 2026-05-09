% Tier 2.3 follow-on + 2.4 — generalised input simulation lsim_ss
% and stability margins gain_margin / phase_margin. All sit on the
% Tier-1.3 expm + Tier-2.2 c2d + Tier-2.4 bode_ss machinery.

% --- 1. lsim_ss with constant unit input matches step_ss exactly.
A = [0-1];
B = [1];
C = [1];
D = [0];
N  = 6;
dt = 0.1;
u  = [1; 1; 1; 1; 1; 1];      % column of ones
y_lsim = lsim_ss(A, B, C, D, u, dt);
y_step = step_ss(A, B, C, D, dt, N);
fprintf('y_lsim(3) = %.6f\n', y_lsim(3, 1));   % matches y_step(3)
fprintf('y_step(3) = %.6f\n', y_step(3, 1));
fprintf('y_lsim(6) = %.6f\n', y_lsim(6, 1));
fprintf('y_step(6) = %.6f\n', y_step(6, 1));

% --- 2. lsim_ss with a ramp input u(t) = t for the lowpass.
%   Closed form for first-order  xdot = -x + u, y = x  with u(t) = t,
%   x(0) = 0:  y(t) = t - 1 + exp(-t).
ur = [0; 0.1; 0.2; 0.3; 0.4; 0.5];
y_ramp = lsim_ss(A, B, C, D, ur, dt);
fprintf('ramp y(0)   = %.6f\n', y_ramp(1, 1));   % 0
fprintf('ramp y(0.5) = %.6f\n', y_ramp(6, 1));   % 0.5 - 1 + exp(-0.5) ~ 0.1065

% --- 3. Type-1 plant L(s) = 4 / (s * (s + 2)) — phase margin.
%   SS:  A = [0 1; 0 -2], B = [0; 1], C = [4 0], D = 0.
A2 = [0 1; 0, 0-2];
B2 = [0; 1];
C2 = [4, 0];
D2 = [0];
% Dense logspaced grid won't lower (Sema-types `none`); use a hand-built
% linear grid covering w_c ~ 1.572.
w = 0.1 + 0.005 * (0:399)';
Pm = phase_margin(A2, B2, C2, D2, w);
fprintf('phase margin = %.4f deg\n', Pm);          % ~ 51.83

% --- 4. First-order plant — gain margin is +Inf (phase asymptotes
% to -90, never reaches -180). Build a w grid with hand-spaced points.
wd = [0.01; 0.1; 1; 10; 100];
Gm = gain_margin(A, B, C, D, wd);
% Print as logical comparison (Inf > 1e10).
if Gm > 1e10
    fprintf('gain margin = Inf (correct for first-order)\n');
else
    fprintf('gain margin = %.4f\n', Gm);
end
