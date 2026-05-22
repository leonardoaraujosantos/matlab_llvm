% Tier 3 — `stepinfo(y, t)` step-response metrics.
% Returns 1 × 5 row [RiseTime, SettlingTime, Overshoot, Peak, PeakTime].
%
% Closed-form values for a 1st-order Hurwitz plant y(t) = K (1 − e^{−t/τ}):
%   RiseTime     = τ · log(9)         ≈ 1.0986·τ
%   SettlingTime = −τ · log(0.02)     ≈ 3.912·τ
%   Overshoot    = 0                  (monotone)
%   Peak         = K  at t = ∞        (PeakTime ≈ last sample)

% --- 1. SISO 1st-order. K = 1, τ = 0.5.
t = (0:599)' * 0.01;
y = 1 - exp(-t / 0.5);
si = stepinfo(y, t);
fprintf('1st-order RiseTime    = %.4f (closed form 1.0986)\n', si.RiseTime);
fprintf('1st-order SettlingTime = %.4f (closed form 1.957)\n', si.SettlingTime);
fprintf('1st-order Overshoot   = %.4f (closed form 0)\n', si.Overshoot);
fprintf('1st-order Peak        = %.4f (closed form 1.0)\n', si.Peak);

% --- 2. Step response of a state-space underdamped 2nd order plant.
% A = [0, 1; -wn^2, -2 zeta wn], wn = 5, zeta = 0.3 → expect overshoot.
wn = 5;
zeta = 0.3;
A = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B = [0; wn*wn];
C = [1, 0];
D = [0];
N = 600;
dt = 0.01;
y2 = step_ss(A, B, C, D, dt, N);
t2 = (0:N-1)' * dt;
si2 = stepinfo(y2, t2);
fprintf('2nd-order RiseTime    = %.4f\n', si2.RiseTime);
fprintf('2nd-order SettlingTime = %.4f\n', si2.SettlingTime);
fprintf('2nd-order Overshoot   = %.4f%% (expected ~37%% for zeta=0.3)\n', ...
        si2.Overshoot);
fprintf('2nd-order Peak        = %.4f\n', si2.Peak);
fprintf('2nd-order PeakTime    = %.4f\n', si2.PeakTime);
