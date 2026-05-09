% Full SISO loop-shaping workflow — Tier 2 functional API end-to-end.
%
% Plant: type-1 servo  L(s) = 4 / (s * (s + 2)).  The classical CST
% workflow:
%   1) Inspect frequency response (bode_ss).
%   2) Compute stability margins (gain_margin / phase_margin).
%   3) Discretise for digital implementation (c2d).
%   4) Simulate the closed-loop time response (lsim_ss).

% --- 1. Plant in state-space form.  L(s) = 4 / (s*(s+2)).
A = [0 1; 0, 0-2];
B = [0; 1];
C = [4, 0];
D = [0];

% --- 2. Frequency response at four checkpoints.
w_check = [0.5; 1.0; 1.5; 2.0];
[mag, phase] = bode_ss(A, B, C, D, w_check);
fprintf('   w     |L|       phase\n');
fprintf('  ---   -----     ------\n');
fprintf('  0.5   %.4f   %.4f\n', mag(1, 1), phase(1, 1));
fprintf('  1.0   %.4f   %.4f\n', mag(2, 1), phase(2, 1));
fprintf('  1.5   %.4f   %.4f\n', mag(3, 1), phase(3, 1));
fprintf('  2.0   %.4f   %.4f\n', mag(4, 1), phase(4, 1));

% --- 3. Stability margins on a dense grid covering w_c ~ 1.572.
w_dense = 0.1 + 0.005 * (0:399)';
Pm = phase_margin(A, B, C, D, w_dense);
Gm = gain_margin (A, B, C, D, w_dense);
fprintf('\nphase margin Pm = %.4f deg\n', Pm);
if Gm > 1e10
    fprintf('gain margin  Gm = Inf (type-1 plant -> phase asymptotes to -180 only at infinity)\n');
else
    fprintf('gain margin  Gm = %.4f\n', Gm);
end

% --- 4. Discretise plant at Ts = 0.1 s.
[Ad, Bd] = c2d(A, B, 0.1);
fprintf('\ndiscrete plant Ad:\n');
disp(Ad);

% --- 5. Closed-loop simulation with unit-feedback step reference.
%   r[k] = 1, e[k] = r[k] - y[k], u[k] = e[k]  (unit-gain controller).
%   Build u as 30 step samples and simulate via lsim_ss with a
%   constant-1 input (open-loop, no feedback — to confirm the plant
%   integrates the input). True closed-loop simulation needs the
%   discrete recurrence x[k+1] = (Ad - Bd*K) x[k] + Bd*r[k]; that's a
%   Tier-2.1 follow-on once `feedback(sys1, sys2)` lands.
N = 30;
u = ones(N, 1);
y_open = lsim_ss(A, B, C, D, u, 0.1);
% Print three checkpoints of the step response.
fprintf('open-loop step y(t):  t=0 -> %.4f,  t=1.0 -> %.4f,  t=3.0 -> %.4f\n', ...
        y_open(1, 1), y_open(11, 1), y_open(N, 1));
