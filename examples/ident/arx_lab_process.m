% arx_lab_process.m — System Identification Toolbox Tier-1 headline.
%
% The canonical "estimate a simple model from laboratory process data"
% workflow (User's Guide §4 "Estimating Simple Models from Real
% Laboratory Process Data"), self-contained: we synthesise input/output
% records from a second-order discrete process with a measured input and
% a small additive disturbance, then run the whole Tier-1 loop —
%
%   iddata  ->  arx  ->  compare (NRMSE fit %)  ->  ss / pole analysis
%
% reusing the already-shipped Control System Toolbox `pole` on the
% identified state-space model.

% ----- 1.  Synthesise a lab record --------------------------------------
% True plant:  y(t) = 1.5 y(t-1) - 0.7 y(t-2) + 1.0 u(t-1) + 0.5 u(t-2)
% (a lightly-damped second-order process; poles at 0.75 +/- 0.36i).
N  = 600;
u  = zeros(N, 1);
e  = zeros(N, 1);
sd = 271828;
for k = 1:N
    sd   = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd / 2147483648 - 0.5) + 0.4 * sin(0.05 * k);  % rich input
    sd   = mod(sd * 1103515245 + 12345, 2147483648);
    e(k) = (sd / 2147483648 - 0.5) * 0.10;                     % disturbance
end
y = zeros(N, 1);
for k = 3:N
    y(k) = 1.5 * y(k-1) - 0.7 * y(k-2) ...
         + 1.0 * u(k-1) + 0.5 * u(k-2) + e(k);
end

Ts = 0.08;                       % the classic dryer-data cadence
z  = iddata(y, u, Ts);
fprintf('Lab record: %.0f samples at Ts = %.2f s\n', N, Ts);

% ----- 2.  Estimate an ARX model ----------------------------------------
% Orders [na nb nk] = [2 2 1]: second-order denominator, two B taps,
% one-sample input delay.  arx solves the regressor least-squares.
m = arx(z, [2 2 1]);
fprintf('\nIdentified A(q) = [1, %.3f, %.3f]\n', m.A(2), m.A(3));
fprintf('Identified B(q) = [%.3f, %.3f, %.3f]\n', m.B(1), m.B(2), m.B(3));
fprintf('Innovations variance V = %.5f\n', m.NoiseVariance);

% ----- 3.  Validate ------------------------------------------------------
fit = compare(z, m);
fprintf('\nSimulation fit (NRMSE) = %.2f %%\n', fit);

% Quality metrics from the loss + parameter count.
fprintf('FPE = %.5f   AIC = %.2f\n', fpe(m), aic(m));

% ----- 4.  Reuse the model for analysis (CST bridge) --------------------
% Convert the identified idpoly to a discrete state-space model and read
% its poles with the shipped Control System Toolbox `pole`.
sys = ss(m);
p   = pole(sys);
fprintf('\nIdentified poles (z-plane):\n');
disp(p);
fprintf('Sample time of ss(model) = %.2f s (discrete)\n', sys.Ts);
