% armax_refine.m — System Identification Toolbox Tier-2 headline.
%
% The canonical "refine a model's noise description" workflow (User's
% Guide §6 "Estimate Models Using armax").  A process driven by a
% measured input AND coloured measurement noise is first fit with a
% plain ARX model (which assumes white equation noise and so must
% distort the dynamics to absorb the colour), then with an ARMAX model
% whose C(q) polynomial captures the noise colour directly.  The ARMAX
% fit recovers the true dynamics and leaves whiter residuals.
%
%   iddata  ->  arx (baseline)  ->  armax (PEM refine)  ->  resid / compare
%
% reusing the already-shipped Optimization Toolbox `lsqnonlin` as the
% prediction-error-minimisation engine.

% ----- 1.  Synthesise an ARMAX record -----------------------------------
% True system:  A(q) = [1 -0.5],  B(q) = [0 1.0],  C(q) = [1 0.7]
%   y(t) = 0.5 y(t-1) + 1.0 u(t-1) + e(t) + 0.7 e(t-1)
N  = 1000;
e  = zeros(N, 1);
u  = zeros(N, 1);
sd = 161803;
for k = 1:N
    sd   = mod(sd * 1103515245 + 12345, 2147483648);
    e(k) = (sd / 2147483648 - 0.5) * 0.4;          % innovation
    sd   = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd / 2147483648 - 0.5);            % PRBS-like input
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5 * y(k-1) + 1.0 * u(k-1) + e(k) + 0.7 * e(k-1);
end
z = iddata(y, u, 1);
fprintf('ARMAX record: %.0f samples, Ts = %.0f s\n', N, z.Ts);

% ----- 2.  Baseline ARX --------------------------------------------------
ma = arx(z, [1 1 1]);
ra = resid(ma, z);
fprintf('\nARX  A(q) = [1, %.3f]   B2 = %.3f\n', ma.A(2), ma.B(2));
fprintf('ARX  max residual autocorr = %.3f  (coloured → large)\n', ra(1));

% ----- 3.  ARMAX refinement (PEM) ---------------------------------------
mx = armax(z, [1 1 1 1]);
rx = resid(mx, z);
fprintf('\nARMAX A(q) = [1, %.3f]   B2 = %.3f   C2 = %.3f\n', ...
        mx.A(2), mx.B(2), mx.C(2));
fprintf('ARMAX max residual autocorr = %.3f  (whitened)\n', rx(1));

% ----- 4.  Validation ----------------------------------------------------
fprintf('\nARX   simulation fit = %.1f %%\n', compare(z, ma));
fprintf('ARMAX simulation fit = %.1f %%\n', compare(z, mx));
fprintf('ARMAX FPE = %.5f   AIC = %.1f\n', fpe(mx), aic(mx));
