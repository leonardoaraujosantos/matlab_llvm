% rfDelayEstimate + rfApplyDelay + rfPassivityEnforce smoke test.
%
% Build a transport-delayed dataset: H(jω) = exp(−jωτ) at τ = 1 ns.
% rfDelayEstimate should recover τ ≈ 1 ns from the phase slope.
% After rfApplyDelay with the recovered τ, the result should be ≈ 1
% (constant, no phase variation).

tau_target = 1.0e-9;       % 1 ns transport delay
K = 20;
freqs = zeros(K, 1);
for k = 1:K
    freqs(k) = 1.0e8 + (k - 1) * 5.0e7;
end
h_re = zeros(K, 1);
h_im = zeros(K, 1);
for k = 1:K
    w = 2.0 * 3.141592653589793 * freqs(k);
    phi = -w * tau_target;       % H(jω) = exp(-jωτ) = cos(-ωτ) + j·sin(-ωτ)
    h_re(k) = cos(phi);
    h_im(k) = sin(phi);
end
tau_est = rfDelayEstimate(freqs, h_re, h_im);
disp(tau_est);                 % ~1e-9

% Apply (remove) the estimated delay.  Result should be ~1.0 throughout.
dd = rfApplyDelay(freqs, h_re, h_im, tau_est);
disp(dd.Delay);                % tau_est

% Passivity enforce: synthesize a rational with |H| > 1 at some freq,
% pass through enforcer, verify the resulting max|H| ≤ 1.
% Build a model with D = 2 (clearly non-passive: at ω=∞ H = 2).
K2 = 50;
ln10 = 2.302585092994046;
fs = zeros(K2, 1);
hr = zeros(K2, 1);
hi = zeros(K2, 1);
for k = 1:K2
    t = (k - 1.0) / (K2 - 1.0);
    fs(k) = exp((7.0 + t * 3.0) * ln10);
end
for k = 1:K2
    w = 2.0 * 3.141592653589793 * fs(k);
    a = 1.0e9;
    den = a*a + w*w;
    hr(k) = a / den + 2.0;     % +D = 2 makes it non-passive
    hi(k) = -w / den;
end
mdl = rationalfit(fs, hr, hi, 1, 8);
disp(rfD(mdl));                 % ~2 (the direct term)

% Enforce passivity over the data band.
mdl2 = rfPassivityEnforce(mdl, 1.0e7, 1.0e10);
disp(rfD(mdl2));                % scaled down below 1
