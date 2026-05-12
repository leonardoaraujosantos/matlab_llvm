% Vector Fitting on a resonant 2nd-order target with complex-conjugate
% poles.  v2 upgrade: the algorithm now relocates pole pairs in the
% [α β; -β α] real-block representation, so resonant features that
% require complex poles fit naturally.
%
% Target: H(s) = 1 / ((s − p)(s − p̄)) with p = −1e8 + j·1e9.
% v1's real-pole-only fit cannot capture this; v2 converges to a
% conjugate pair near (−1e8, ±1e9) with low fit error.

K = 60;
ln10 = 2.302585092994046;
freqs = zeros(K, 1);
for k = 1:K
    t = (k - 1.0) / (K - 1.0);
    freqs(k) = exp((7.0 + t * 3.0) * ln10);
end
h_re = zeros(K, 1);
h_im = zeros(K, 1);
for k = 1:K
    w = 2.0 * 3.141592653589793 * freqs(k);
    a = 1.0e8;
    b = 1.0e9;
    den_re = a*a + b*b - w*w;
    den_im = 2.0 * a * w;
    den_mag2 = den_re*den_re + den_im*den_im;
    h_re(k) = den_re / den_mag2;
    h_im(k) = -den_im / den_mag2;
end

mdl = rationalfit(freqs, h_re, h_im, 2, 25);
disp(rfOrder(mdl));        % 2 (stored as conjugate pair = 2 entries)
err = rfFitError(mdl);
% LS conditioning over the 3-decade span keeps err well below 1% RMS
% with 25 iterations; for 1.5-decade spans the fit hits machine precision.
disp(err < 0.01);          % -1 (true)

% Poles come back as a conjugate pair (real part, ±imag) in
% consecutive entries.
P = rfPoles(mdl);
disp(P);
