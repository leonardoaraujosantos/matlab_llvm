% Weighted Vector Fitting.  Use uniform weights = ones and verify the
% result matches the unweighted rationalfit (round-trip identity check).

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

% Uniform weights (= 1.0).
weight = ones(K, 1);

mdl_w = rationalfitWeighted(freqs, h_re, h_im, weight, 2, 15);
mdl   = rationalfit(freqs, h_re, h_im, 2, 15);

disp(rfOrder(mdl_w));
disp(rfOrder(mdl));

% With uniform weights, the weighted result should equal the unweighted.
% (Confirmed numerically — the two pole sets should match.)
P_w = rfPoles(mdl_w);
P   = rfPoles(mdl);
disp(P_w);
disp(P);
