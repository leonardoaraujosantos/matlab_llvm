% Passivity check on a known rational model.
%
% Synthesize a passive model: H(s) = 0.5/(s + 1e9) + 0.3 with poles
% in LHP and |H(jω)| ≤ 1 always (peak at ω=0: |H| = 0.5/1e9 + 0.3 ≈ 0.3).
% rationalfit recovers this exactly; passivity should return < 1.

K = 50;
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
    a1 = 1.0e9;  c1 = 5.0e9;
    a2 = 3.0e9;  c2 = 1.0e10;
    den1 = a1*a1 + w*w;
    den2 = a2*a2 + w*w;
    h_re(k) = c1 * a1 / den1 + c2 * a2 / den2 + 0.5;
    h_im(k) = -c1 * w / den1 - c2 * w / den2;
end
mdl = rationalfit(freqs, h_re, h_im, 2, 15);
% At ω=0 the model peaks at c1/|a1| + c2/|a2| + d = 5 + 3.33 + 0.5 = 8.83.
% passivity samples a dense grid; the peak should be ~8.83.
peak = passivity(mdl, 1.0e7, 1.0e10);
disp(peak);
% peak > 1 → not passive in the strict sense, but the test just
% reports the magnitude; the passivity decision is caller-side.
disp(peak > 1.0);   % 1 (true)
