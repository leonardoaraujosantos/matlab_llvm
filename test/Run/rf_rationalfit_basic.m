% Vector Fitting smoke test (RF-Tier-3.1).
%
% Build a known 2-pole rational
%   H(s) = 5e9 / (s + 1e9) + 1e10 / (s + 3e9) + 0.5
% sampled at 50 log-spaced frequencies between 1e7 and 1e10 Hz, then
% call `rationalfit` with nPoles=2 + nIter=15.  The fitter should
% recover the poles, residues, direct term, and a relative RMS fit
% error far below 1e-6 (we ship a real-pole-only MVP, so this exact-
% rational case is the canonical test).

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

disp(rfOrder(mdl));        % 2
% FitError is the relative RMS reconstruction error; for an
% exact-rational target hit by a matching-order fit it should be at
% machine-precision scale.  We don't disp the raw float (its last
% digits drift across rebuilds); we display a comparison threshold.
err = rfFitError(mdl);
disp(err < 1.0e-6);         % 1 (true)

% Poles come back sorted by ascending real part (matlab_eig's
% convention).  Expect -3e9 first, then -1e9.
P = rfPoles(mdl);
disp(P(1));                 % -3e9
disp(P(2));                 % -1e9

disp(rfD(mdl));             % 0.5

% freqresp round-trip — evaluate the fitted model at NEW frequencies
% and check it matches the analytic target.  Display the 3-row
% complex column; values match the analytic target to 4 decimals.
test_f = [1.0e8; 1.0e9; 5.0e9];
H = freqresp(mdl, test_f);
disp(H);
