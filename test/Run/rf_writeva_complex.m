% Tier-1 Verilog-A export — complex-conjugate-pair smoke test.
%
% Build a target with a complex-conjugate pole pair so the emitter has
% to fold it into a real-coefficient biquad section.  Target:
%   H(s) = 1e18 / (s^2 + 2e8*s + 1e18) + 0.1
% which has poles at s = -1e8 +/- j*sqrt(1e18 - 1e16) ≈ -1e8 +/- j*9.95e8.

K = 40;
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
    a  = 1.0e8;          % real part of pole magnitude
    w0 = 1.0e9;          % natural freq
    den_re = (w0*w0 - w*w);
    den_im = 2.0*a*w;
    n      = w0 * w0;
    dmag2  = den_re*den_re + den_im*den_im;
    h_re(k) = (n*den_re) / dmag2 + 0.1;
    h_im(k) = (-n*den_im) / dmag2;
end

mdl = rationalfit(freqs, h_re, h_im, 2, 25);
ok  = writeVerilogA(mdl, "/tmp/_rf_writeva_complex.va");
disp(ok);              % 1
