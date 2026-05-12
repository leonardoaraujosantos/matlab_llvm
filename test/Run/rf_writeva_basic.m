% Tier-1 Verilog-A export — smoke test.
%
% Build the canonical 2-real-pole rational
%   H(s) = 5e9 / (s + 1e9) + 1e10 / (s + 3e9) + 0.5
% fit with rationalfit, write to a .va file, verify the return value
% is 1 (success).  The byte-level content of the .va is exercised by
% examples/verilog_a/rf_rational_writeva.m (with an inspected golden).

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
ok  = writeVerilogA(mdl, "/tmp/_rf_writeva_basic.va");
disp(ok);              % 1
