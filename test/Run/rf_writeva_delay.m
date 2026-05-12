% Tier-1 Verilog-A export — bulk-delay path.
%
% rationalfit returns a struct without a Delay field; users can set
% Delay via direct struct field assignment before calling writeVerilogA
% (this is the same shape rfPassivityEnforce produces).  The emitter
% surfaces Delay as a parameterized absdelay() wrap around the section
% sum.

K = 30;
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
    den1 = a1*a1 + w*w;
    h_re(k) = c1 * a1 / den1 + 0.3;
    h_im(k) = -c1 * w / den1;
end

mdl = rationalfit(freqs, h_re, h_im, 1, 15);
mdl.Delay = 2.5e-9;
ok = writeVerilogA(mdl, "/tmp/_rf_writeva_delay.va");
disp(ok);              % 1
