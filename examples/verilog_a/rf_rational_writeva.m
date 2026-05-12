% Vector-fit a resonant RF S-parameter target with a complex-conjugate
% pole pair and a real direct term, then export the fitted model as a
% Verilog-A behavioral module via the rfmodel.rational/writeVerilogA
% path (Tier-1 of docs/verilog_a_plan.md).
%
% The .m source itself runs end-to-end through the matlab_llvm
% compiler — sanity-check the numerical behavior of `freqresp` /
% `passivity` here, then drop the emitted .va into Cadence Spectre /
% ngspice / Xyce for transistor-level co-simulation.

K = 60;
ln10 = 2.302585092994046;
freqs = zeros(K, 1);
for k = 1:K
    t = (k - 1.0) / (K - 1.0);
    freqs(k) = exp((7.0 + t * 3.0) * ln10);   % 1e7 .. 1e10 Hz
end

h_re = zeros(K, 1);
h_im = zeros(K, 1);
for k = 1:K
    w  = 2.0 * 3.141592653589793 * freqs(k);
    a  = 1.0e8;
    w0 = 1.0e9;
    den_re = (w0*w0 - w*w);
    den_im = 2.0*a*w;
    n      = w0 * w0;
    dmag2  = den_re*den_re + den_im*den_im;
    h_re(k) = (n*den_re) / dmag2 + 0.1;
    h_im(k) = (-n*den_im) / dmag2;
end

mdl = rationalfit(freqs, h_re, h_im, 2, 25);

% Sanity-check: report fit order.
disp(rfOrder(mdl));                  % 2

ok = writeVerilogA(mdl, "rf_rational.va");
disp(ok);                            % 1
