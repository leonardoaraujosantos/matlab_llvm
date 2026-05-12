% Touchstone v1 .s2p reader smoke test.
%
% Reads fixtures/rf/test_amp.s2p (2 freqs, MA format, 50 Ω
% reference) and verifies the parsed cube.

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
disp(data.Z0);                       % 50
disp(data.NumPorts);                 % 2

s11 = tsS11(data);
s12 = tsS12(data);
s21 = tsS21(data);
s22 = tsS22(data);
f   = tsFreqs(data);

% Scalar real-part subscripting on a complex matrix returns the
% real component (subscript1_s is complex-aware).
disp(s11(1));         % 0.2
disp(s21(1));         % 2.0
disp(f(1));           % 1e9
disp(s11(2));         % 0.3
disp(f(2));           % 2e9

% Feed the loaded params into the closed-form analyses to check the
% end-to-end pipeline.  At freq 1, gamma_in (with matched 50 Ω load)
% reduces to s11.  VSWR(0.2) = 1.5.
gin = gammaIn(s11, s12, s21, s22, 50.0, 50.0);
v   = vswr(gin);
disp(v(1));           % 1.5

% Rollett K for this synthetic device at freq 1:
%   |s11|² = 0.04, |s22|² = 0.0025, Δ = 0.2·0.05 - 0.1·2 = -0.19
%   |Δ|² = 0.0361
%   K = (1 - 0.04 - 0.0025 + 0.0361) / (2 · |0.1·2|)
%     = 0.9936 / 0.4
%     = 2.484
K = stabilityK(s11, s12, s21, s22);
disp(K(1));           % 2.484
