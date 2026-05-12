% Stability factors on a textbook unconditionally-stable 2-port.
%
% Pick s11 = 0.3∠0, s12 = 0.05∠0, s21 = 2.0∠0, s22 = 0.4∠0.
% These are real-valued; the formulas reduce to scalar arithmetic:
%   Δ = s11·s22 - s12·s21 = 0.12 - 0.10 = 0.02
%   |s11|² = 0.09, |s22|² = 0.16, |Δ|² = 0.0004
%   K = (1 - 0.09 - 0.16 + 0.0004) / (2 · |0.05·2.0|)
%     = 0.7504 / 0.20
%     = 3.752
%   |Δ| = 0.02 < 1     → unconditionally stable (K > 1 + |Δ| < 1)

s11 = complex(0.3, 0.0);
s12 = complex(0.05, 0.0);
s21 = complex(2.0, 0.0);
s22 = complex(0.4, 0.0);

K = stabilityK(s11, s12, s21, s22);
disp(K(1));   % 3.752

% Edwards-Sinsky mu1 (source-side):
%   num = 1 - |s11|² = 0.91
%   conj(s11)·Δ = 0.3 · 0.02 = 0.006
%   |s22 - conj(s11)·Δ| = |0.4 - 0.006| = 0.394
%   |s12·s21| = 0.10
%   mu1 = 0.91 / (0.394 + 0.10) = 0.91 / 0.494 = 1.8421
%   mu1 > 1 → unconditionally stable.
mu1 = stabilityMu(s11, s12, s21, s22, 0);
disp(mu1(1));   % ~1.8421
