% wcoherence_pair.m — Wavelet Toolbox Tier-3.
% ----------------------------------------------------------------------
% Wavelet coherence between two signals that share a common oscillation:
% the smoothed cross-scalogram is high where the signals are phase-locked.
fs = 1000;
t  = (0:1023)/fs;
common = sin(2*pi*40*t);
x = common + 0.5*sin(2*pi*120*t);
y = common + 0.5*cos(2*pi*15*t);

R = wcoherence(x, y);
fprintf('coherence map size = %.0f x %.0f\n', size(R,1), size(R,2));
fprintf('peak coherence = %.2f\n', max(max(R)));
fprintf('mean coherence = %.2f\n', mean(mean(R)));
