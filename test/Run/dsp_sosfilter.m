% DSP System Toolbox Tier-1 — dsp.SOSFilter cascaded biquad streaming.
%
% A cascaded second-order-section filter object must reproduce sosfilt
% across frame boundaries (per-section state carried forward).
[b, a] = butter(4, 0.3);
sos = tf2sos(b, a);
x = [1 2 3 4 5 6 7 8 9 10];

yref = sosfilt(sos, x);

g = dsp.SOSFilter('SOSMatrix', sos);
y1 = g(x(1:5));
y2 = g(x(6:10));
ys = [y1 y2];

fprintf('sections %d\n', size(sos, 1));
fprintf('maxdiff %.6f\n', max(abs(ys - yref)));
