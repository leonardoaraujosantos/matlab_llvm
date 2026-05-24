% DSP System Toolbox Tier-1 — dsp.IIRFilter (a != [1]) streaming.
%
% A System Object that persists its filter state across frame calls
% must reproduce the monolithic filter(b, a, x) for a true IIR (the
% denominator a is non-trivial), not just the FIR (a = [1]) case.
[b, a] = butter(4, 0.4);
x = [0.5 -0.3 0.8 -0.1 0.4 -0.6 0.2 -0.4 0.5 -0.3 0.7 0.1 -0.5 0.2 -0.3 0.6];

yref = filter(b, a, x);

iir = dsp.IIRFilter('Numerator', b, 'Denominator', a);
y1 = iir(x(1:8));                  % frame 1
y2 = iir(x(9:16));                 % frame 2 — state carries
ys = [y1 y2];

fprintf('butter order=%d\n', numel(a) - 1);
fprintf('maxdiff %.6f\n', max(abs(ys - yref)));
