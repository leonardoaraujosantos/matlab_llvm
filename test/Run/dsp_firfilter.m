% DSP System Toolbox Tier-1 — dsp.FIRFilter frame-based streaming.
%
% A System Object that persists its tapped-delay state across frame calls
% must reproduce the monolithic filter(b,1,x).  Exercises the
% `obj = dsp.FIRFilter('Numerator', b)` constructor (package-syntax fold),
% the `y = obj(frame)` call-syntax -> step dispatch, and matrix-typed
% DiscreteState writeback inside the runtime step.
b = [0.2 0.2 0.2 0.2 0.2];           % 5-tap moving average
x = [1 2 3 4 5 6 7 8];               % row signal

yref = filter(b, 1, x);

firFilt = dsp.FIRFilter('Numerator', b);
y1 = firFilt(x(1:4));                % first frame
y2 = firFilt(x(5:8));                % second frame — state carries over
ys = [y1 y2];

fprintf('maxdiff %.4f\n', max(abs(ys - yref)));
disp(ys);
