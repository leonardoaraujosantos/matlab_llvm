% DSP System Toolbox Tier-1 — dsp.Delay integer delay line.
%
% A delay of D samples shifts the input by D, with zeros prepended for
% the first D samples.  State persists across frames so the delayed
% samples appear in subsequent step calls.
D = 3;
del = dsp.Delay(D);
x = [1 2 3 4 5 6 7 8 9 10];

% First frame: outputs 0 0 0 1 2 (the first 3 are the initial delay).
y1 = del(x(1:5));
% Second frame: continues 3 4 5 6 7 — the carried state's tail emerges.
y2 = del(x(6:10));

fprintf('frame1: %.0f %.0f %.0f %.0f %.0f\n', y1(1), y1(2), y1(3), y1(4), y1(5));
fprintf('frame2: %.0f %.0f %.0f %.0f %.0f\n', y2(1), y2(2), y2(3), y2(4), y2(5));

% Reset zeroes the delay line; next frame outputs 3 zeros + 11 12.
del.reset();
y3 = del([11 12 13 14 15]);
fprintf('after reset: %.0f %.0f %.0f %.0f %.0f\n', y3(1), y3(2), y3(3), y3(4), y3(5));
