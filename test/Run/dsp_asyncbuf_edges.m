% DSP System Toolbox Tier-5 — dsp.AsyncBuffer capacity edge cases.
%
% Wrap-around (write past Capacity) and read-more-than-available
% (returns fewer than requested without erroring), plus reset.  The
% steady-state FIFO order is covered in dsp_spectrum; this test
% exercises the boundary conditions.
ab = dsp.AsyncBuffer('Capacity', 4);

% Push 3 samples (count = 3, no wrap yet).
ab.write([10 20 30]);
y1 = ab.read(2);
fprintf('r2: %.0f %.0f\n', y1(1), y1(2));

% Write 4 more — the 1 leftover + 4 wraps, count caps at Capacity = 4,
% the FIFO keeps the *most-recent* C samples.
ab.write([40 50 60 70]);
y2 = ab.read(4);
fprintf('r4: %.0f %.0f %.0f %.0f\n', y2(1), y2(2), y2(3), y2(4));

% Reset clears Count.  Push 2 samples and read 4 — the runtime returns
% just what's available (2), no overrun.
ab.reset();
ab.write([99 88]);
y3 = ab.read(4);
fprintf('after reset: %.0f %.0f\n', y3(1), y3(2));
