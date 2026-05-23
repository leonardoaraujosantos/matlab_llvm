% DSP System Toolbox Tier-5 — sources + sliding stats not in the
% main streaming_stats test: dsp.NCO, dsp.Chirp, dsp.MovingMinimum,
% dsp.MovingStandardDeviation.
%
% These classdefs all forward through the shared phase-accumulator or
% sliding-window-step runtime, so the test is light: confirm output
% length + numeric sanity.

% NCO — alias of SineWave at the simulation level.
nco = dsp.NCO();
nco.Frequency = 50;
nco.SampleRate = 1000;
nco.SamplesPerFrame = 100;
y_nco = nco();
fprintf('nco n=%d peak=%.3f trough=%.3f\n', ...
        numel(y_nco), max(y_nco), min(y_nco));

% Chirp — instantaneous frequency sweeps linearly.  Start at 10 Hz,
% sweep at 100 Hz/s for 0.1 s -> final freq 20 Hz; output is still a
% finite sinusoid with peak amplitude 1.
ch = dsp.Chirp(10, 100);
ch.SampleRate = 1000;
ch.SamplesPerFrame = 100;
y_ch = ch();
fprintf('chirp n=%d peak=%.3f\n', numel(y_ch), max(y_ch));

% MovingMinimum + MovingStandardDeviation over the same NCO sine.
mmin = dsp.MovingMinimum('WindowLength', 16);
mstd = dsp.MovingStandardDeviation('WindowLength', 16);
y_min = mmin(y_nco);
y_std = mstd(y_nco);
fprintf('movmin tail=%.3f (sine trough -1)\n', y_min(100));
fprintf('movstd tail=%.3f (sine std ~ 0.7)\n', y_std(100));
