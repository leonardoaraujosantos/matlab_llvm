% streaming_stats.m — DSP System Toolbox Tier-5 headline.
%
% Drive a sine through a sliding moving-average smoother and a sliding RMS
% estimator, and detect peaks frame by frame.  Every object persists its
% window state across the streaming-frame call, so the result for one
% long signal equals the concatenation of results for the signal split
% into frames.

src = dsp.SineWave('Frequency', 5);
src.SampleRate = 1000;
src.SamplesPerFrame = 100;

% Make a multi-frame signal by stepping the source repeatedly.
x = zeros(1, 1000);
for k = 1:10
    x((k - 1) * 100 + (1:100)) = src();
end

ma   = dsp.MovingAverage('WindowLength', 16);
mrms = dsp.MovingRMS('WindowLength', 32);
pf   = dsp.PeakFinder();

% Stream through three independent objects with state-carry across the
% 10 frames.
ys = zeros(1, 1000);
yr = zeros(1, 1000);
yp = zeros(1, 1000);
for k = 1:10
    idx = (k - 1) * 100 + (1:100);
    ys(idx) = ma(x(idx));
    yr(idx) = mrms(x(idx));
    yp(idx) = pf(x(idx));
end

fprintf('signal   peak = %.3f\n', max(x));
fprintf('smoothed peak = %.3f   (lowpass attenuation)\n', max(ys));
fprintf('RMS settled   = %.3f   (sine RMS = 1/sqrt(2))\n', yr(800));
fprintf('peaks found   = %d     (5 Hz * 1 s = 5 cycles)\n', sum(yp > 0));
