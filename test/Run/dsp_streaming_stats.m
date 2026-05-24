% DSP System Toolbox Tier-5 — streaming statistics + sources.
%
% Generate a sine with dsp.SineWave, then route it through three
% sliding-window stats objects and a peak detector.  Each object persists
% its window state across the streaming frame call; the metrics over one
% 200-sample frame match the closed-form values for a 50 Hz sine at 1 kHz
% (10 cycles -> 10 peaks; RMS = 1/sqrt(2)).
src = dsp.SineWave('Frequency', 50);
src.SampleRate = 1000;
src.SamplesPerFrame = 200;
x = src();

ma  = dsp.MovingAverage('WindowLength', 5);
mr  = dsp.MovingRMS('WindowLength', 32);
mx  = dsp.MovingMaximum('WindowLength', 16);
pf  = dsp.PeakFinder();
dcb = dsp.DCBlocker();

ya = ma(x);
yr = mr(x);
ym = mx(x);
yp = pf(x);
yd = dcb(x + 0.5);                    % add DC bias and watch it die

fprintf('sine n=%d max=%.2f\n', numel(x), max(x));
fprintf('movavg(W=5)  peak       = %.3f\n', max(ya));
fprintf('movrms(W=32) tail       = %.3f\n', yr(200));
fprintf('movmax(W=16) peak       = %.3f\n', max(ym));
fprintf('peakfinder count        = %d\n', sum(yp > 0));
fprintf('dcblocker resid mean    = %.4f\n', mean(yd(150:200)));
