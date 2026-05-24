% DSP System Toolbox Tier-4 — multirate System Objects.
%
% dsp.FIRDecimator filters then downsamples by M, persisting the
% anti-alias FIR state across frames.  dsp.FIRInterpolator zero-stuffs by
% L and filters.  dsp.SampleRateConverter combines them as an L/M rational
% rate change.  dsp.CICDecimator is the multiplier-free Hogenauer cascade.
b = fir1(20, 0.25);
M = 4;
x = sin(2 * pi * 0.05 * (0:127));

% Reference: a second SO consuming the whole signal in one frame.
ref = dsp.FIRDecimator(M, b);
yfull = ref(x);

dec = dsp.FIRDecimator(M, b);
y1 = dec(x(1:64));                       % frame 1
y2 = dec(x(65:128));                     % frame 2 — state carries forward
ys = [y1 y2];

fprintf('lengths in=%d framed=%d whole=%d\n', numel(x), numel(ys), numel(yfull));
fprintf('frame-vs-whole maxdiff %.6f\n', max(abs(ys - yfull)));

% Interpolator: in*L samples out.
itp = dsp.FIRInterpolator(M, b);
yi = itp(x(1:32));
fprintf('interp out=%d (expected %d)\n', numel(yi), 32 * M);

% Sample rate converter L/M.
src = dsp.SampleRateConverter(3, 2, fir1(30, 0.3));
ys2 = src(x(1:100));
fprintf('rateconv 3/2 in=%d out=%d\n', 100, numel(ys2));

% CIC: rate change only, no coefficients.
cic = dsp.CICDecimator(M);
cic.NumSections = 2;
yc = cic(x);
fprintf('cic decim out=%d sum=%.2f\n', numel(yc), sum(yc));
