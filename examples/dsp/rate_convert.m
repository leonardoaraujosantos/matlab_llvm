% rate_convert.m — DSP System Toolbox Tier-4 headline.
%
% Multistage rational sample-rate conversion.  Build a 3/2 sample-rate
% converter (interpolate-by-3, then decimate-by-2), feed it a multi-tone
% signal, and confirm the output rate is fs_in * 3 / 2 while the polyphase
% filter state persists across streaming frame calls.

L = 3;     % interpolation factor
M = 2;     % decimation factor

% Anti-aliasing / anti-imaging lowpass at min(1/L, 1/M) with a small
% margin.  For L=3, M=2 the binding constraint is 1/L = 0.333.
fc = 0.31;
b  = fir1(60, fc);

% A clean tone at 0.1 of the input Nyquist, plus a high-band tone that
% the anti-aliasing FIR will suppress.
n  = 0:255;
x  = sin(2 * pi * 0.10 * n) + 0.5 * sin(2 * pi * 0.40 * n);

src = dsp.SampleRateConverter(L, M, b);

% Stream in 4 frames of 64 samples each — the polyphase commutator + FIR
% state carry across frames.
ys = zeros(1, 256 * L / M);
for k = 1:4
    idx_in  = (k - 1) * 64 + (1:64);
    idx_out = (k - 1) * (64 * L / M) + (1:(64 * L / M));
    ys(idx_out) = src(x(idx_in));
end

fprintf('input rate (samples)  = %d\n', numel(x));
fprintf('output rate (samples) = %d   (expected %d)\n', numel(ys), 256 * L / M);

% Compare framed-streaming result to a fresh whole-signal SO — they must
% agree exactly (the proof that polyphase state is consistent).
src_ref = dsp.SampleRateConverter(L, M, b);
yref = src_ref(x);
fprintf('frame-vs-whole maxdiff = %.6f\n', max(abs(ys - yref)));

% CIC decimator demo: multiplier-free 4× decimation, 2 sections.
cic = dsp.CICDecimator(4);
cic.NumSections = 2;
yc = cic(x);
fprintf('CIC out length = %d (rate = in / 4)\n', numel(yc));
