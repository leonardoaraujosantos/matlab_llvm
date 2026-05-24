% streaming_fir.m — DSP System Toolbox Tier-1 headline.
%
% The canonical frame-based streaming demo: design a lowpass FIR, build a
% dsp.FIRFilter System Object, and run a frame loop where the object
% persists its tapped-delay state across calls (y = firFilt(frame)).  The
% compiler lowers the call-syntax `firFilt(frame)` to the object's `step`
% method and the handle semantics carry the internal state forward — so a
% signal processed frame-by-frame is bit-identical to filtering it whole.
%
% Exercises the System-Object setup -> step (x N) -> reset lifecycle.

% A 16-tap lowpass FIR (windowed-sinc moving-average-ish smoother).
b = fir1(15, 0.25);

% A noisy tone: 50-sample signal = sine + a square-wave-ish disturbance.
n  = (0:49);
clean = sin(2 * pi * 0.03 * n);
noise = 0.5 * sin(2 * pi * 0.40 * n);   % high-freq component to reject
x = clean + noise;

% Build the System Object once; its Numerator is Nontunable.
firFilt = dsp.FIRFilter('Numerator', b);

% Stream the signal through in 5 frames of 10 samples each.  Each call to
% firFilt(frame) is a `step` that filters the frame and carries the
% delay-line state into the next frame.
y = zeros(1, 50);
for k = 1:5
    idx = (k - 1) * 10 + (1:10);
    y(idx) = firFilt(x(idx));
end

% The high-frequency disturbance should be strongly attenuated.  Measure
% the energy of the rejected band by comparing input vs output power.
pin  = sum(x .^ 2) / 50;
pout = sum(y .^ 2) / 50;
fprintf('input power  = %.4f\n', pin);
fprintf('output power = %.4f\n', pout);
fprintf('attenuation  = %.2f dB\n', 10 * log10(pin / pout));

% Confirm the streamed result equals the monolithic filter (state carried
% correctly across all five frames).
yref = filter(b, 1, x);
fprintf('frame-vs-whole maxdiff = %.6f\n', max(abs(y - yref)));
