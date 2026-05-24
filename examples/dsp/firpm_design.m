% firpm_design.m — DSP System Toolbox Tier-2 headline.
%
% Design a narrow-transition equiripple lowpass FIR with the
% Parks-McClellan algorithm (firpm), compare it against the least-squares
% design (firls), and build a dsp.FIRFilter System Object from the
% equiripple taps to filter a multi-tone signal — the Tier-1 streaming
% object consuming a Tier-2 design.

% Equiripple lowpass: pass [0,0.2], stop [0.3,1] (normalised, 1 = Nyquist).
N = 30;
edges = [0 0.2 0.3 1];
amp   = [1 1 0 0];
b_pm = firpm(N, edges, amp);
b_ls = firls(N, edges, amp);

% Passband (DC) and stopband (Nyquist) gains via the closed-form sums.
k   = 0:N;
alt = cos(pi * k);
fprintf('firpm  passband gain = %.4f   stopband(Nyq) = %.4f\n', ...
        sum(b_pm), sum(b_pm .* alt));
fprintf('firls  passband gain = %.4f   stopband(Nyq) = %.4f\n', ...
        sum(b_ls), sum(b_ls .* alt));

% Two tones: one in the passband (0.1) and one in the stopband (0.45).
n = 0:255;
x = sin(2 * pi * 0.05 * n) + sin(2 * pi * 0.225 * n);

% Stream through a dsp.FIRFilter built from the equiripple taps.
firFilt = dsp.FIRFilter('Numerator', b_pm);
y = zeros(1, 256);
for f = 1:8
    idx = (f - 1) * 32 + (1:32);
    y(idx) = firFilt(x(idx));
end

% The stopband tone should be heavily attenuated -> output power drops.
fprintf('input power  = %.4f\n', sum(x .^ 2) / 256);
fprintf('output power = %.4f\n', sum(y .^ 2) / 256);

% A second iirnotch demo: notch out a 0.4-Nyquist interferer.
[bn, an] = iirnotch(0.4, 0.08);
interferer = sin(0.4 * pi * n);
yn = filter(bn, an, interferer);
fprintf('notch suppresses interferer to %.4f\n', max(abs(yn(200:256))));
