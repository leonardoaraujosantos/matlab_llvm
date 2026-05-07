% Tier-1 (Signal Processing Toolbox roadmap §2.1): IIR lowpass design
% + freqz frequency response. Lowpass scope; high/band/stop variants
% and ellip/besself/order-helpers are deferred to follow-on slices.

% Order-4 Butterworth lowpass at Wn = 0.4 (40% of Nyquist).
[b, a] = butter(4, 0.4);
disp(b);
disp(a);
% DC gain identity: H(z=1) = sum(b)/sum(a) = 1.
disp(sum(b) / sum(a));
% Numerator symmetric, denominator monic.
disp(b(1) - b(5));     % 0
disp(a(1));            % 1

% Order-3 Chebyshev I, Rp = 0.5 dB, Wn = 0.3.
[bc, ac] = cheby1(3, 0.5, 0.3);
disp(bc);
disp(ac);
disp(sum(bc) / sum(ac));   % 1 — odd N, unit DC gain

% freqz at 4 frequencies on the Butterworth filter. abs(H) gives the
% magnitude response — 1 at DC, attenuated past the cutoff.
H = freqz(b, a, 4);
disp(abs(H));
