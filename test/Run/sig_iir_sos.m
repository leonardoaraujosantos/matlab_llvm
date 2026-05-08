% Tier-1 (Signal Processing Toolbox roadmap §2.1 follow-on, third
% wave): SOS form conversions. tf2sos / sos2tf round-trip an IIR
% transfer function through cascade-of-biquads form for numerical
% stability of high-order filters.

% butter(4, 0.4) → tf2sos → 2-section SOS matrix (2x6).
[b, a] = butter(4, 0.4);
sos = tf2sos(b, a);
disp(size(sos, 1));    % 2 — two biquad sections
disp(size(sos, 2));    % 6 — [b0 b1 b2 a0 a1 a2]

% Round-trip via sos2tf: should recover (b, a) up to small DK noise.
% Verify lengths and that sums stay close to the originals.
[b_r, a_r] = sos2tf(sos);
disp(size(b_r, 2));    % 5 — order-4 numerator
disp(size(a_r, 2));    % 5 — order-4 denominator
