% Tier-1 (Signal Processing Toolbox roadmap §2.1 follow-on):
% cheby2 lowpass + buttord / cheb1ord order-selection helpers.
%
% Lowpass scope only — band variants, ellip/besself, analog prototypes,
% standalone bilinear/freqs, and form conversions are deferred.

% Order-4 Chebyshev II with 40 dB stopband attenuation at Wn = 0.4.
% cheby2 has finite j-axis zeros (n or n-1 of them) that bilinear-
% transform to specific points on the unit circle — distinct from the
% n zeros at z = -1 that Butterworth and Chebyshev I produce.
[bc2, ac2] = cheby2(4, 40, 0.4);
disp(bc2);
disp(ac2);
disp(sum(bc2) / sum(ac2));     % unit DC gain by normalization

% Order-3 Chebyshev II — odd order, one zero at infinity (i.e., one
% of the n design "zeros" lands at z = -1 after bilinear, the other
% n - 1 at finite j-axis-image locations).
[bc3, ac3] = cheby2(3, 30, 0.3);
disp(bc3);
disp(ac3);

% buttord(Wp, Ws, Rp, Rs): smallest order Butterworth lowpass to meet
%   passband ripple <= Rp dB on [0, Wp] and stopband attenuation
%   >= Rs dB on [Ws, 1].
[n, Wn] = buttord(0.2, 0.5, 1, 40);
disp(n);                 % 5 — required order
disp(Wn);                % natural cutoff (3 dB) point

% cheb1ord same specs — Chebyshev I always meets order with fewer
% taps than Butterworth on the same spec.
[n2, Wn2] = cheb1ord(0.2, 0.5, 1, 40);
disp(n2);                % 4
disp(Wn2);               % equals Wp (Cheby I meets passband at Wp)
