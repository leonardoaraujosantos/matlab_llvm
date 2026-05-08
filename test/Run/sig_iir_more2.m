% Tier-1 (Signal Processing Toolbox roadmap §2.1 follow-on, second
% wave): standalone analog↔digital conversions + form conversions +
% cheb2ord. The first wave shipped band variants; this wave fills in
% the analog-side helpers that close the bilinear-design pipeline.

% cheb2ord(Wp, Ws, Rp, Rs) — order helper for Chebyshev II. Cheby II
% anchors at the **stopband** edge Ws (where the ripple peaks live),
% not the passband edge like Cheby I.
[n, Wn] = cheb2ord(0.2, 0.5, 1, 40);
disp(n);             % 4 (matches scipy)
disp(Wn);            % 0.5 — stopband edge

% Standalone bilinear: convert an analog 1st-order LP H(s) = 1/(s+1)
% to digital with sampling rate fs. The digital filter approximates
% an analog LP with -3 dB at omega = 1 rad/s.
b_an = [1];
a_an = [1 1];
[bd, ad] = bilinear(b_an, a_an, 1);
disp(bd);            % [0.33333, 0.33333] (scipy.signal.bilinear)
disp(ad);            % [1, -0.33333]

% Analog frequency response: |H(j*0)| = 1, |H(j*1)| = 1/sqrt(2).
H = freqs(b_an, a_an, [0 1 100]);
disp(abs(H));        % [1, 0.7071, 0.01]

% tf2zp / zp2tf round-trip on an order-2 polynomial.
%   b = z^2 - 5z + 6 = (z-2)(z-3)  → roots [2, 3]
%   a = z^2 - 7z + 12 = (z-3)(z-4) → roots [3, 4]
%   k = 1
% Durand-Kerner returns roots in solver-order; assert via the
% Vieta's-formulas symmetric functions (sum + product) so the test
% is invariant across the C / Python / TS lanes.
b_in = [1 -5 6];
a_in = [1 -7 12];
[zs, ps, kk] = tf2zp(b_in, a_in);
disp(sum(real(zs)));      % 2 + 3 = 5
disp(prod(real(zs)));     % 2 * 3 = 6
disp(sum(real(ps)));      % 3 + 4 = 7
disp(prod(real(ps)));     % 3 * 4 = 12
disp(kk);                 % 1

% Reverse: zp2tf rebuilds polynomials from roots + gain. Polynomial
% reconstruction is order-invariant so the (b_r, a_r) display matches
% across lanes.
[b_r, a_r] = zp2tf(zs, ps, kk);
disp(b_r);           % [1, -5, 6]
disp(a_r);           % [1, -7, 12]

% besself(n, Wo) — analog Bessel-Thomson lowpass (norm='phase' /
% MATLAB convention: poles of B_n(s) scaled by Wo).
% B_4(s) = s^4 + 10s^3 + 45s^2 + 105s + 105.
[bb, ab] = besself(4, 1);
disp(bb);            % [105]
disp(ab);            % [1 10 45 105 105]
% At Wo = 2 the analog cutoff is twice as high; coefficients scale
% by Wo^i in MATLAB-order position i.
[bb2, ab2] = besself(4, 2);
disp(bb2);           % [1680]
disp(ab2);           % [1, 20, 180, 840, 1680]
