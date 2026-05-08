% Tier-1 (Signal Processing Toolbox roadmap §2.1 follow-on): band
% variants of butter / cheby1 / cheby2 — high / bandpass / bandstop.
% Dispatch is by call shape:
%   butter(n, Wn)              lowpass   (Wn scalar)
%   butter(n, [W1 W2])         bandpass  (Wn 2-element row)
%   butter(n, Wn, 'high')      highpass
%   butter(n, [W1 W2], 'stop') bandstop
% cheby1 / cheby2 take an extra Rp / Rs parameter at position 1.

% Butterworth highpass — symmetric / anti-symmetric pattern, monic a.
[bh, ah] = butter(4, 0.4, 'high');
disp(bh);        % MATLAB: [0.16718, -0.66872, 1.00308, -0.66872, 0.16718]
disp(ah);

% Butterworth bandpass — every other b coefficient is zero (BP zeros at
% z = +1 and z = -1 cancel through the polynomial expansion).
[bp, ap] = butter(4, [0.2 0.6]);
disp(bp);
disp(ap);
% DC gain identity: H(z=1) = 0 for BP (zeros at z=1), so the test is
% the centre-frequency gain. omega_c = 2*atan(sqrt(W1a*W2a)/2) at the
% geometric mean of the prewarped edges.

% Butterworth bandstop — full polynomial b (no zero coefficients), unity
% DC gain (BS keeps DC).
[bs, as] = butter(4, [0.2 0.6], 'stop');
disp(bs);
disp(as);
disp(sum(bs) / sum(as));   % unit DC gain — BS is a low/high pass-through

% Chebyshev I highpass.
[bch, ach] = cheby1(3, 0.5, 0.3, 'high');
disp(bch);
disp(ach);

% Chebyshev I bandpass.
[bcp, acp] = cheby1(3, 0.5, [0.3 0.6]);
disp(bcp);
disp(acp);

% Chebyshev II highpass — finite j-axis zeros bilinear-transform to
% finite z-plane points (not all at z = +1 like Butterworth/Cheby1).
[bc2h, ac2h] = cheby2(4, 40, 0.4, 'high');
disp(bc2h);
disp(ac2h);

% Chebyshev II bandstop.
[bc2s, ac2s] = cheby2(4, 40, [0.2 0.6], 'stop');
disp(bc2s);
disp(ac2s);
