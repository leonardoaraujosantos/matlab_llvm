% Tier-2 (Signal Processing Toolbox roadmap §3.1 + §3.2): cross-spectral
% helpers (cpsd / mscohere / tfestimate) and parametric PSD methods
% (pyulear / pburg). All single-output forms, default fs = 1.

x = [1 2 3 4 5 6 7 8];
w = hamming(4);

% cpsd(x, x) is identical to pwelch(x) (cross-spectrum with self).
Pxx = pwelch(x, w, 0);
Pxy = cpsd(x, x, w, 0);
disp(max(abs(Pxx - real(Pxy))) < 1e-10);    % true

% mscohere(x, x) — perfect coherence, all bins equal 1.
C = mscohere(x, x, w, 0);
disp(C);

% pyulear / pburg on an alternating-sign decaying signal.
y = [1 -1 0.5 -0.5 0.25 -0.25 0.125 -0.125];
P_y = pyulear(y, 2, 4);
disp(size(P_y, 1));    % 4 grid points
disp(P_y);

P_b = pburg(y, 2, 4);
disp(P_b);
