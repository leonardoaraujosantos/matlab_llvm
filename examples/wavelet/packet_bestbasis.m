% packet_bestbasis.m — Wavelet Toolbox Tier-5.
% ----------------------------------------------------------------------
% Wavelet packet harmonic-interference removal (the UG demo): decompose a
% signal carrying a narrow-band interference into a full packet tree, find
% the node where the interference concentrates from the node-energy map,
% zero it, and reconstruct.
n = 512;
t = (0:n-1);
sig = sin(2*pi*t/64);                 % signal of interest (low frequency)
interf = 3.0*sin(2*pi*t/6);           % strong narrow-band interference
x = sig + interf;

T = wpdec(x, 4, 'db4');
e = wenergy(T);                       % per-node energy (%)
fprintf('packet nodes = %.0f\n', size(T,1));
[~, k] = max(e);
fprintf('dominant interference node = %.0f\n', k - 1);

% zero the interference node and rebuild
T(k, :) = 0;
xr = wprec(T, 'db4');
fprintf('interference norm before = %.2f\n', norm(x - sig));
fprintf('interference norm after  = %.2f\n', norm(xr - sig));
