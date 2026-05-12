% Comm Tier-2 §4.5 example — 4-FSK round-trip.
%
% Orthogonal-tone M-FSK: continuous-phase modulation with freqsep =
% 1/T_sym (well above the coherent-orthogonality floor of 1/(2 T_sym)).
% Noncoherent demod tolerates the phase drift between symbols.

rng(7);
M = 4;
freqsep = 1000.0;
nsamp = 8;
fs = 8000.0;

data = randi(M, 32, 1) - 1.0;
y = fskmod(data, M, freqsep, nsamp, fs);
data_hat = fskdemod(y, M, freqsep, nsamp, fs, 1);
nerr = sum(abs(data - data_hat));
disp(nerr);          % 0
