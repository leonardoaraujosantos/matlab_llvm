% Comm Tier-2 §4.5: FSK round-trip.
%
% 4-FSK with freqsep = 200 Hz, 8 samples/symbol at 8 kHz sample rate.
% Modulate a 16-symbol random sequence, immediately demodulate
% coherently — should recover every symbol exactly with no AWGN.

% Note: orthogonal FSK requires freqsep ≥ 1/(2·T_sym) for coherent
% demod.  With nsamp=8, fs=8000 ⇒ T_sym = 1 ms ⇒ freqsep ≥ 500 Hz.
% We use 1000 Hz (well above the bound) for clean recovery.
rng(42);
M = 4;
freqsep = 1000.0;
nsamp = 8;
fs = 8000.0;

data = randi(M, 16, 1) - 1.0;     % 16 symbols in [0, 3]
y    = fskmod(data, M, freqsep, nsamp, fs);
% Noncoherent demod (mode=1) is robust to the accumulated phase across
% continuous-phase FSK symbols.  Mode 0 (coherent) would require
% per-symbol phase tracking.
data_hat = fskdemod(y, M, freqsep, nsamp, fs, 1);
nerr = sum(abs(data - data_hat));
disp(nerr);                        % 0 — perfect recovery on noiseless
disp(numel(y));                    % 16 * 8 = 128
