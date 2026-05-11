% ofdm_awgn.m — single-symbol OFDM loopback over AWGN.
%
% Builds one OFDM symbol carrying 64 QPSK-modulated data subcarriers,
% adds AWGN, demodulates, and reports the symbol-error rate.  At
% SNR = 15 dB the loopback should recover all 64 symbols.
%
% This is the function-form OFDM workflow per
% docs/comm_toolbox_roadmap.md §7.1.  Pilots / guards / multi-symbol
% bursts are caller-side compositions on top of this primitive.

rng(2033);

Nfft = 64;
Lcp  = 16;
M    = 4;

% --- Tx: QPSK data on every subcarrier ---
data = randi(M, Nfft, 1) - 1;
sym  = pskmod(data, M, pi/4, 1);     % unit-power QPSK
tx   = ofdmmod(sym, Nfft, Lcp);

% --- Channel: pure AWGN ---
snr_dB = 15;
rx = awgn(tx, snr_dB);

% --- Rx: OFDM demod + per-subcarrier ML detect ---
rx_data = ofdmdemod(rx, Nfft, Lcp);
alpha   = pskmod((0:M-1)', M, pi/4, 1);
data_hat = mlDetect(rx_data, alpha);

% Compare via integer-equality count (avoid size-on-complex on
% rx_data path).
fprintf('=== OFDM over AWGN, Nfft = %.0f, CP = %.0f, SNR = %.0f dB ===\n', ...
        Nfft, Lcp, snr_dB);
fprintf('symerr (per subcarrier): %.4f\n', symerr(data, data_hat));
% Also show the per-stage shapes.
fprintf('tx samples per OFDM symbol: %.0f\n', size(abs(tx), 1));
fprintf('rx_data subcarrier count  : %.0f\n', size(abs(rx_data), 1));
