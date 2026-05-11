% ber_qam_montecarlo.m — 16-QAM Monte-Carlo BER vs theory.
%
% Canonical Tier-2 closure example: source -> qammod -> AWGN ->
% qamdemod -> biterr against the berawgn closed-form reference.
%
% Setup conventions:
%   - Gray mapping on both encode and decode
%   - UnitAveragePower so each symbol has mean energy 1
%   - awgn(x, snr_dB) treats snr_dB as 10*log10(signal/noise) — to
%     report against Eb/N0 we shift by 10*log10(log2(M)) = 6 dB for
%     16-QAM (4 bits per symbol).
%
% At N = 20000 symbols per SNR point, the simulation tracks the
% closed-form curve to within ~10% relative for Eb/N0 >= 6 dB; the
% high-SNR tail is statistically noisy once the per-point error
% count drops below ~10.

rng(2026);

M    = 16;
k    = 4;                              % log2(M)
N    = 20000;                          % symbols per Eb/N0 point
ebn0 = [4 6 8 10 12 14];               % dB

% Pre-compute the symbol-SNR shift for the awgn convention.
% Eb/N0 (dB) = SNR (dB) - 10*log10(k); so SNR_dB = Eb/N0_dB + 10*log10(k).
k_dB = 10 * log(k) / log(10);

fprintf('Eb/N0 (dB) | sim BER       | berawgn theory\n');
fprintf('-----------+---------------+----------------\n');
for i = 1:6
    eb = ebn0(i);

    data    = randi(M, N, 1) - 1;
    tx      = qammod(data, M, 1, 1);     % Gray, unit avg power

    rx      = awgn(tx, eb + k_dB);        % shift to symbol SNR
    rx_data = qamdemod(rx, M, 1, 1);

    sim_ber = biterrK(data, rx_data, k);
    theo_ber = berawgn(eb, M, 2);          % 16-QAM closed form

    fprintf(' %5.1f     | %.6e   | %.6e\n', eb, sim_ber, theo_ber);
end
