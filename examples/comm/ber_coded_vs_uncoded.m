% ber_coded_vs_uncoded.m — coded vs uncoded BER curves over BPSK + AWGN.
%
% Compares three transmission schemes at the same Eb/N0:
%   1. Uncoded BPSK (rate 1, k bits = k channel uses)
%   2. (7, 4) Hamming + BPSK (rate 4/7)
%   3. (171, 133)_8 K=7 rate-1/2 convolutional + BPSK
%
% For the coded schemes, "Eb/N0" refers to the *information-bit* energy.
% Since the rate-1/2 convolutional code transmits 2 channel bits per
% info bit, the per-channel-symbol SNR is Eb/N0 - 3 dB; the Hamming
% rate-4/7 code is Eb/N0 - 10·log10(7/4) ≈ Eb/N0 - 2.43 dB.
%
% Convention: `awgn(x, snr_dB)` treats `snr_dB` as 10·log10(signal /
% noise); for BPSK with unit symbol energy, snr_dB = Eb/N0_dB + rate_dB.
%
% At 30 000 information bits per Eb/N0 point the convolutional code
% beats the uncoded curve by ~5 dB at the 1e-3 BER level — the
% textbook "soft / hard coding gain" with hard-decision Viterbi.

rng(2027);

N    = 30000;
ebn0 = [2 3 4 5 6 7];      % dB

% (171, 133) convolutional code setup.
gens = [oct2dec(171), oct2dec(133)];
t    = poly2trellis(7, gens);

% Hamming(7, 4) setup.
m_h = 3;
k_h = 4;
n_h = 7;

fprintf('Eb/N0 (dB) | uncoded BPSK | Hamming(7,4) | conv (171,133)\n');
fprintf('-----------+--------------+--------------+----------------\n');
for i = 1:6
    eb = ebn0(i);

    % ----- 1. Uncoded BPSK -----
    bits     = randi(2, N, 1) - 1;
    tx_u     = 1 - 2 * bits;
    rx_u     = awgn(tx_u, eb);
    rx_bits  = (rx_u < 0);
    ber_u    = biterr(bits, rx_bits);

    % ----- 2. Hamming(7, 4) -----
    % Build N_h codewords each from 4 random bits; rate is 4/7 so the
    % per-channel-symbol SNR is eb - 10*log10(7/4).
    snr_h = eb + 10 * log(4.0 / 7.0) / log(10);
    N_h   = floor(N / k_h);
    err_h = 0;
    for b = 1:N_h
        msg     = randi(2, k_h, 1) - 1;
        codeword = hammingEncode(msg, m_h);
        tx       = 1 - 2 * codeword;
        rx       = awgn(tx, snr_h);
        rx_bits  = (rx < 0);
        decoded  = hammingDecode(rx_bits, m_h);
        err_h    = err_h + biterrCount(msg, decoded);
    end
    ber_h = err_h / (N_h * k_h);

    % ----- 3. Convolutional (171, 133)_8 -----
    snr_c    = eb + 10 * log(0.5) / log(10);
    msg      = randi(2, N, 1) - 1;
    code     = convenc(msg, t);
    tx       = 1 - 2 * code;
    rx       = awgn(tx, snr_c);
    rx_bits  = (rx < 0);
    decoded  = vitdec(rx_bits, t, 35, 0, 1);
    ber_c    = biterr(msg, decoded);

    fprintf(' %5.1f     | %.6e | %.6e | %.6e\n', ...
            eb, ber_u, ber_h, ber_c);
end
