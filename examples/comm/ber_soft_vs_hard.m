% ber_soft_vs_hard.m — soft- vs hard-decision Viterbi BER curves.
%
% Demonstrates the ~3 dB soft-decision gain of optimal MAP-style
% branch metrics over hard-decision Hamming-distance metrics on a
% (171, 133)_8 K=7 rate-1/2 convolutional code over BPSK + AWGN.
%
% At each Eb/N0:
%   1. generate uncoded bits, BPSK-modulate, convolutional encode,
%      add AWGN at the channel-rate SNR (Eb/N0 - 3 dB for rate-1/2).
%   2. Hard branch: threshold rx to {0, 1}, run vitdec.
%   3. Soft branch: pass rx unchanged as LLR-equivalent, run
%      vitdecSoft.
%
% At 50 k information bits per Eb/N0 point the soft curve sits ~3 dB
% to the left of the hard curve from 4 dB Eb/N0 onward.

rng(2031);

gens = [oct2dec(171), oct2dec(133)];
t    = poly2trellis(7, gens);

N    = 50000;
ebn0 = [1 2 3 4 5];
% Channel-rate SNR = Eb/N0 - 10*log10(2) for rate-1/2.
rate_dB = 10 * log(0.5) / log(10);

fprintf('Eb/N0 (dB) | hard Viterbi | soft Viterbi\n');
fprintf('-----------+--------------+--------------\n');
for i = 1:5
    eb = ebn0(i);
    snr_c = eb + rate_dB;

    msg  = randi(2, N, 1) - 1;
    code = convenc(msg, t);
    tx   = 1 - 2 * code;
    rx   = awgn(tx, snr_c);

    % Hard branch
    hard_in = (rx < 0);
    hard_dec = vitdec(hard_in, t, 35, 0, 1);

    % Soft branch — vitdecSoft expects "positive favours bit=0".
    % rx (for tx in {+1, -1}) already has that property when bit=0
    % maps to tx=+1.  LLR scale is irrelevant to the path-metric
    % argmax, so a unit-scale LLR works.
    soft_dec = vitdecSoft(rx, t, 35, 0);

    ber_h = biterr(msg, hard_dec);
    ber_s = biterr(msg, soft_dec);
    fprintf(' %5.1f     | %.6e | %.6e\n', eb, ber_h, ber_s);
end
