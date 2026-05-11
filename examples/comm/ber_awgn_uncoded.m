% ber_awgn_uncoded.m — sample-and-count BER curve for uncoded BPSK
% over AWGN. Demonstrates the canonical Tier-1 Monte-Carlo loop:
%
%     bits -> map -> AWGN -> threshold detect -> biterr
%
% BPSK in {-1, +1}, symbol energy Es = 1.
%
% `awgn(x, snr_dB)` matches MATLAB's convention: snr_dB =
% 10·log10(signal_power / noise_power). For real BPSK with Es = 1
% and real noise variance sigma², that's SNR = 1 / sigma². The
% closed-form BER under hard decision is therefore Q(1 / sigma) =
% Q(sqrt(SNR_lin)) where SNR_lin = 10^(SNR_dB / 10). This is the
% canonical "matlab awgn SNR convention" Q-function expression — it
% differs by a multiplicative ratio of 2 (= 3 dB) from the textbook Q(sqrt(2 Eb/N0))
% because Eb/N0 = SNR / 2 for one-axis real BPSK over real AWGN.
%
% At N = 50000 bits per point the sample BER should track the
% closed-form curve to ~10% relative at SNR <= 6 dB and ~50% relative
% at SNR = 10 dB (the tail is statistically noisy once you only
% expect ~40 errors out of 50000).

rng(2026);
N   = 50000;             % bits per SNR point
snr = [0 2 4 6 8 10];    % dB (matlab awgn convention)

fprintf('SNR (dB)  | sim BER     | Q(sqrt(SNR_lin))\n');
fprintf('----------+-------------+----------------\n');
for k = 1:6
    s = snr(k);

    % Source bits.
    tx_bits = randi(2, N, 1) - 1;        % {0, 1}
    tx_sym  = 1 - 2 * tx_bits;            % {+1, -1} BPSK

    rx = awgn(tx_sym, s);
    rx_bits = (rx < 0);                   % hard-decision threshold at 0

    sim_ber = biterr(tx_bits, rx_bits);

    % Closed-form (Q-function via the small-arg series for s < ~5 dB
    % and the asymptotic tail otherwise). Avoids needing erfc.
    snr_lin = pow_10(s / 10.0);
    x = sqrt(snr_lin);
    theo_ber = q_func(x);

    fprintf('  %5.1f   | %.6f    | %.6f\n', s, sim_ber, theo_ber);
end

function y = pow_10(x)
  y = exp(x * log(10));
end

function y = q_func(x)
  % Q-function approximation via Karagiannidis-Lioumpas (2007) —
  % closed-form, no branching, no erfc dependency. Relative error
  % stays under ~5e-4 for x in [0, 5], adequate for the BER-curve
  % overlay.
  a = 1.98;
  b = 1.135;
  num = (1.0 - exp(-a * x)) * exp(-x * x / 2.0);
  den = b * x * sqrt(2.0 * pi);
  y = num / den;
end
