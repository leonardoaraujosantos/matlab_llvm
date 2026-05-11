% modern_codes_ber.m — Tier-7 closure: BER comparison of Polar / LDPC /
% Turbo vs uncoded BPSK at one fixed SNR point.
%
% Single SNR comparison (multiple-SNR sweep takes too many compile-cycles
% to keep in the smoke).  Each code is run over the same number of
% information bits so the comparison reflects relative coding gain
% rather than per-trial luck.

rng(2035);
SNR_DB = 5;     % chosen so uncoded BPSK shows visible errors

% --- Common message ---
NUM_INFO = 64;
msg = randi(2, NUM_INFO, 1) - 1;

% ============================================================
% Uncoded BPSK
% ============================================================
tx = 1 - 2 * msg;
rx = awgn(tx, SNR_DB);
dec_unc = (rx < 0);
err_unc = biterrCount(msg, dec_unc);

% ============================================================
% Polar (N=128, K=64): rate-1/2 stand-in
% ============================================================
N_pol = 128;
frozen_pol = ones(N_pol, 1);
% Information positions: 65..128 (upper half).
for i = 65:N_pol
    frozen_pol(i) = 0;
end
u_pol = zeros(N_pol, 1);
for i = 1:NUM_INFO
    u_pol(64 + i) = msg(i);
end
cw_pol = polarEncode(u_pol, N_pol);
tx_pol = 1 - 2 * cw_pol;
rx_pol = awgn(tx_pol, SNR_DB);
llr_pol = 2 * rx_pol;
u_hat = polarSCdecode(llr_pol, frozen_pol, N_pol);
% Extract info bits.
info_hat = zeros(NUM_INFO, 1);
for i = 1:NUM_INFO
    info_hat(i) = u_hat(64 + i);
end
err_pol = biterrCount(msg, info_hat);

% ============================================================
% Turbo PCCC (k=64, rate 1/3, (7, 5)_8 RSC, shift-by-11 perm)
% ============================================================
gens = [oct2dec(7), oct2dec(5)];
t = poly2trellis(3, gens);
perm = zeros(NUM_INFO, 1);
for i = 1:NUM_INFO
    perm(i) = mod(i - 1 + 11, NUM_INFO) + 1;
end
code_tur = turboEncode(msg, t, perm);
tx_tur = 1 - 2 * code_tur;
rx_tur = awgn(tx_tur, SNR_DB);
llr_tur = 2 * rx_tur;
llr_sys = llr_tur(1:NUM_INFO);
llr_p1  = llr_tur(NUM_INFO+1:2*NUM_INFO);
llr_p2  = llr_tur(2*NUM_INFO+1:3*NUM_INFO);
dec_tur = turboDecode(llr_sys, llr_p1, llr_p2, t, perm, 6);
err_tur = biterrCount(msg, dec_tur);

% ============================================================
% LDPC (6, 3) hand-rolled — applied as a block code per 3-bit chunk
% so the comparison stays on the same NUM_INFO=64 message.
% ============================================================
P_ldpc = [1 1 0; 0 1 1; 1 0 1];
H_ldpc = [1 0 1 1 0 0; 1 1 0 0 1 0; 0 1 1 0 0 1];
% Build 21 blocks of 3 information bits (last block uses 1 padding bit).
nb = 21;
err_ldpc = 0;
for b = 1:nb
    s = (b-1) * 3 + 1;
    e = s + 2;
    if e > NUM_INFO; e = NUM_INFO; end
    chunk = zeros(3, 1);
    for j = s:e
        chunk(j - s + 1) = msg(j);
    end
    cw = ldpcEncode(chunk, P_ldpc);
    tx = 1 - 2 * cw;
    rx = awgn(tx, SNR_DB);
    llr_b = 2 * rx;
    dec_cw = ldpcDecodeMS(llr_b, H_ldpc, 20);
    % First 3 bits of cw are systematic; recover the message bits.
    for j = s:e
        if dec_cw(j - s + 1) ~= msg(j)
            err_ldpc = err_ldpc + 1;
        end
    end
end

% ============================================================
% Report
% ============================================================
fprintf('=== Modern codes BER comparison at SNR = %.0f dB, k = %.0f bits ===\n', ...
        SNR_DB, NUM_INFO);
fprintf('  uncoded BPSK : %.0f errors\n', err_unc);
fprintf('  Polar (128,64) SC decode : %.0f errors\n', err_pol);
fprintf('  Turbo PCCC (7,5)_8 6 iters: %.0f errors\n', err_tur);
fprintf('  LDPC (6,3) min-sum 20 iters: %.0f errors (across %.0f blocks)\n', ...
        err_ldpc, nb);
