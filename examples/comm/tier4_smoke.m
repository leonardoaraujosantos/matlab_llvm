% tier4_smoke.m — exercise every Tier-4 entry once.
%
% Tier-4 covers equalisation, sync, and RF impairments. All are
% function-form here; the comm.LinearEqualizer / DFE /
% CarrierSynchronizer / SymbolSynchronizer / PreambleDetector /
% PhaseNoise / MemorylessNonlinearity System Objects stay gated on
% the SO lowering fix.

rng(2030);

% --- §6.1 LMS / RLS adaptive equalisers ---
% Channel with first tap dominant so the equaliser has no delay to
% absorb and converges directly against d = tx.
disp('=== §6.1 LMS / RLS ===');
ch = [1.0; 0.3; -0.15];
N = 4000;
tx = 2 * (randi(2, N, 1) - 1) - 1;
rx = conv(tx, ch);
rx = rx(1:N);
rx_n = awgn(rx, 30);
y  = lms(rx_n, tx, 0.05, 7);
yr = rls(rx_n, tx, 0.999, 100, 7);
eq_lms = (y (end-499:end) >= 0);
eq_rls = (yr(end-499:end) >= 0);
ref    = (tx(end-499:end) >= 0);
fprintf('LMS post-converge BER : %.4f\n', biterr(ref, eq_lms));
fprintf('RLS post-converge BER : %.4f\n', biterr(ref, eq_rls));

% --- §6.1 CMA blind equaliser (uses constant-modulus property) ---
% Unit-circle PSK source; the CMA criterion drives |y|^2 -> R2 = 1.
disp(' ');
disp('=== §6.1 CMA blind equaliser ===');
M = 4;
nsym = 3000;
data = randi(M, nsym, 1) - 1;
sym  = pskmod(data, M, 0, 0);            % complex
% We only have a real CMA in the runtime so far; flatten on the I axis.
% (Complex CMA is a Tier-4 follow-on; we exercise the API on the real
% projection so the smoke test still hits the dispatch path.)
rx_re = awgn(real(sym), 20);
y_cma = cma(rx_re, 0.001, 9, 1.0);
% Convergence indicator: ||y_tail|| / sqrt(100) -> sqrt(R2) = 1 if
% the constant-modulus criterion is satisfied.  norm returns f64
% (sum() returns a 1x1 matrix, which fprintf can't take).
last_100 = y_cma(end-99:end);
fprintf('CMA mean |y|^2 last 100 (target 1.0): %.4f\n', ...
        norm(last_100) * norm(last_100) / 100.0);

% --- §6.1 DFE decision-feedback equaliser ---
disp(' ');
disp('=== §6.1 DFE ===');
ydfe = dfe(rx_n, tx, 0.02, 5, 3);
eq_dfe = (ydfe(end-499:end) >= 0);
fprintf('DFE post-converge BER: %.4f\n', biterr(ref, eq_dfe));

% --- §6.2 sync ---
disp(' ');
disp('=== §6.2 Costas PLL (de-rotates a fixed offset) ===');
clean = pskmod(randi(4, 512, 1) - 1, 4, pi/4, 1);
rot   = phaseFreqOffset(clean, 0.002, 1.0);
locked = costasPll(rot, 4, 0.01, 1.0);
% Sanity: the locked output should sit closer to the canonical 4-PSK
% constellation than the rotated input.  Use ||Re .* Im|| which is
% near zero on the constellation axis points (e.g. e^{j pi/4}).
prod_in  = real(rot)    .* imag(rot);
prod_out = real(locked) .* imag(locked);
fprintf('||Re .* Im|| before lock: %.4f\n', norm(prod_in));
fprintf('||Re .* Im|| after  lock: %.4f\n', norm(prod_out));

disp(' ');
disp('=== §6.2 Mueller-Mueller symbol timing (4 sps -> symbol rate) ===');
sps = 4;
nsym2 = 32;
tx_sym = 2 * (randi(2, nsym2, 1) - 1) - 1;
samples = zeros(nsym2 * sps, 1);
for k = 1:nsym2
    samples((k-1)*sps + 1 : k*sps) = tx_sym(k);
end
syms_out = symbolSyncMM(samples, sps, 0.05);
sliced = 2 * (syms_out >= 0) - 1;
n_out = size(sliced, 1);
match_n = sum(sliced(n_out-19:n_out) == tx_sym(nsym2-19:nsym2));
fprintf('last-20 symbol-sync count: '); disp(match_n);

disp(' ');
disp('=== §6.2 preamble detection ===');
preamble = [1; -1; 1; 1; -1; 1; -1; -1];
frame = zeros(40, 1);
frame(11:18) = preamble;
fprintf('detected index (expect 11): %.0f\n', preambleDetect(frame, preamble));

% --- §6.3 RF impairments ---
disp(' ');
disp('=== §6.3 phaseFreqOffset round-trip ===');
clean = pskmod((0:31)', 4, 0, 0);
rot   = phaseFreqOffset(clean, 0.05, 1.0);
und   = phaseFreqOffset(rot, -0.05, 1.0);
fprintf('|abs round-trip error|: %.6e\n', norm(abs(clean) - abs(und)));

disp(' ');
disp('=== §6.3 IQ imbalance (0.5 dB / 5 deg) ===');
sym2 = pskmod((0:99)', 4, pi/4, 1);
sym_imb = iqimbal(sym2, 0.5, 5.0);
fprintf('||abs(clean)|| = %.4f, ||abs(imb)|| = %.4f\n', ...
        norm(abs(sym2)), norm(abs(sym_imb)));

disp(' ');
disp('=== §6.3 memorylessNl — Rapp PA ===');
sym3 = pskmod((0:99)', 4, pi/4, 1);
out_rapp = memorylessNl(sym3, 2, 3.0, 1.0, 0, 0);
fprintf('||abs(out_rapp)|| (target near sqrt(100)=10 with sat=1.0): %.4f\n', ...
        norm(abs(out_rapp)));

disp(' ');
disp('=== §6.3 phaseNoise (-100 dBc/Hz at fs=1 MHz) ===');
sym4 = pskmod((0:127)', 4, pi/4, 1);
sym_pn = phaseNoise(sym4, -100, 1.0e6);
% Check that magnitudes are preserved (phase noise is unit-modulus).
fprintf('|x| identity after phaseNoise: %.6e\n', ...
        norm(abs(sym_pn) - abs(sym4)));

disp(' ');
disp('=== §6.x soft Viterbi vs hard ===');
gens = [oct2dec(171), oct2dec(133)];
t = poly2trellis(7, gens);
msg = randi(2, 200, 1) - 1;
code = convenc(msg, t);
% Modulate to BPSK +/-1, add noise, demod to hard bits
tx_b = 1 - 2 * code;
rx_b = awgn(tx_b, 3);                    % low-ish SNR
hard_bits = (rx_b < 0);
hard_dec  = vitdec(hard_bits, t, 25, 0, 1);
% Soft input: LLR ~ 2*y/sigma^2 (sigma known a-priori). At SNR 3 dB
% the noise variance for unit-power BPSK is ~10^(-0.3) = 0.501;
% LLR = 2*rx/sigma^2; positive favours bit=0 (matches the
% qamdemodLlr convention).
llr = 2 * rx_b / 0.501;
soft_dec = vitdecSoft(llr, t, 25, 0);
fprintf('hard Viterbi errors: %.0f\n', biterrCount(msg, hard_dec));
fprintf('soft Viterbi errors: %.0f\n', biterrCount(msg, soft_dec));
