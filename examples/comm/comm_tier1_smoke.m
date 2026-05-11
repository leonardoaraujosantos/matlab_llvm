% comm_tier1_smoke.m — exercise every Comm-Tier-1 entry once.
%
% This is the smoke fixture: walks rng / randi / randsrc / randerr /
% int2bit / bit2int / de2bi / bi2de / awgn / biterr / symerr in
% canonical shapes. Numbers are determinism-bound by the rng(42)
% seed at the top; change the seed for a fresh run.

rng(42);
fprintf('=== rng / randi ===\n');
fprintf('Saved state    : %.0f\n', rngGet());
fprintf('randi(10)      : %.0f\n', randi(10));
disp('randi(4, 3, 5) =');
disp(randi(4, 3, 5));

fprintf('\n=== randsrc / randerr ===\n');
alpha = [-3; -1; 1; 3];   % 4-PAM
S = randsrc(2, 6, alpha);
disp('randsrc(2, 6, [-3 -1 1 3]) =');
disp(S);
E = randerr(3, 8, 2);
disp('randerr(3, 8, 2) =');
disp(E);

fprintf('\n=== int2bit / bit2int (MSB-first) ===\n');
ints = [5; 2; 7];
bits = int2bit(ints, 3);
disp('bits =');
disp(bits');
back = bit2int(bits, 3);
disp('round-trip ints =');
disp(back');

fprintf('\n=== de2bi / bi2de (legacy LSB-first) ===\n');
b = de2bi(ints, 3);
disp('de2bi(ints, 3) =');
disp(b);
d = bi2de(b);
disp('bi2de(b) =');
disp(d');

fprintf('\n=== awgn (real, 10 dB SNR) ===\n');
% Built-in random Gaussian source; AWGN adds noise on top to hit SNR.
n = 4096;
x = randn(1, n);
y = awgn(x, 10);
% Sanity: mean stays near 0, sample variance ~ 1 + 1/snr_lin = 1.1.
disp('mean(y) =');
disp(mean(y));
disp('var(y) =');
disp(var(y));

fprintf('\n=== biterr / symerr ===\n');
% Hand-crafted bit-error count: 3 of 8 differ.
tx = [0; 1; 1; 0; 1; 0; 1; 1];
rx = [0; 0; 1; 0; 1; 1; 1; 0];
fprintf('biterrCount  : %.0f (expect 3)\n', biterrCount(tx, rx));
fprintf('biterr (BER) : %.4f (expect 0.3750)\n',  biterr(tx, rx));
% Symbol-level: 4-PAM symbols where 2 of 5 differ.
txs = [0; 1; 2; 3; 1];
rxs = [0; 2; 2; 1; 1];
fprintf('symerrCount  : %.0f (expect 2)\n', symerrCount(txs, rxs));
fprintf('symerr (SER) : %.4f (expect 0.4000)\n', symerr(txs, rxs));
% k-bit symbol BER: each symbol carries 2 bits.
fprintf('biterrK ratio (k=2) : %.4f\n', biterrK(txs, rxs, 2));
