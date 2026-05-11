% tier2_smoke.m — exercise every Tier-2 entry once.
%
% Order code:   0 = binary (natural), 1 = Gray
% Mod code:     0 = PAM, 1 = PSK, 2 = QAM, 3 = DPSK,
%               4 = FSK orth-coherent, 5 = FSK orth-noncoherent
% Shape code:   0 = root-raised-cosine ('sqrt'), 1 = full RC ('normal')
% UnitAveragePower flag: 0 = natural-power constellation, 1 = unit-power

rng(7);

% --- §4.1 PAM ---
fprintf('=== PAM (M=8, Gray) ===\n');
M = 8;
data = randi(M, 1, 12) - 1;
sym  = pammod(data, M, 1);                % Gray
back = pamdemod(sym, M, 1);
fprintf('symbols : '); disp(sym);
fprintf('round-trip errors : %.0f\n', symerrCount(data', back'));

% --- §4.3 PSK ---
fprintf('\n=== QPSK with pi/4 offset, Gray ===\n');
M = 4;
data = randi(M, 1, 8) - 1;
sym  = pskmod(data, M, pi/4, 1);
fprintf('|symbols|  : '); disp(abs(sym));    % all 1.0 on the unit circle
back = pskdemod(sym, M, pi/4, 1);
fprintf('round-trip errors : %.0f\n', symerrCount(data', back'));

% --- §4.2 QAM ---
fprintf('\n=== 16-QAM, Gray, unit avg power ===\n');
M = 16;
data = randi(M, 1, 200) - 1;
sym  = qammod(data, M, 1, 1);
% |sym| should be one of {0.316, 0.949, ...}; mean(|sym|^2) ≈ 1 (unit avg).
mag = abs(sym);
disp('first 8 |sym| values (unit-avg-power normalised):');
disp(mag(1:8));
back = qamdemod(sym, M, 1, 1);
fprintf('hard-decision symbol errors : %.0f\n', symerrCount(data', back'));

% Bit-output demod: returns N*log2(M) bits.
bits_out = qamdemodBit(sym, M, 1, 1);
fprintf('bit-output length (200 symbols x 4 bits) : %.0f\n', size(bits_out, 1));

% --- §4.6 generic constellation ---
% Build the 4-PSK constellation as the IQ pair (1, 0), (0, 1), (-1, 0),
% (0, -1) by reusing pskmod itself (any complex column would do — this
% just sidesteps the complex-literal-concat path).
fprintf('\n=== genqammod on user constellation ===\n');
alpha = pskmod((0:3)', 4, 0, 0);              % 4-PSK at indices 0..3
data  = randi(4, 1, 6) - 1;
sym   = genqammod(data, alpha);
fprintf('data : '); disp(data);
fprintf('|sym|: '); disp(abs(sym));
back = genqamdemod(sym, alpha);
fprintf('round-trip : '); disp(back);

% --- §4.7 pulse shaping ---
fprintf('\n=== rcosdesign / gaussdesign ===\n');
b = rcosdesign(0.25, 8, 8, 0);              % RRC, beta=0.25, span=8, sps=8
fprintf('RRC length (= 8*8+1 = 65) : %.0f\n', size(b, 1));
fprintf('RRC energy (norm^2 = 1)   : %.4f\n', norm(b) * norm(b));
g = gaussdesign(0.3, 4, 8);                  % GSM-style BT=0.3
fprintf('Gauss length (= 4*8+1)    : %.0f\n', size(g, 1));
fprintf('Gauss norm                : %.4f\n', norm(g));

% --- §4.8 berawgn ---
fprintf('\n=== berawgn closed-form curves at 10 dB Eb/N0 ===\n');
fprintf('  BPSK    : %.6e   (textbook ~3.87e-06)\n', berawgn(10, 2, 1));
fprintf('  QPSK    : %.6e\n', berawgn(10, 4, 1));
fprintf('  8-PSK   : %.6e\n', berawgn(10, 8, 1));
fprintf('  4-PAM   : %.6e\n', berawgn(10, 4, 0));
fprintf('  16-QAM  : %.6e   (textbook ~1.75e-03)\n', berawgn(10, 16, 2));
fprintf('  DBPSK   : %.6e\n', berawgn(10, 2, 3));
fprintf('  qfunc(3): %.6f   (textbook 1.35e-03)\n', qfunc(3.0));

% --- §4.9 scatterplot numeric form ---
fprintf('\n=== scatterplot numeric ===\n');
pts = scatterplot(sym);
fprintf('scatterplot first 4 (re, im) pairs:\n');
disp(pts(1:4, :));
