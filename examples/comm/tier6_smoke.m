% tier6_smoke.m — exercise every Tier-6 entry once.
%
% Tier-6 covers spreading sequences (PN / Gold / Walsh-Hadamard) and
% source coding (uniform quantiser, A-law / mu-law companding, DPCM,
% Lloyd-Max optimisation).  All function-form; the System-Object
% surfaces stay gated on the SO lowering fix.

% --- §8.1 PN sequence: x^4 + x + 1 (period 15) ---
disp('=== §8.1 PN sequence (period 15) ===');
poly = 19;                       % 0b10011
pn = pnSequence(poly, 1, 32, 0); % length 32, output_mode 0 = {0,1}
disp(pn(1:15)');                 % one period

% Bipolar form (output_mode = 1 -> {-1, +1})
pn_bp = pnSequence(poly, 1, 16, 1);
disp(pn_bp(1:15)');

% --- §8.1 Gold sequence: XOR of two preferred-pair PNs ---
disp(' ');
disp('=== §8.1 Gold sequence ===');
gold = goldSequence(19, 25, 1, 1, 32, 1);
disp(gold(1:16)');

% --- §8.1 Hadamard + Walsh codes ---
disp(' ');
disp('=== §8.1 Hadamard(4) (Walsh codes as rows) ===');
H = hadamard(4);
disp(H);
disp('walshCode(8, 3) - third Walsh code at length 8');
disp(walshCode(8, 3)');

% --- §8.2 uniform 4-level quantiser on a sin wave ---
disp(' ');
disp('=== §8.2 quantiz / quantizApply (4-level uniform) ===');
t = linspace(0, 2*pi, 12)';
x = sin(t);
part = [-0.5; 0; 0.5];
cb   = [-0.75; -0.25; 0.25; 0.75];
idx = quantiz(x, part, cb);
qx  = quantizApply(idx, cb);
fprintf('codebook indices: '); disp(idx');
fprintf('quantised x     : '); disp(qx');

% --- §8.2 Lloyd-Max codebook optimisation ---
disp(' ');
disp('=== §8.2 lloydsQuant (4 levels, Gaussian-ish input) ===');
rng(2034);
sig = randn(2000, 1);
init_cb = [-1.5; -0.5; 0.5; 1.5];
opt_cb  = lloydsQuant(sig, init_cb, 30, 1e-6);
fprintf('initial codebook: '); disp(init_cb');
fprintf('optimised cb    : '); disp(opt_cb');

% --- §8.2 mu-law companding round-trip ---
disp(' ');
disp('=== §8.2 mu-law compand round-trip (G.711) ===');
x = linspace(-1, 1, 9)';
mu_compress = compandMu(x, 255, 1, 0);
mu_expand   = compandMu(mu_compress, 255, 1, 1);
fprintf('compressed: '); disp(mu_compress');
fprintf('round-trip: '); disp(mu_expand');
fprintf('round-trip error: %.3e\n', norm(x - mu_expand));

% --- §8.2 A-law companding round-trip ---
disp(' ');
disp('=== §8.2 A-law compand round-trip ===');
a_compress = compandA(x, 87.6, 1, 0);
a_expand   = compandA(a_compress, 87.6, 1, 1);
fprintf('compressed: '); disp(a_compress');
fprintf('round-trip error: %.3e\n', norm(x - a_expand));

% --- §8.2 DPCM encode / decode ---
disp(' ');
disp('=== §8.2 DPCM (delta-modulated sin wave) ===');
t = linspace(0, 4*pi, 40)';
x = 0.8 * sin(t);
% Residual codebook: small +/- step + zero
part_d = [-0.05; 0.05];
cb_d   = [-0.1; 0.0; 0.1];
idx_d  = dpcmEncode(x, part_d, cb_d);
xhat   = dpcmDecode(idx_d, cb_d);
fprintf('||original - DPCM reconstruction||: %.3f\n', norm(x - xhat));
