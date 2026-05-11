% pulse_shape_demo.m — pulse-shaping FIR design for digital comms.
%
% Demonstrates the canonical Tx/Rx pulse pair:
%   1. rcosdesign(beta, span, sps, 'sqrt')   -- RRC for matched-filter use
%   2. rcosdesign(beta, span, sps, 'normal') -- full RC (cascade RRC x RRC)
%   3. gaussdesign(BT, span, sps)            -- GMSK/GFSK Gaussian filter
%
% For each filter we print the impulse-response statistics so the
% headless lane can verify the design via numeric assertions (no
% interactive plotting required).

% --- RRC: typical 5G NR-style 0.25 roll-off, span 8 symbols, 8 sps.
beta = 0.25;
span = 8;
sps  = 8;
b_rrc = rcosdesign(beta, span, sps, 0);
b_rc  = rcosdesign(beta, span, sps, 1);

N_rrc  = size(b_rrc, 1);
centre = (N_rrc - 1) / 2 + 1;
peak_rrc = b_rrc(centre);
peak_rc  = b_rc(centre);

% norm(b) returns f64; squared norm equals filter energy (sum |b|^2).
e_rrc = norm(b_rrc) * norm(b_rrc);
e_rc  = norm(b_rc)  * norm(b_rc);
fprintf('=== Root-raised-cosine (beta=0.25, span=8, sps=8) ===\n');
fprintf('  length        : %.0f taps\n', N_rrc);
fprintf('  centre tap    : %.0f\n', centre);
fprintf('  peak value    : %.4f\n', peak_rrc);
fprintf('  energy        : %.4f (expected 1.0 unit-energy normalised)\n', e_rrc);

fprintf('\n=== Full raised-cosine (same params) ===\n');
fprintf('  peak value    : %.4f\n', peak_rc);
fprintf('  energy        : %.4f\n', e_rc);

% --- Gaussian: GSM-style BT=0.3, span=4, sps=8.
g_gsm = gaussdesign(0.3, 4, 8);
g_bt  = gaussdesign(0.5, 4, 8);

Ng    = size(g_gsm, 1);
g_ctr = (Ng - 1) / 2 + 1;
fprintf('\n=== Gaussian filter (span=4, sps=8) ===\n');
% disp() handles the matrix return from sum(); avoids fprintf-on-mat.
fprintf('  GSM (BT=0.3)        peak %.4f, sum =\n', g_gsm(g_ctr));
disp(sum(g_gsm));
fprintf('  Bluetooth (BT=0.5)  peak %.4f, sum =\n', g_bt(g_ctr));
disp(sum(g_bt));

% --- Matched-filter cascade: RRC x RRC should approximate a full RC,
% which has zero ISI at integer symbol multiples (the Nyquist criterion).
fprintf('\n=== RRC matched-filter cascade ===\n');
mf = conv(b_rrc, b_rrc);
Nmf = size(mf, 1);
c_mf = (Nmf - 1) / 2 + 1;
peak_mf  = mf(c_mf);
isi_1    = mf(c_mf + sps);
isi_2    = mf(c_mf + 2 * sps);
fprintf('  cascade length         : %.0f\n', Nmf);
fprintf('  centre tap (~peak)     : %.4f\n', peak_mf);
fprintf('  tap at centre + sps    : %.4f (expect small / Nyquist zero)\n', isi_1);
fprintf('  tap at centre + 2*sps  : %.4f\n', isi_2);
