% impairment_demo.m — applies the four canonical RF impairments to a
% clean QPSK constellation and reports the constellation-distortion
% statistics for each.
%
% Impairments demonstrated (PROP-Tier-4 §6.3):
%   1. Phase + frequency offset            — phaseFreqOffset
%   2. IQ amplitude + phase imbalance      — iqimbal
%   3. Memoryless PA nonlinearity (Rapp)   — memorylessNl (model 2)
%   4. Phase noise                         — phaseNoise

rng(2031);
N = 4000;
data = randi(4, N, 1) - 1;
clean = pskmod(data, 4, pi/4, 1);          % unit-power QPSK at pi/4

% Helper: report ||x - clean|| over the vector to show distortion.
ref_pow = norm(abs(clean));

fprintf('=== Clean QPSK reference ===\n');
fprintf('  ||abs(clean)|| (sqrt(N)=%.2f): %.4f\n', sqrt(N), ref_pow);

% --- 1. Phase + frequency offset ---
% Slow drift across the burst.  df_Hz / fs_Hz = 1e-4 -> 0.4 rad over N=4000.
fprintf('\n=== 1. phaseFreqOffset (df = 1e-4 normalised) ===\n');
y_pf = phaseFreqOffset(clean, 1e-4, 1.0);
fprintf('  |x| preserved (norm diff ~0)     : %.6e\n', ...
        norm(abs(y_pf) - abs(clean)));

% --- 2. IQ imbalance ---
fprintf('\n=== 2. iqimbal (0.5 dB amp / 5 deg phase) ===\n');
y_iq = iqimbal(clean, 0.5, 5.0);
fprintf('  ||abs(y) - abs(clean)|| (RF-amp shift visible): %.4f\n', ...
        norm(abs(y_iq) - abs(clean)));

% --- 3. Memoryless Rapp PA ---
% Operate at 1.5x rated amplitude so saturation kicks in.
fprintf('\n=== 3. memorylessNl - Rapp (p=3, Asat=1, drive=1.5) ===\n');
clean_hot = clean * 1.5;
y_pa = memorylessNl(clean_hot, 2, 3.0, 1.0, 0, 0);
fprintf('  pre-PA  ||abs(hot)||  : %.4f (=1.5*sqrt(N))\n', norm(abs(clean_hot)));
fprintf('  post-PA ||abs(y_pa)|| : %.4f (clipped near sqrt(N))\n', norm(abs(y_pa)));

% --- 4. Phase noise ---
% -90 dBc/Hz integrated over fs=1 MHz - moderate oscillator quality.
fprintf('\n=== 4. phaseNoise (-90 dBc/Hz @ 1 MHz) ===\n');
y_pn = phaseNoise(clean, -90, 1.0e6);
fprintf('  |x| preserved (unit-magnitude rotation): %.6e\n', ...
        norm(abs(y_pn) - abs(clean)));

% --- Combined chain: PA + phaseFreqOffset + IQ imbalance + phaseNoise ---
fprintf('\n=== Combined RF chain ===\n');
y = clean * 1.2;
y = memorylessNl(y, 2, 3.0, 1.0, 0, 0);
y = phaseFreqOffset(y, 5e-5, 1.0);
y = iqimbal(y, 0.3, 3.0);
y = phaseNoise(y, -100, 1.0e6);
fprintf('  ||y|| after full chain: %.4f\n', norm(abs(y)));
fprintf('  ||y - clean||         : %.4f (constellation degradation)\n', ...
        norm(abs(y) - abs(clean)));
