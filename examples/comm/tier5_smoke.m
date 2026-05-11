% tier5_smoke.m — exercise every Tier-5 entry once.
%
% Tier-5 covers OFDM, Rayleigh / Rician fading channels, and Alamouti
% 2-Tx space-time block coding.  Function-form throughout — the
% comm.OFDMModulator / RayleighChannel / RicianChannel / OSTBC*
% System Objects stay gated on the SO lowering fix.
%
% Convention notes:
%   - All OFDM / fading / Alamouti entries operate on complex columns
%     (matlab_mat_c).  Downstream code uses abs(...) before any
%     `size` / scalar-indexing operation because the size() runtime
%     reads the real-matrix layout (the complex-magic-aware size
%     polymorphism is a Tier-2/3 follow-on).
%   - `delays` and `gains_dB` for the fading channels must be column
%     vectors with at least two elements (single-element `[0]` literal
%     gets typed as scalar f64 and fails the dispatch).

rng(2032);

% --- §7.1 OFDM round-trip ---
disp('=== §7.1 OFDM (Nfft = 64, CP = 16, Nsym = 1) ===');
Nfft = 64;
Lcp  = 16;
sym = pskmod(randi(4, Nfft, 1) - 1, 4, pi/4, 1);
tx  = ofdmmod(sym, Nfft, Lcp);
rx_data = ofdmdemod(tx, Nfft, Lcp);
% Magnitude-domain check (avoids size-on-complex pitfall).
err_round = norm(abs(sym) - abs(rx_data));
fprintf('OFDM round-trip ||abs|| error: %.6e\n', err_round);
% Length of the time-domain stream via abs-then-length.
tx_real = abs(tx);
fprintf('OFDM tx samples per symbol: %.0f (expect %.0f)\n', ...
        size(tx_real, 1), Nfft + Lcp);

% --- §7.2 Rayleigh channel ---
disp(' ');
disp('=== §7.2 Rayleigh (2-tap, static, no Doppler) ===');
src = pskmod(randi(4, 1024, 1) - 1, 4, 0, 0);
delays = [0; 4];
gains_dB = [0; -3];
y_ray = rayleighChannel(src, delays, gains_dB, 0.0, 1.0);
% Effective channel-output length via magnitude.
y_ray_mag = abs(y_ray);
fprintf('Rayleigh output length: %.0f (input 1024 + max_delay 4)\n', ...
        size(y_ray_mag, 1));
% Average output power should equal sum of path-linear gains squared.
fprintf('||y_ray||^2 / N: %.4f\n', ...
        norm(y_ray_mag) * norm(y_ray_mag) / size(y_ray_mag, 1));

% --- §7.2 Rician channel ---
disp(' ');
disp('=== §7.2 Rician (K = 10 dB, 2-tap) ===');
y_ric = ricianChannel(src, 10, delays, gains_dB, 0.0, 1.0);
y_ric_mag = abs(y_ric);
fprintf('Rician output length: %.0f\n', size(y_ric_mag, 1));
fprintf('||y_ric||^2 / N: %.4f (LOS-dominated should approach 1.0)\n', ...
        norm(y_ric_mag) * norm(y_ric_mag) / size(y_ric_mag, 1));

% --- §7.3 Alamouti encode ---
disp(' ');
disp('=== §7.3 Alamouti encode (2-Tx) ===');
N_pairs = 200;
src2 = pskmod(randi(4, N_pairs, 1) - 1, 4, pi/4, 1);
encoded = ostbcEncode(src2);
% encoded is N x 2 complex; just confirm via abs.
enc_mag = abs(encoded);
fprintf('Alamouti encoded rows: %.0f cols: %.0f\n', ...
        size(enc_mag, 1), size(enc_mag, 2));

% --- §7.3 ML detector on a 4-PSK alphabet ---
disp(' ');
disp('=== §7.3 mlDetect (4-PSK alphabet, 10 dB AWGN) ===');
alpha = pskmod((0:3)', 4, pi/4, 1);
tx_ml = pskmod(randi(4, 64, 1) - 1, 4, pi/4, 1);
rx_ml = awgn(tx_ml, 10);
labels = mlDetect(rx_ml, alpha);
fprintf('mlDetect first 6 labels: '); disp(labels(1:6)');
