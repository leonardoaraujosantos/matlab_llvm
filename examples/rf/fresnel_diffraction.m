% fresnel_diffraction.m
% ====================================================================
% Fresnel-zone radii + multi-edge diffraction loss on a synthetic
% terrain profile, demonstrating PROP-Tier-1a §3.1.3-3.1.4.
% ====================================================================

% Link geometry: 10 km path, 5.8 GHz, both ends 20 m above ground.
d_total  = 10e3;
freq_hz  = 5.8e9;
lambda   = 2.998e8 / freq_hz;
h_tx     = 20.0;
h_rx     = 20.0;

% --- Fresnel zone radii at mid-path for the first three zones ---
r1 = fresnelZoneRadius(d_total/2, d_total/2, lambda, 1);
r2 = fresnelZoneRadius(d_total/2, d_total/2, lambda, 2);
r3 = fresnelZoneRadius(d_total/2, d_total/2, lambda, 3);
fprintf('=== Fresnel zones at link midpoint (10 km, 5.8 GHz) ===\n');
fprintf('1st zone radius : %.2f m\n', r1);
fprintf('2nd zone radius : %.2f m\n', r2);
fprintf('3rd zone radius : %.2f m\n', r3);

% --- Synthetic 64-point terrain profile with two ridges between TX/RX.
N = 64;
x = linspace(0, 1, N);
% Two Gaussian humps at x=0.3 and x=0.65, peaks at 15 and 25 m.
ridge1 = 15.0 * exp(-((x - 0.30) / 0.05).^2);
ridge2 = 25.0 * exp(-((x - 0.65) / 0.04).^2);
profile = (ridge1 + ridge2)';   % column vector

% --- Single-edge knife-edge using the tallest peak.
% We know the synthesised tallest peak is the 25 m ridge2 at x = 0.65.
peak_h_scalar = 25.0;
d1 = 0.65 * d_total;
d2 = d_total - d1;
los_h = h_tx + 0.65 * (h_rx - h_tx);
h_obs = peak_h_scalar - los_h;
L_ke = diffractionKnifeEdge(h_obs, d1, d2, lambda);

% --- Multi-edge methods: Bullington (single equivalent edge) and
% Deygout (recursive 3-edge).
L_bull = diffractionBullington(profile, h_tx, h_rx, d_total, lambda);
L_deyg = diffractionDeygout   (profile, h_tx, h_rx, d_total, lambda);

% --- Fresnel clearance percentage along the link.
clear_pct = fresnelClearance(profile, h_tx, h_rx, d_total, lambda, 1.0);

fprintf('\n=== Two-ridge synthetic terrain ===\n');
fprintf('Tallest peak height       : %.2f m\n', peak_h_scalar);
fprintf('Single-edge knife-edge L  : %.2f dB\n', L_ke);
fprintf('Bullington equivalent edge: %.2f dB\n', L_bull);
fprintf('Deygout 3-edge method     : %.2f dB\n', L_deyg);
fprintf('Fresnel clearance         : %.1f %% (>60%% = TIA clean)\n', clear_pct);
