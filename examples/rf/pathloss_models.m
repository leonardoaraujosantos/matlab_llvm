% pathloss_models.m
% ====================================================================
% Compare the closed-form propagation models in PROP-Tier-1a at a single
% link geometry. Useful for picking a model when only an order-of-
% magnitude estimate is needed.
%
% Geometry: 30 m base station, 1.5 m mobile, 5 km link, 2.4 GHz.
% Outputs the predicted path loss in dB from each model.
% ====================================================================

% Geometry
f_Hz    = 2.4e9;
f_MHz   = f_Hz * 1e-6;
ht      = 30.0;     % m
hr      = 1.5;      % m
d_m     = 5000.0;   % 5 km
d_km    = d_m * 1e-3;

% Environment codes
ENV_URBAN_LARGE = 1;
ENV_SUBURBAN    = 3;
ENV_OPEN        = 4;
TERR_B          = 2;  % SUI terrain category

% Closed-form ITU-R / NIST
L_fs    = fspl(d_m, f_Hz);
L_ci    = pathlossCloseIn(d_m, f_Hz, 3.0, 4.0, 1.0);

% Cellular empirical
L_hata_urban = pathlossHata     (f_MHz, ht, hr, d_km, ENV_URBAN_LARGE);
L_hata_sub   = pathlossHata     (f_MHz, ht, hr, d_km, ENV_SUBURBAN);
L_hata_open  = pathlossHata     (f_MHz, ht, hr, d_km, ENV_OPEN);
L_cost231    = pathlossCost231  (f_MHz, ht, hr, d_km, ENV_URBAN_LARGE);
L_egli       = pathlossEgli     (f_MHz, ht, hr, d_km);
L_ecc33      = pathlossEcc33    (f_MHz, ht, hr, d_km);
L_sui        = pathlossSui      (f_MHz, ht, hr, d_km, TERR_B);
L_eric       = pathlossEricsson9999(f_MHz, ht, hr, d_km, ENV_URBAN_LARGE);

% Atmospheric add-ons
L_rain  = pathlossRain (d_m, f_Hz, 25.0, 1.0);   % 25 mm/h, vertical pol
L_gas   = pathlossGas  (d_m, f_Hz, 288.15, 1013.25, 10.0);
L_fog   = pathlossFog  (d_m, f_Hz, 0.05);        % light fog

fprintf('=== Path-loss model comparison (5 km, 2.4 GHz) ===\n');
fprintf('Free-space (FSPL)       : %.2f dB\n', L_fs);
fprintf('Close-in (n=3, sigma=4) : %.2f dB\n', L_ci);
fprintf('Hata urban-large        : %.2f dB\n', L_hata_urban);
fprintf('Hata suburban           : %.2f dB\n', L_hata_sub);
fprintf('Hata open (rural)       : %.2f dB\n', L_hata_open);
fprintf('COST-231 urban-large    : %.2f dB\n', L_cost231);
fprintf('Egli VHF/UHF            : %.2f dB\n', L_egli);
fprintf('ECC-33 urban            : %.2f dB\n', L_ecc33);
fprintf('SUI terrain B           : %.2f dB\n', L_sui);
fprintf('Ericsson 9999 urban     : %.2f dB\n', L_eric);
fprintf('\n=== Atmospheric add-ons (5 km, 2.4 GHz) ===\n');
fprintf('Rain (25 mm/h, V pol)   : %.4f dB total\n', L_rain);
fprintf('Gas (15 C, 1013 hPa, 10 g/m^3) : %.4f dB total\n', L_gas);
fprintf('Light fog (0.05 g/m^3)  : %.4f dB total\n', L_fog);
