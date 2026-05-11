% longley_rice_link.m
% ====================================================================
% Stand-alone Longley-Rice (ITM) demo: walk a single PtP link's path
% loss as the reliability triple sweeps from (50,50,50) median to
% (95,99,99) microwave-design conservatism, then sweep climate codes
% and frequencies. Useful for sanity-checking the variability path.
% ====================================================================

freq_hz = 5.8e9;
ht      = 30.0;
hr      = 30.0;
d_total = 15e3;            % 15 km link
empty_profile = zeros(0, 1); % no terrain -> ITM falls back to area mode

% Polarisation + nominal Ns / sigma / eps_r
POL_VERTICAL = 1;
NS_DEFAULT   = 301;
SIG_AVG      = 0.005;
EPSR_AVG     = 15;

% --- Reliability sweep at climate = 5 (continental temperate) ---
disp('=== Longley-Rice 15 km flat link, 5.8 GHz, V-pol, climate = 5 ===');
qts = [50; 70; 80; 90; 95];
qls = [50; 90; 99; 99; 99];
qss = [50; 95; 99; 99; 99];
for k = 1:5
    qt = qts(k);
    ql = qls(k);
    qs_v = qss(k);
    L = itmPathloss(empty_profile, freq_hz, ht, hr, POL_VERTICAL, ...
                     5, NS_DEFAULT, SIG_AVG, EPSR_AVG, ...
                     d_total, qt, ql, qs_v);
    fprintf('  (q_time, q_loc, q_sit) = (%2.0f, %2.0f, %2.0f) : L = %.2f dB\n', ...
            qt, ql, qs_v, L);
end

% --- Climate sweep at (50, 50, 50) median ---
% Note: for short paths well within the radio horizon, free-space loss
% is the dominant floor and climate adjustments cannot push the result
% below it. To see climate effects clearly we pick a 60 km
% over-the-horizon path where the diffraction regime takes over.
disp(' ');
disp('=== Climate sweep at (50, 50, 50) median, 60 km path ===');
% Climate codes: 1 equatorial, 2 cont subtropical, 3 maritime subtropical,
% 4 desert, 5 cont temperate, 6 maritime over land, 7 maritime over sea.
d_long = 60e3;
for c = 1:7
    L = itmPathloss(empty_profile, freq_hz, ht, hr, POL_VERTICAL, ...
                     c, NS_DEFAULT, SIG_AVG, EPSR_AVG, ...
                     d_long, 50, 50, 50);
    fprintf('  climate code %.0f : L = %.2f dB\n', c, L);
end

% --- Frequency sweep at (80, 99, 99) ---
disp(' ');
disp('=== Frequency sweep at (80, 99, 99), climate = 5 ===');
fs = [0.3e9; 0.9e9; 2.4e9; 5.8e9; 11.0e9; 18.0e9];
for k = 1:6
    f = fs(k);
    L = itmPathloss(empty_profile, f, ht, hr, POL_VERTICAL, ...
                     5, NS_DEFAULT, SIG_AVG, EPSR_AVG, ...
                     d_total, 80, 99, 99);
    fprintf('  f = %5.1f GHz : L = %.2f dB\n', f * 1e-9, L);
end
