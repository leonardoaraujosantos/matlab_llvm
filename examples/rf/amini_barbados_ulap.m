% amini_barbados_ulap.m
% ============================================================
% Amini Barbados ULAP Site Survey
%
% Three sites in Bridgetown sharing two directional PtP links and a
% per-site 5G access bubble.  Uses the function-form PROP runtime
% (`itmPathloss` / `linkBudget` / `coverageGridMulti` / `fresnelZoneRadius`
% / `losObstruction`) shipped in PROP-Tier-1a/2a/2b/3 of
% `docs/comm_toolbox_roadmap.md §3`.
%
% Sites and infrastructure constraints:
%
%   - Police Command Center
%       lat = 13.112279, lon = -59.603618
%       building = 16 m, mast on top
%       role: PtP node + 3 km 5G access
%
%   - Ilaro Court
%       lat = 13.103643, lon = -59.586568
%       ground-level antenna (no building)
%       role: PtP peer + 3 km 5G access
%
%   - Queen Elizabeth Hospital
%       lat = 13.095956, lon = -59.606678
%       building = 22 m, mast on top
%       role: PtP node + sector-coverage 5G inside the hospital
%
% Workflow per directional link:
%   1. Haversine distance + initial bearing.
%   2. Terrain profile along the great-circle path (sampled from a
%      synthetic DEM — replace with `load('srtm.mat').heights` when
%      a real DEM is available).
%   3. Longley-Rice (climate 3 maritime subtropical) median path loss
%      at 5.8 GHz with vertical polarisation, reliability triple
%      80/99/99 (TSB-10F microwave-link conservatism).
%   4. First-Fresnel-zone radius at link mid-point.
%   5. LOS-clearance check (4/3 Earth-bulge); reports the worst
%      obstruction height above the line-of-sight chord.
%   6. Suggested minimum mast height to clear the 60 % first-Fresnel
%      bar (`fresnel_60 = 0.6 · F1` at link midpoint).
%   7. Link budget against a 5 W (37 dBm) TX with 22 dBi cosine-
%      pattern directional dishes both ends.
%
% Workflow per site for 5G access bubble:
%   - 3 km × 3 km grid centred on each site.
%   - 3.5 GHz n78 band, 10 W (40 dBm) TX, three 120° sectors per
%     site (Police + Ilaro) or one omni equivalent at the Hospital
%     for indoor coverage representativity.
%   - Coverage = pixels above the -85 dBm threshold typical for 600
%     Mbit 5G NR.
%
% Numeric outputs only; the coverage grid is summarised by basic
% statistics (min / median / max RX power, % cells above the
% threshold).  The matrix itself can be dumped to PNG via the
% Cairo backend separately.
% ============================================================

% ---------- Sites ----------
P_LAT = 13.112279;  P_LON = -59.603618;     P_BLDG = 16;  P_MAST = 5;
I_LAT = 13.103643;  I_LON = -59.586568;     I_BLDG = 0;   I_MAST = 30;
H_LAT = 13.095956;  H_LON = -59.606678;     H_BLDG = 22;  H_MAST = 5;

% Effective antenna height above ground (terrain + building + mast)
P_AGL = P_BLDG + P_MAST;
I_AGL = I_BLDG + I_MAST;
H_AGL = H_BLDG + H_MAST;

fprintf('=== Amini Barbados ULAP — site inventory ===\n');
fprintf('  Police Command  : (%.6f, %.6f)\n', P_LAT, P_LON);
fprintf('    AGL %.0f m  = building %.0f + mast %.0f\n', P_AGL, P_BLDG, P_MAST);
fprintf('  Ilaro Court     : (%.6f, %.6f)\n', I_LAT, I_LON);
fprintf('    AGL %.0f m  = ground + mast %.0f\n', I_AGL, I_MAST);
fprintf('  QE Hospital     : (%.6f, %.6f)\n', H_LAT, H_LON);
fprintf('    AGL %.0f m  = building %.0f + mast %.0f\n', H_AGL, H_BLDG, H_MAST);

% ---------- Synthetic Bridgetown-area DEM ----------
% 48x48 cells across the bounding box.  Bridgetown sits at near-sea-
% level with a gentle northeast rise.  Replace with a real DEM when
% available — runtime accepts any matlab_mat as the heightmap.
LAT_MIN = 13.07; LAT_MAX = 13.16;
LON_MIN = -59.65; LON_MAX = -59.55;
NLAT = 48; NLON = 48;
lat_v = linspace(LAT_MIN, LAT_MAX, NLAT);
lon_v = linspace(LON_MIN, LON_MAX, NLON);
[LON, LAT] = meshgrid(lon_v, lat_v);
% Gentle NE-direction terrain rise (max ~30 m near the upper corner)
% plus a slight depression around Bridgetown centre.
DX = (LAT - 13.07) / 0.09;
DY = (LON + 59.65) / 0.10;
heightmap = 30 * (DX + DY) / 2 - 5 * exp(-((LAT - 13.10) / 0.03).^2 ...
                                          -((LON + 59.605) / 0.03).^2);
% Floor at 0 (sea level).
heightmap = max(heightmap, zeros(NLAT, NLON));

% ---------- Common link parameters ----------
FREQ_DIR  = 5.8e9;          % unlicensed PtP band
LAMBDA    = 2.998e8 / FREQ_DIR;
TX_PWR_W  = 5;              % 5 W = ~37 dBm
TX_GAIN   = 22;             % 22 dBi cosine-pattern dish (typical 60 cm)
RX_GAIN   = 22;             % matching dish at the far end
MODEL_ITM = 7;
CLIM_MAR  = 3;              % maritime subtropical
TIME_Q    = 80;             % 80 % time
LOC_Q     = 99;             % 99 % location
SIT_Q     = 99;             % 99 % situation

% ============================================================
% Link 1: Police Command Center  <->  Ilaro Court
% ============================================================
fprintf('\n=== Link 1: Police Command Center <-> Ilaro Court ===\n');

d_PI = haversine(P_LAT, P_LON, I_LAT, I_LON);
az_PI = bearing(P_LAT, P_LON, I_LAT, I_LON);
az_IP = bearing(I_LAT, I_LON, P_LAT, P_LON);
fprintf('  distance  : %.2f km\n', d_PI / 1000.0);
fprintf('  bearing P -> I: %.2f deg compass\n', az_PI);
fprintf('  bearing I -> P: %.2f deg compass\n', az_IP);

prof_PI = terrainProfile(heightmap, ...
                          LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                          P_LAT, P_LON, I_LAT, I_LON, 128);
los_PI = losObstruction(prof_PI, P_AGL, I_AGL, d_PI);
fprintf('  worst-point obstruction above LOS chord: %+.2f m\n', los_PI);
fresnel_F1_mid_PI = fresnelZoneRadius(d_PI/2, d_PI/2, LAMBDA, 1);
fresnel_60_PI     = 0.6 * fresnel_F1_mid_PI;
fprintf('  first-Fresnel radius at midpoint: %.2f m (60%% bar = %.2f m)\n', ...
        fresnel_F1_mid_PI, fresnel_60_PI);

% Suggested minimum mast height adjustment on the Ilaro end to clear
% the 60 % bar.  losObstruction returned the worst-point offset above
% the line-of-sight chord; positive means a tree / hump pokes above.
% To clear the 60 % first-Fresnel bar we need the line-of-sight to
% sit at least fresnel_60 metres above the worst point.  If
% los_PI < -fresnel_60 we already pass; otherwise raise Ilaro by the
% deficit.
deficit_PI = los_PI + fresnel_60_PI;
if deficit_PI > 0
    fprintf('  60%% Fresnel bar NOT met — raise Ilaro mast by %.1f m (or both ends share the lift).\n', ...
            deficit_PI);
else
    fprintf('  60%% Fresnel bar CLEAR (margin %.1f m above the bar).\n', -deficit_PI);
end

lb_PI = linkBudget(P_LAT, P_LON, P_AGL, FREQ_DIR, TX_PWR_W, TX_GAIN, ...
                    I_LAT, I_LON, I_AGL,                  RX_GAIN, ...
                    MODEL_ITM, prof_PI, CLIM_MAR, TIME_Q, LOC_Q, SIT_Q);
fprintf('  Longley-Rice path loss   : %.2f dB\n', lb_PI.PathLoss);
fprintf('  TX power                 : %.2f dBm\n', lb_PI.TxPower_dBm);
fprintf('  Received power at Ilaro  : %.2f dBm\n', lb_PI.ReceivedPower);
fprintf('  SNR (vs kT*1MHz)         : %.2f dB\n', lb_PI.Snr);
fprintf('  Link margin (10 dB thr)  : %.2f dB\n', lb_PI.LinkMargin);

% ============================================================
% Link 2: Police Command Center  <->  Queen Elizabeth Hospital
% ============================================================
fprintf('\n=== Link 2: Police Command Center <-> Queen Elizabeth Hospital ===\n');

d_PH = haversine(P_LAT, P_LON, H_LAT, H_LON);
az_PH = bearing(P_LAT, P_LON, H_LAT, H_LON);
az_HP = bearing(H_LAT, H_LON, P_LAT, P_LON);
fprintf('  distance  : %.2f km\n', d_PH / 1000.0);
fprintf('  bearing P -> H: %.2f deg compass\n', az_PH);
fprintf('  bearing H -> P: %.2f deg compass\n', az_HP);

prof_PH = terrainProfile(heightmap, ...
                          LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                          P_LAT, P_LON, H_LAT, H_LON, 128);
los_PH = losObstruction(prof_PH, P_AGL, H_AGL, d_PH);
fprintf('  worst-point obstruction above LOS chord: %+.2f m\n', los_PH);
fresnel_F1_mid_PH = fresnelZoneRadius(d_PH/2, d_PH/2, LAMBDA, 1);
fresnel_60_PH     = 0.6 * fresnel_F1_mid_PH;
fprintf('  first-Fresnel radius at midpoint: %.2f m (60%% bar = %.2f m)\n', ...
        fresnel_F1_mid_PH, fresnel_60_PH);
deficit_PH = los_PH + fresnel_60_PH;
if deficit_PH > 0
    fprintf('  60%% Fresnel bar NOT met — raise QE Hospital mast by %.1f m.\n', ...
            deficit_PH);
else
    fprintf('  60%% Fresnel bar CLEAR (margin %.1f m above the bar).\n', -deficit_PH);
end

lb_PH = linkBudget(P_LAT, P_LON, P_AGL, FREQ_DIR, TX_PWR_W, TX_GAIN, ...
                    H_LAT, H_LON, H_AGL,                  RX_GAIN, ...
                    MODEL_ITM, prof_PH, CLIM_MAR, TIME_Q, LOC_Q, SIT_Q);
fprintf('  Longley-Rice path loss      : %.2f dB\n', lb_PH.PathLoss);
fprintf('  Received power at QE Hospital: %.2f dBm\n', lb_PH.ReceivedPower);
fprintf('  Link margin                 : %.2f dB\n', lb_PH.LinkMargin);

% ============================================================
% Per-site 5G access bubbles
%
% 3.5 GHz n78 mid-band, 10 W (40 dBm) per sector, three 120-deg
% sectors per site, 14 dBi sector antennas.  Mobile RX at 1.5 m AGL
% with isotropic 0 dBi antenna.  Coverage threshold -85 dBm for
% 600 Mbit/s 5G NR.
% ============================================================
FREQ_5G   = 3.5e9;
PWR_5G    = 10;
GAIN_SEC  = 14;
BW_AZ     = 120;
BW_EL     = 12;
FB_dB     = 25;
NUM_GRID  = 48;
COV_THR   = -85;
AGG_BS    = 0;          % best-server

% Coverage box: 3 km on a side centred on the site.  Convert via
% small-angle to lat/lon range.
COV_HALF_DEG = 0.014;   % ~ 1.55 km north-south at this latitude

% --- Police ---
fprintf('\n=== 5G access bubble: Police Command Center (3 km, 3 sectors) ===\n');
% sites: [lat lon h_m P_W f_Hz n_ant]
sites_P = [P_LAT, P_LON, P_AGL, PWR_5G, FREQ_5G, 3];
% antennas: [code gain bw_az bw_el fb_or_n mount_az mount_tilt _]
ants_P = [1, GAIN_SEC, BW_AZ, BW_EL, FB_dB,   0, 5, 0;
          1, GAIN_SEC, BW_AZ, BW_EL, FB_dB, 120, 5, 0;
          1, GAIN_SEC, BW_AZ, BW_EL, FB_dB, 240, 5, 0];
grid_P = coverageGridMulti(sites_P, ants_P, heightmap, ...
                            P_LAT - COV_HALF_DEG, P_LAT + COV_HALF_DEG, ...
                            P_LON - COV_HALF_DEG, P_LON + COV_HALF_DEG, ...
                            NUM_GRID, NUM_GRID, 1.5, 0.0, ...
                            MODEL_ITM, AGG_BS, ...
                            CLIM_MAR, 50, 50, 50);
disp('  max RX power (dBm):');     disp(max(grid_P(:)));
disp('  median RX power (dBm):');  disp(median(grid_P(:)));
disp('  min RX power (dBm):');     disp(min(grid_P(:)));
covered_P = norm(sum(sum(grid_P > COV_THR))) / numel(grid_P) * 100.0;
fprintf('  cells above %.0f dBm 5G NR 600 Mbit threshold: %.1f %%\n', ...
        COV_THR, covered_P);

% --- Ilaro Court ---
fprintf('\n=== 5G access bubble: Ilaro Court (3 km, 3 sectors) ===\n');
sites_I = [I_LAT, I_LON, I_AGL, PWR_5G, FREQ_5G, 3];
ants_I = [1, GAIN_SEC, BW_AZ, BW_EL, FB_dB,   0, 5, 0;
          1, GAIN_SEC, BW_AZ, BW_EL, FB_dB, 120, 5, 0;
          1, GAIN_SEC, BW_AZ, BW_EL, FB_dB, 240, 5, 0];
grid_I = coverageGridMulti(sites_I, ants_I, heightmap, ...
                            I_LAT - COV_HALF_DEG, I_LAT + COV_HALF_DEG, ...
                            I_LON - COV_HALF_DEG, I_LON + COV_HALF_DEG, ...
                            NUM_GRID, NUM_GRID, 1.5, 0.0, ...
                            MODEL_ITM, AGG_BS, ...
                            CLIM_MAR, 50, 50, 50);
disp('  max RX power (dBm):');     disp(max(grid_I(:)));
disp('  median RX power (dBm):');  disp(median(grid_I(:)));
disp('  min RX power (dBm):');     disp(min(grid_I(:)));
covered_I = norm(sum(sum(grid_I > COV_THR))) / numel(grid_I) * 100.0;
fprintf('  cells above %.0f dBm 5G NR 600 Mbit threshold: %.1f %%\n', ...
        COV_THR, covered_I);

% --- QE Hospital: sector coverage for in-building reach ---
fprintf('\n=== 5G access bubble: QE Hospital (3 sectors, narrower bw_el for indoor focus) ===\n');
BW_EL_H = 8;
sites_H = [H_LAT, H_LON, H_AGL, PWR_5G, FREQ_5G, 3];
ants_H = [1, GAIN_SEC, BW_AZ, BW_EL_H, FB_dB,   0, 8, 0;
          1, GAIN_SEC, BW_AZ, BW_EL_H, FB_dB, 120, 8, 0;
          1, GAIN_SEC, BW_AZ, BW_EL_H, FB_dB, 240, 8, 0];
grid_H = coverageGridMulti(sites_H, ants_H, heightmap, ...
                            H_LAT - COV_HALF_DEG, H_LAT + COV_HALF_DEG, ...
                            H_LON - COV_HALF_DEG, H_LON + COV_HALF_DEG, ...
                            NUM_GRID, NUM_GRID, 1.5, 0.0, ...
                            MODEL_ITM, AGG_BS, ...
                            CLIM_MAR, 50, 50, 50);
disp('  max RX power (dBm):');     disp(max(grid_H(:)));
disp('  median RX power (dBm):');  disp(median(grid_H(:)));
disp('  min RX power (dBm):');     disp(min(grid_H(:)));
covered_H = norm(sum(sum(grid_H > COV_THR))) / numel(grid_H) * 100.0;
fprintf('  cells above %.0f dBm 5G NR 600 Mbit threshold: %.1f %%\n', ...
        COV_THR, covered_H);

fprintf('\nDone.\n');
