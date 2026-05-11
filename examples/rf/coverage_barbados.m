% coverage_barbados.m
% ============================================================
% Barbados — point-to-point + coverage map with two directional
% antennas, propagating with the Longley-Rice (ITM) model.
%
% Scenario:
%   - Site A : Mount Hillaby (the highest point of Barbados, 343 m,
%              roughly 13.2156° N, 59.5747° W).
%   - Site B : Bridgetown waterfront, 50 m mast.
%   - Each site mounts a 22 dBi 8°-beamwidth directional dish, aimed
%     across the island to its peer (≈ az 200° from Hillaby toward
%     Bridgetown; ≈ az 20° back).
%   - We then compute a coverage map of the link from Hillaby's
%     antenna over the western half of Barbados.
%
% Outputs to stdout:
%   - Link distance + bearing (PtP geometry).
%   - Longley-Rice median path loss + Fresnel clearance + LOS flag.
%   - Received power at Bridgetown.
%   - Summary statistics of the 64x64 coverage grid (best-server cell
%     received-power, in dBm).
%
% Notes:
%   - All propagation primitives are functions; no Site Viewer.
%   - The heightmap below is a synthetic-but-Barbados-shaped 64x64
%     DEM. The MATLAB-canonical workflow would `load('srtm.mat')`
%     instead; we keep the example hermetic.
%
%   ENV  = 1  (urban-large for Hata only; ITM ignores the tag)
%   POL  = 1  (vertical)
%   CLIM = 3  (maritime subtropical — closest to Barbados)
%
% ============================================================

LAT_MIN = 13.05;
LAT_MAX = 13.35;
LON_MIN = -59.70;
LON_MAX = -59.40;

% Synthesise a 64x64 heightmap with a Mount-Hillaby-like ridge in the
% NE and lowlands sloping down to the western coast (vectorised).
NLAT = 64;
NLON = 64;
lat_v = linspace(LAT_MIN, LAT_MAX, NLAT);
lon_v = linspace(LON_MIN, LON_MAX, NLON);
[LON, LAT] = meshgrid(lon_v, lat_v);
% Mount-Hillaby ridge centred at 13.22 N, 59.575 W.
DX  = (LAT - 13.22) / 0.07;
DY  = (LON + 59.575) / 0.07;
R2  = DX.*DX + DY.*DY;
H_ridge = 340 * exp(-R2);
% Secondary spine sloping west toward Bridgetown.
DX2 = (LAT - 13.18) / 0.20;
DY2 = (LON + 59.520) / 0.15;
H_spine = 90 * exp(-(DX2.*DX2 + DY2.*DY2));
heightmap = max(H_ridge, H_spine);

% ---------- Sites ----------
SITE_A_LAT = 13.2156; SITE_A_LON = -59.5747;
SITE_B_LAT = 13.0975; SITE_B_LON = -59.6133;
SITE_A_H = 30;    % 30 m mast on top of the ridge
SITE_B_H = 50;    % 50 m mast on the Bridgetown waterfront
SITE_A_PW = 5;    % 5 W = ~37 dBm
SITE_B_PW = 5;
FREQ = 5.8e9;     % 5.8 GHz unlicensed PtP band

% Directional dish: 22 dBi, 8° half-beamwidth — cosine pattern.
TX_GAIN_A = 22;   TX_GAIN_B = 22;
RX_GAIN   = 22;   % both ends have matching dishes

% ---------- PtP geometry ----------
d_m = haversine(SITE_A_LAT, SITE_A_LON, SITE_B_LAT, SITE_B_LON);
az_AB = bearing (SITE_A_LAT, SITE_A_LON, SITE_B_LAT, SITE_B_LON);
az_BA = bearing (SITE_B_LAT, SITE_B_LON, SITE_A_LAT, SITE_A_LON);

disp('--- Point-to-point geometry ---');
fprintf('Link distance     : %.2f km\n', d_m / 1000.0);
fprintf('A-to-B compass bearing: %.2f deg\n', az_AB);
fprintf('B-to-A compass bearing: %.2f deg\n', az_BA);

% ---------- Terrain profile + LOS check ----------
N_PROF = 128;
profile = terrainProfile(heightmap, ...
                          LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                          SITE_A_LAT, SITE_A_LON, ...
                          SITE_B_LAT, SITE_B_LON, N_PROF);
los_clr  = losClear      (profile, SITE_A_H, SITE_B_H, d_m);
los_blk  = losObstruction(profile, SITE_A_H, SITE_B_H, d_m);

% ---------- Longley-Rice path loss + link budget ----------
% Model 7 = ITM/Longley-Rice. Climate 3 = maritime subtropical.
% Reliability triple (50, 50, 50) = long-term median; (80, 99, 99)
% would give TSB-10F microwave-link conservatism.
MODEL_ITM = 7;
CLIM_MAR  = 3;
TIME_Q = 80; LOC_Q = 99; SIT_Q = 99;

% Aim each antenna directly at its peer (compass az).
% Mount tilt small (links are close to horizontal).
% Pattern code 2 = cosinePattern; we go through applyMountAz/El to
% exercise the rotation wiring (both end up at the boresight).
az_local_A = applyMountAz(az_AB, 0.0, az_AB, 0.0);
el_local_A = applyMountEl(az_AB, 0.0, az_AB, 0.0);
Gtx_align  = cosinePattern(az_local_A, el_local_A, 8, 8, TX_GAIN_A, 30);
az_local_B = applyMountAz(az_BA, 0.0, az_BA, 0.0);
el_local_B = applyMountEl(az_BA, 0.0, az_BA, 0.0);
Grx_align  = cosinePattern(az_local_B, el_local_B, 8, 8, RX_GAIN,   30);

lb = linkBudget( ...
   SITE_A_LAT, SITE_A_LON, SITE_A_H, FREQ, SITE_A_PW, Gtx_align, ...
   SITE_B_LAT, SITE_B_LON, SITE_B_H,                 Grx_align, ...
   MODEL_ITM, profile, CLIM_MAR, TIME_Q, LOC_Q, SIT_Q);

disp(' ');
disp('--- Longley-Rice link budget (80/99/99 reliability) ---');
fprintf('Path loss              : %.2f dB\n', lb.PathLoss);
fprintf('TX power               : %.2f dBm\n', lb.TxPower_dBm);
fprintf('TX antenna gain        : %.2f dBi\n', Gtx_align);
fprintf('RX antenna gain        : %.2f dBi\n', Grx_align);
fprintf('Received power at B    : %.2f dBm\n', lb.ReceivedPower);
fprintf('Thermal noise floor    : %.2f dBm (kT*1 MHz)\n', lb.NoiseFloor);
fprintf('SNR                    : %.2f dB\n', lb.Snr);
fprintf('Link margin (10 dB thr): %.2f dB\n', lb.LinkMargin);
fprintf('Fresnel-zone clearance : %.1f %% (>60%% = TIA-clean)\n', ...
        lb.FresnelClearance);
fprintf('LOS clear              : %.0f\n', lb.LosClear);
if los_blk > 0
  fprintf('Worst terrain obstr.  : +%.1f m above LOS\n', los_blk);
else
  fprintf('No terrain obstruction (peak %+.1f m below LOS).\n', los_blk);
end

% ---------- Coverage map from Site A's antenna ----------
%
% We build a single-site descriptor for coverageGridMulti, including
% the directional pattern. coverage_grid_multi handles best-server
% / sum-power / SINR aggregation; with one site, best-server is
% simply that site's received power.

% sites: [num_sites x 6] = [lat lon h_m P_W f_Hz n_ant]
sites = [SITE_A_LAT, SITE_A_LON, SITE_A_H, SITE_A_PW, FREQ, 1];
% antennas: [sum(n_ant) x 8] = [code gain bw_az bw_el fb_or_n mount_az mount_tilt _]
% code 2 = cosinePattern, fb_or_n = 30 (cosine-power).
antennas = [2, TX_GAIN_A, 8, 8, 30, az_AB, 0.0, 0];

NLAT_GRID = 48;
NLON_GRID = 48;
RX_H = 1.5;     % handheld receiver height
RX_G = 0;       % isotropic mobile RX

AGG_BEST_SERVER = 0;
grid = coverageGridMulti(sites, antennas, heightmap, ...
                          LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                          NLAT_GRID, NLON_GRID, RX_H, RX_G, ...
                          MODEL_ITM, AGG_BEST_SERVER, ...
                          CLIM_MAR, 50, 50, 50);

% Aggregate stats — the runtime returned [NLAT_GRID x NLON_GRID]
% of received power in dBm. max/min/median of a matrix go through the
% existing reduce kernels (they return 1x1 matrices); we display them
% via disp() so the matrix print path takes care of the formatting.
P_max = max(grid(:));
P_min = min(grid(:));
P_med = median(grid(:));

disp(' ');
disp('--- Coverage map (Site A, best-server, ITM) ---');
fprintf('Grid size              : %.0fx%.0f cells\n', NLAT_GRID, NLON_GRID);
disp('Best-server max RX power (dBm) =');
disp(P_max);
disp('Best-server median RX power (dBm) =');
disp(P_med);
disp('Best-server min RX power (dBm) =');
disp(P_min);

disp(' ');
disp('Done.');
