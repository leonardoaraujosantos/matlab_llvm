% coverage_three_sector.m
% ====================================================================
% Multi-site coverage map with three 120-degree sectors per site.
% Exercises PROP-Tier-3 best-server aggregation across two sites.
% ====================================================================

% Geographic frame: 0.5 deg lat x 0.5 deg lon, flat terrain.
LAT_MIN = 13.0; LAT_MAX = 13.5;
LON_MIN = -59.7; LON_MAX = -59.2;

% Flat heightmap (sea-level everywhere).
heightmap = zeros(48, 48);

% Two sites, each with 3 sectors.
% sites: [lat lon h_m P_W f_Hz n_ant]
sites = [13.40, -59.55, 30, 5, 2.4e9, 3;
         13.10, -59.35, 30, 5, 2.4e9, 3];

% antennas: [code gain bw_az bw_el fb_or_n mount_az mount_tilt _]
% Three 120-deg sectors per site at 0/120/240 deg. Pattern code 1 = sector.
antennas = [1, 14, 120, 12, 25,   0, 5, 0;
            1, 14, 120, 12, 25, 120, 5, 0;
            1, 14, 120, 12, 25, 240, 5, 0;
            1, 14, 120, 12, 25,   0, 5, 0;
            1, 14, 120, 12, 25, 120, 5, 0;
            1, 14, 120, 12, 25, 240, 5, 0];

% Coverage grid: 48x48, best-server, FSPL model (no terrain effects).
MODEL_FSPL = 0;
AGG_BEST_SERVER = 0;
grid = coverageGridMulti(sites, antennas, heightmap, ...
                          LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                          48, 48, 1.5, 0.0, ...
                          MODEL_FSPL, AGG_BEST_SERVER, ...
                          5, 50, 50, 50);

disp('=== Two-site three-sector FSPL coverage (best-server) ===');
fprintf('Grid               : 48x48 cells\n');
disp('Max RX power (dBm) =');
disp(max(grid(:)));
disp('Median RX power (dBm) =');
disp(median(grid(:)));
disp('Min RX power (dBm) =');
disp(min(grid(:)));

% Same scenario with SINR aggregation.
AGG_SINR = 2;
grid_sinr = coverageGridMulti(sites, antennas, heightmap, ...
                               LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                               48, 48, 1.5, 0.0, ...
                               MODEL_FSPL, AGG_SINR, ...
                               5, 50, 50, 50);
disp(' ');
disp('=== Same scenario, SINR aggregation (dB) ===');
disp('Max SINR =');
disp(max(grid_sinr(:)));
disp('Median SINR =');
disp(median(grid_sinr(:)));
disp('Min SINR =');
disp(min(grid_sinr(:)));
