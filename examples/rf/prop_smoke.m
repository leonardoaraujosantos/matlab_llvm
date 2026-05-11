% prop_smoke.m — minimum end-to-end PROP smoke test.
%
% Walks through every PROP-Tier-1a entry plus a tiny coverage grid
% just to confirm the dispatch table wiring is live.

freq  = 2.4e9;          % 2.4 GHz
d     = 1000.0;         % 1 km
lam   = 2.998e8 / freq;

% Free-space loss
L_fs = fspl(d, freq);
disp(L_fs);

% Hata urban-large (env=1)
L_h = pathlossHata(freq*1e-6, 30, 1.5, d*1e-3, 1.0);
disp(L_h);

% Fresnel zone radius mid-path
r1 = fresnelZoneRadius(d/2, d/2, lam, 1.0);
disp(r1);

% Haversine between two Barbados landmarks
% Bridgetown ↔ Speightstown ≈ 18 km
d_bb = haversine(13.1132, -59.5988, 13.2520, -59.6447);
disp(d_bb);

% Sector pattern gain (0° az, 0° el, peak 17 dBi, 65° az bw, 10° el bw)
g = sectorPattern(0.0, 0.0, 65.0, 10.0, 17.0, 25.0);
disp(g);

% Mount-to-local rotation
ml = applyMountOrientation(120.0, 5.0, 60.0, 2.0);
disp(ml);
