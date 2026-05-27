% Sensor Fusion Tier-5 — global-nearest-neighbour multi-object tracking.
% Three aircraft on parallel northbound headings emit noisy 2-D detections
% over 40 timesteps.  A `trackerGNN` over constvel trackingEKF filters
% confirms all three tracks using Mahalanobis gating + Munkres assignment.
%
% Closes the Tier-5 headline `gnn_air_traffic.m` per
% docs/sensor_fusion_toolbox_roadmap.md.

dt   = 1.0;
nT   = 40;
trk  = trackerGNN(16);

% Three aircraft starting positions (x, y) and constant heading speed.
p1x = 0.0;   p1y =  0.0;
p2x = 0.0;   p2y = 30.0;
p3x = 0.0;   p3y = 60.0;
v   = 1.0;       % m/s

sd  = 123;
nsteps = nT;
fprintf('Air-traffic GNN tracking: %d steps, 3 targets at y=0/30/60\n', nsteps);

for k = 1:nsteps
    p1x = p1x + v * dt;
    p2x = p2x + v * dt;
    p3x = p3x + v * dt;
    sd = mod(sd*1103515245 + 12345, 2147483648);
    nz1 = (sd/2147483648 - 0.5) * 1.2;
    sd = mod(sd*1103515245 + 12345, 2147483648);
    nz2 = (sd/2147483648 - 0.5) * 1.2;
    sd = mod(sd*1103515245 + 12345, 2147483648);
    nz3 = (sd/2147483648 - 0.5) * 1.2;
    det = [p1x + nz1, p1y + nz1*0.5;
           p2x + nz2, p2y + nz2*0.5;
           p3x + nz3, p3y + nz3*0.5];
    step(trk, det, dt);
end

nc = numConfirmed(trk);
fprintf('Confirmed tracks at end: %.0f / 3\n', nc(1));

% Inspect each confirmed track's state.
S = trk.States;
fprintf('Final track positions (x, y):\n');
fprintf('  track 1: (%.1f, %.1f)\n', S(1), S(3));
fprintf('  track 2: (%.1f, %.1f)\n', S(5), S(7));
fprintf('  track 3: (%.1f, %.1f)\n', S(9), S(11));
