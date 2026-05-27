% Navigation Toolbox Tier-6 — Frenet-frame trajectory generation.
% Mirrors the "Highway Trajectory Planning Using Frenet Reference Path"
% workflow: build a reference path, then generate a smooth lane-change
% trajectory expressed in Frenet (arc-length s, lateral offset d) coordinates
% and convert it back to global (x,y).

% Reference path: a gently curving road.
wp = [0 0; 20 0; 40 8; 60 8];
rp = referencePathFrenet(wp);
fprintf('Reference path length = %.2f m\n', rp.PathLength);

% Where is a vehicle at global (22, 3) relative to the path?
fr = global2frenet(rp, [22 3]);
fprintf('vehicle at (22,3): s=%.2f  d=%.2f\n', fr(1), fr(2));

% Plan a lane change: from the centreline (d=0) to +3.5 m (one lane left),
% over 30 m of travel.
tg = trajectoryGeneratorFrenet(rp);
traj = connect(tg, [fr(1) 0], [fr(1)+30, 3.5], 4.0);
fprintf('trajectory: %d samples over %.1f s\n', size(traj,1), traj(end,1));
fprintf('  start global = (%.2f, %.2f)\n', traj(1,4), traj(1,5));
fprintf('  end   global = (%.2f, %.2f)\n', traj(end,4), traj(end,5));

% Confirm the terminal Frenet lateral offset reached the target lane.
fprintf('  end lateral d = %.2f m (target 3.5)\n', traj(end,3));
