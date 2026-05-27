% Navigation Tier-6 — referencePathFrenet + trajectoryGeneratorFrenet.
wp = [0 0; 10 0; 20 0];
rp = referencePathFrenet(wp);
% A point 2 m left of the straight path at x=10 -> s=10, d=2.
fr = global2frenet(rp, [10 2]);
fprintf('s=%.2f d=%.2f\n', fr(1), fr(2));
% Round-trip back to global.
g = frenet2global(rp, [10 2]);
fprintf('global=(%.2f,%.2f)\n', g(1), g(2));
% Lane-change trajectory: shift 3 m laterally over 12 m of travel.
tg = trajectoryGeneratorFrenet(rp);
traj = connect(tg, [0 0], [12 3], 4);
fprintf('traj rows=%.0f end=(%.2f,%.2f)\n', size(traj,1), traj(end,4), traj(end,5));
