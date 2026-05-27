% Navigation Tier-5 — monteCarloLocalization dead-reckoning under odometry.
rng(0);
map = occupancyMap(10, 10, 1);
setOccupancy(map, [5 5], 1.0);
mcl = monteCarloLocalization(map);
mcl.NumParticles = 800;
empty = zeros(0, 1);
p0 = step(mcl, [5 5 0], empty, empty);
for k = 1:4
    pose = step(mcl, [5+k, 5, 0], empty, empty);
end
fprintf('odometry-tracked x ~ 9: %.0f\n', round(pose(1)));
