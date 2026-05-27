% Navigation Toolbox Tier-5 — monteCarloLocalization on an occupancyMap.
% Mirrors the "Localize TurtleBot Using Monte Carlo Localization" workflow:
% a particle filter seeded around the start pose tracks the robot as it drives,
% propagating particles by the odometry motion model and (when a scan is
% supplied) weighting them by a likelihood field over the occupied cells.

map = occupancyMap(12, 12, 1);
% A few walls so the map has structure for the likelihood field.
for y = 2:10
    setOccupancy(map, [3 y], 1.0);
end
for x = 3:9
    setOccupancy(map, [x 10], 1.0);
end

mcl = monteCarloLocalization(map);
mcl.NumParticles = 1000;

% Ground-truth path: drive +x from (5,5) to (9,5).
empty = zeros(0, 1);
truth0 = [5 5 0];
pose = step(mcl, truth0, empty, empty);        % seed cloud at the start
fprintf('MCL: %d particles, start (%.1f,%.1f)\n', mcl.NumParticles, pose(1), pose(2));

for k = 1:4
    truex = 5 + k;
    pose = step(mcl, [truex, 5, 0], empty, empty);
    fprintf('  step %d: true x=%.1f  est=(%.2f,%.2f)\n', k, truex, pose(1), pose(2));
end

fprintf('Final localization error = %.2f m\n', abs(pose(1) - 9));
