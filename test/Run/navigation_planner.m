% Navigation Tier-2 — plannerRRT / plannerRRTStar on an open map.
% Randomised internally; assert only the deterministic exit flag (goal reached).
rng(0);
map = occupancyMap(20, 20, 1);
ss = stateSpaceSE2([0 20; 0 20; -pi pi]);
sv = validatorOccupancyMap(ss, map);
planner = plannerRRT(ss, sv);
planner.MaxConnectionDistance = 4;
planner.MaxIterations = 8000;
res = plan(planner, [2 2 0], [18 18 0]);
fprintf('RRT goalReached=%.0f\n', res(1,2));
star = plannerRRTStar(ss, sv);
star.MaxConnectionDistance = 4;
star.MaxIterations = 8000;
res2 = plan(star, [2 2 0], [18 18 0]);
fprintf('RRTStar goalReached=%.0f\n', res2(1,2));
