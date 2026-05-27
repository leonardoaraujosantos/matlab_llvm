% Navigation Tier-2 — plannerAStarGrid around a wall (deterministic).
map = occupancyMap(10, 10, 1);
for y = 3:8
    setOccupancy(map, [5 y], 1.0);
end
ag = plannerAStarGrid(map);
p = plan(ag, [10 1], [1 10]);            % grid (row,col), 1-based
fprintf('astar cells=%.0f\n', size(p, 1));
fprintf('start=(%.0f,%.0f) goal=(%.0f,%.0f)\n', p(1,1), p(1,2), p(end,1), p(end,2));
