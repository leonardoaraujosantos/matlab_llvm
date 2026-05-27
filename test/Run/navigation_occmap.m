% Navigation Tier-1 — occupancyMap: set / get / check / inflate + world limits.
map = occupancyMap(10, 10, 2);          % 10x10 m at 2 cells/m -> 20x20 grid
gs = map.GridSize;
fprintf('GridSize: %.0f x %.0f\n', gs(1), gs(2));
setOccupancy(map, [4 4], 1.0);
setOccupancy(map, [4 5], 1.0);
fprintf('occ(4,4)=%.1f free(1,1)=%.1f\n', getOccupancy(map, [4 4]), getOccupancy(map, [1 1]));
fprintf('check(4,4)=%.0f check(1,1)=%.0f\n', checkOccupancy(map, [4 4]), checkOccupancy(map, [1 1]));
inflate(map, 1.0);                       % grow obstacles by 1 m
fprintf('after inflate occ(3.5,4)=%.0f\n', checkOccupancy(map, [3.5 4]));
