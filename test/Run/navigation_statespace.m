% Navigation Tier-1 — stateSpaceSE2 distance / interpolate + validator.
ss = stateSpaceSE2([0 10; 0 10; -pi pi]);
fprintf('dist=%.3f\n', distance(ss, [0 0 0], [3 4 0]));
seg = interpolate(ss, [0 0 0], [2 0 0], [0; 0.5; 1]);
fprintf('mid=(%.2f,%.2f)\n', seg(2,1), seg(2,2));
map = occupancyMap(10, 10, 1);
setOccupancy(map, [5 5], 1.0);
sv = validatorOccupancyMap(ss, map);
fprintf('valid(2,2)=%.0f valid(5,5)=%.0f\n', isStateValid(sv, [2 2 0]), isStateValid(sv, [5 5 0]));
fprintf('motion(1,1->9,9)=%.0f\n', isMotionValid(sv, [1 1 0], [9 9 0]));
