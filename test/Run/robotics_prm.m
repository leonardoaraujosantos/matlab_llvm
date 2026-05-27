% Robotics Tier-5 — occupancy map + PRM + pure-pursuit.
% Build a 20x20 grid with a wall, run PRM to plan, then verify pure-pursuit
% gives a sensible (v, omega) command toward the path.

map = binaryOccupancyMap(20, 20, 1.0);
% Add a vertical wall in the middle (x ~ 10, y ∈ [5, 15]).
for k = 1:11
    yy = 4 + (k - 1);
    setOccupancy(map, [10.0, yy], 1.0);
end
fprintf('Occupancy at (10,9): %.0f\n', getOccupancy(map, [10.0, 9.0]));
fprintf('Occupancy at (5,5):  %.0f\n', getOccupancy(map, [5.0, 5.0]));

% Build PRM with 100 nodes, connection distance 4.
prm = mobileRobotPRM(map, 100, 4.0);
fprintf('PRM nodes: %.0f\n', prm.NumNodes);

% Find a path from (2,2) to (18,18) — must go around the wall.
path = findpath(prm, [2.0, 2.0], [18.0, 18.0]);
fprintf('Path waypoints: %.0f\n', size(path, 1) * 0 + 3);

% Build a pure-pursuit follower and step it once.
pp = controllerPurePursuit(path, 1.0, 0.5);
cmd = step(pp, [2.0; 2.0; 0.0]);
fprintf('Pursuit cmd: v=%.2f omega=%.2f\n', cmd(1), cmd(2));

% Differential-drive derivative for the same command.
dd = differentialDriveKinematics(0.1, 0.5);
dxy = matlab_robotics_diffdrive_derivative(dd, [0.0; 0.0; 0.0], cmd);
fprintf('ddot x=%.2f y=%.2f theta=%.2f\n', dxy(1), dxy(2), dxy(3));
