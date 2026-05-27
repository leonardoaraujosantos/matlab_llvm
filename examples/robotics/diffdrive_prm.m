% Robotics Tier-5 — mobile-robot tracer.  Plan a path on an occupancy map
% with a wall in the middle, then drive a differential-drive robot toward
% it via pure-pursuit.  Closes the Tier-5 headline `diffdrive_prm.m`.

map = binaryOccupancyMap(20, 20, 1.0);
for k = 1:11
    yy = 4 + (k - 1);
    setOccupancy(map, [10.0, yy], 1.0);
end
fprintf('Wall placed at x=10, y=4..14 (occupied cells).\n');

prm = mobileRobotPRM(map, 200, 4.0);
fprintf('PRM built with %.0f nodes.\n', prm.NumNodes);

path = findpath(prm, [2.0, 2.0], [18.0, 18.0]);
nrows = size(path, 1);
fprintf('Planned path has %.0f waypoints.\n', nrows);

pp = controllerPurePursuit(path, 1.5, 0.6);
dd = differentialDriveKinematics(0.1, 0.5);

px = 2.0; py = 2.0; pth = 0.0;
for k = 1:5
    pose = [px; py; pth];
    cmd  = step(pp, pose);
    dxy  = matlab_robotics_diffdrive_derivative(dd, pose, cmd);
    px  = px  + 0.5 * dxy(1);
    py  = py  + 0.5 * dxy(2);
    pth = pth + 0.5 * dxy(3);
    fprintf('  tick %d: pose=(%.2f, %.2f, %.2f)  v=%.2f w=%.2f\n', ...
            k, px, py, pth, cmd(1), cmd(2));
end
