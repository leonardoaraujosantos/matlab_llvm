% Navigation Toolbox Tier-5 — controllerVFH reactive obstacle avoidance.
% Mirrors the "Obstacle Avoidance with VFH" workflow: a robot uses a lidar
% scan + a goal direction to pick a collision-free steering direction via a
% Vector Field Histogram.

vfh = controllerVFH();
vfh.NumAngularSectors = 180;
vfh.DistanceLimits = [0.05, 4.0];
vfh.RobotRadius = 0.25;
vfh.SafetyDistance = 0.2;

% 180-degree forward-facing lidar.
ang = (-pi/2 : 0.05 : pi/2)';
M = size(ang, 1);

fprintf('VFH reactive steering (target straight ahead = 0 rad):\n');

% Case 1 — clear field: drive straight at the goal.
r = 8 * ones(M, 1);
s1 = step(vfh, r, ang, 0.0);
fprintf('  clear field          -> steer %.2f rad\n', s1);

% Case 2 — wall dead ahead: must divert around it.
for k = 1:M
    if abs(ang(k)) < 0.4
        r(k) = 0.8;
    end
end
s2 = step(vfh, r, ang, 0.0);
fprintf('  obstacle dead-ahead  -> steer %.2f rad\n', s2);

% Case 3 — same wall, but the goal is off to the right: bias the opening.
s3 = step(vfh, r, ang, 0.7);
fprintf('  obstacle + goal-right-> steer %.2f rad\n', s3);
