% Sensor Fusion Tier-1 — quaternion + orientation/rotation math demo.
% Builds quaternions from Euler angles, rotates a point, interpolates
% with slerp, and round-trips through the matrix/Euler conversions.

% Start with two simple orientations:
%   q1: identity (no rotation)
%   q2: 30-deg yaw about +Z
q1 = quaternion(1, 0, 0, 0);

E2 = [30 * pi/180, 0.0, 0.0];        % yaw 30°
q2_data = eul2quat(E2);
q2 = quaternion(q2_data);
disp(q1);
disp(q2);

% Convert q2 to a rotation matrix.
R2 = quat2rotm(q2_data);
fprintf('R2 yaw block:\n');
fprintf('  [%.3f %.3f]\n', R2(1), R2(2));
fprintf('  [%.3f %.3f]\n', R2(4), R2(5));

% Rotate the body-frame x-axis to the nav frame via the rotation-matrix
% form (the function-form `rotatepoint(q, v)` on a quaternion object goes
% through the same kernel — the obj-arg overload is a Tier-1 follow-on).
v  = [1.0; 0.0; 0.0];
vp = R2 * v;
fprintf('rotated v = [%.3f %.3f %.3f]\n', vp(1), vp(2), vp(3));

% Spherical-linear interpolation halfway between q1 and q2.
qhalf = slerp(q1.Data, q2.Data, 0.5);
Eh = quat2eul(qhalf);
fprintf('slerp midpoint yaw = %.3f rad (expected %.3f)\n', Eh(1), 0.5 * 30 * pi/180);

% T1.7 core gaps — cross, dot, deg2rad — used in everyday orientation math.
n = cross([1, 0, 0], [0, 1, 0]);
fprintf('cross([1 0 0],[0 1 0]) = [%g %g %g]\n', n(1), n(2), n(3));
d = dot([1, 2, 3], [4, 5, 6]);
fprintf('dot([1 2 3],[4 5 6]) = %g\n', d(1));
r = deg2rad([180.0]);
fprintf('deg2rad(180) = %.4f rad (pi)\n', r(1));
