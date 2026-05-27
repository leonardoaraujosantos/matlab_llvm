% Navigation Toolbox — "Rotations, Orientation, and Quaternions".
% Mirrors https://www.mathworks.com/help/nav/ug/rotations-orientation-and-quaternions.html
%
% Navigation reuses the Sensor Fusion `quaternion` value type and the
% orientation-conversion surface (eul2quat / quat2eul / quat2rotm / slerp).
% This shows the everyday orientation maths a navigation stack performs:
% build an orientation from yaw/pitch/roll, convert to a rotation matrix,
% rotate a body vector into the navigation frame, and interpolate between
% two orientations with slerp.

% A platform yawed 45 deg about +Z (NED: nose swung to the right).
yaw = 45 * pi/180;
q = quaternion(eul2quat([yaw, 0.0, 0.0]));
disp(q);

% Rotation matrix form — the upper-left 2x2 is the planar yaw block.
R = quat2rotm(q.Data);
fprintf('yaw block: [%.3f %.3f; %.3f %.3f]\n', R(1), R(2), R(4), R(5));

% Rotate the body x-axis (forward) into the navigation frame.
fwd_body = [1.0; 0.0; 0.0];
fwd_nav  = R * fwd_body;
fprintf('forward in nav frame = [%.3f %.3f %.3f]\n', fwd_nav(1), fwd_nav(2), fwd_nav(3));

% Interpolate halfway between level (identity) and the 90-deg-yaw pose.
q0 = quaternion(1, 0, 0, 0);
q90 = quaternion(eul2quat([pi/2, 0.0, 0.0]));
qmid = slerp(q0.Data, q90.Data, 0.5);
emid = quat2eul(qmid);
fprintf('slerp midpoint yaw = %.4f rad (expected %.4f)\n', emid(1), pi/4);

% Cross product gives the rotation axis between two heading vectors.
ax = cross([1, 0, 0], [0, 1, 0]);
fprintf('heading-change axis = [%g %g %g]\n', ax(1), ax(2), ax(3));
