% Sensor Fusion Tier-1 — quaternion construction + conversions + algebra.
% Builds an identity quaternion, an Euler-angle quaternion (ZYX), and a
% rotation matrix; round-trips through quat2eul / quat2rotm and exercises
% rotatepoint via the rotation-matrix path.  Also exercises the T1.7 core
% gaps (cross / dot / deg2rad).
q1 = quaternion(1, 0, 0, 0);
disp(q1);

% Euler ZYX [yaw pitch roll] = [0.5 0 0] rad → quaternion via free function.
E  = [0.5, 0.0, 0.0];
qe = eul2quat(E);
% qe is a 1×4 numeric matrix.
fprintf('eul2quat w = %.4f\n', qe(1));
fprintf('eul2quat z = %.4f\n', qe(4));

% Round-trip: quat2eul(eul2quat(E)) ≈ E.
Eback = quat2eul(qe);
fprintf('roundtrip yaw err = %.6f\n', Eback(1) - E(1));

% Rotation matrix from the same quaternion: a yaw of 0.5 rad about Z.
R  = quat2rotm(qe);
fprintf('R(1,1) = %.4f\n', R(1));      % cos(0.5)
fprintf('R(2,1) = %.4f\n', R(4));      % sin(0.5)

% T1.7 core gaps.
c = cross([1, 0, 0], [0, 1, 0]);
fprintf('cross = [%g %g %g]\n', c(1), c(2), c(3));
d = dot([1, 2, 3], [4, 5, 6]);
fprintf('dot   = %g\n', d(1));
rd = deg2rad([180.0]);
fprintf('deg2rad(180) = %.4f\n', rd(1));
