% Robotics Tier-1 — coordinate transformations + tform conversions.
T = trvec2tform([1.0, 2.0, 3.0]);
fprintf('T(1,4)=%.2f T(2,4)=%.2f T(3,4)=%.2f\n', T(4), T(8), T(12));

E = [0.5, 0.0, 0.0];
R = eul2tform(E);
fprintf('R yaw block: [%.3f %.3f]\n', R(1), R(2));
fprintf('             [%.3f %.3f]\n', R(5), R(6));

% Round-trip Euler → matrix → Euler.
Eback = tform2eul(R);
fprintf('eul roundtrip yaw err = %.6f\n', Eback(1) - E(1));

% Compose, then invert.
Tc = trvec2tform([1, 0, 0]);
Ti = matlab_robotics_tform_inv(Tc);
fprintf('inv T pos = [%.2f, %.2f, %.2f]\n', Ti(4), Ti(8), Ti(12));

% Apply to a 1×3 point.
p  = [0.0, 0.0, 0.0];
p2 = homtrans(Tc, p);
fprintf('homtrans p2 = [%.2f, %.2f, %.2f]\n', p2(1), p2(2), p2(3));

% wrapToPi.
a = wrapToPi([3.5, -3.5]);
fprintf('wrapToPi[3.5,-3.5] = [%.2f, %.2f]\n', a(1), a(2));

% vecnorm of a 3-vector.
n = vecnorm([3, 4, 0]);
fprintf('vecnorm[3,4,0] = %.2f\n', n(1));
