% Robotics Tier-4 follow-on — full rigid-body dynamics (RNEA / CRBA /
% forwardDynamics), verifying the forward/inverse dynamics round-trip at
% machine precision.  (URDF import via importrobot is exercised by the
% robotics_urdf gating test; here we use the baked planar arm so the example
% is self-contained.)

arm = rigidBodyTree();
loadrobot(arm, 'planar2');
fprintf('Loaded %0.f-body arm.\n', arm.NumBodies);

q   = [0.4; -0.6];
qd  = [0.3;  0.1];
qdd = [0.8; -0.2];

% Inverse dynamics (recursive Newton-Euler).
tau = inverseDynamics(arm, q, qd, qdd);
fprintf('Required torque tau = [%.4f, %.4f]\n', tau(1), tau(2));

% Joint-space inertia matrix (composite-rigid-body) — symmetric.
M = massMatrix(arm, q);
fprintf('Mass matrix M = [%.4f %.4f; %.4f %.4f]\n', M(1), M(2), M(3), M(4));

% Forward dynamics must recover qdd from tau.
qdd_rec = forwardDynamics(arm, q, qd, tau);
fprintf('FD/ID round-trip error = %.2e\n', ...
        abs(qdd_rec(1) - qdd(1)) + abs(qdd_rec(2) - qdd(2)));

% Coriolis/centrifugal vector + center of mass.
vp = velocityProduct(arm, q, qd);
com = centerOfMass(arm, q);
fprintf('velocityProduct = [%.4f, %.4f]\n', vp(1), vp(2));
fprintf('centerOfMass = [%.3f, %.3f, %.3f]\n', com(1), com(2), com(3));
