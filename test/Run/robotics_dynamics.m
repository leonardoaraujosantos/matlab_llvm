% Robotics Tier-4 — full RNEA / CRBA / forwardDynamics on the 2-link arm.
arm = rigidBodyTree();
loadrobot(arm, 'planar2');

q   = [0.3; -0.4];
qd  = [0.5;  0.2];
qdd = [1.0; -0.5];

% Inverse dynamics: required torque for (q, qd, qdd).
tau = inverseDynamics(arm, q, qd, qdd);
fprintf('tau = [%.4f, %.4f]\n', tau(1), tau(2));

% Mass matrix must be symmetric.
M = massMatrix(arm, q);
fprintf('M = [%.4f %.4f; %.4f %.4f]\n', M(1), M(2), M(3), M(4));
fprintf('M symmetric err = %.2e\n', abs(M(2) - M(3)));

% forwardDynamics must invert inverseDynamics: given tau from ID, recover qdd.
qdd_rec = forwardDynamics(arm, q, qd, tau);
fprintf('qdd recover err = %.2e\n', abs(qdd_rec(1) - qdd(1)) + abs(qdd_rec(2) - qdd(2)));

% Gravity torque at rest.
g = gravityTorque(arm, q);
fprintf('gravityTorque = [%.4f, %.4f]\n', g(1), g(2));

% Center of mass in the base frame.
com = centerOfMass(arm, q);
fprintf('COM = [%.4f, %.4f, %.4f]\n', com(1), com(2), com(3));
