% Robotics Tier-2 follow-on — URDF importrobot.
% Parse a 2-joint URDF arm, then run FK + dynamics on the imported tree.
arm = rigidBodyTree();
importrobot(arm, 'data/twolink.urdf');
fprintf('Imported URDF: %.0f bodies\n', arm.NumBodies);

% FK at the home configuration: link lengths sum to 2.0 along +x.
qh = homeConfiguration(arm);
Th = getTransform(arm, qh);
fprintf('Home EE: [%.3f, %.3f]\n', Th(4), Th(8));

% FK at a bent configuration.
q = [pi/2; 0.0];
Tb = getTransform(arm, q);
fprintf('Bent EE: [%.3f, %.3f]\n', Tb(4), Tb(8));

% Dynamics on the imported tree (mass/inertia came from <inertial>).
M = massMatrix(arm, qh);
fprintf('M(1,1)=%.4f symmetric_err=%.2e\n', M(1), abs(M(2) - M(3)));

tau = inverseDynamics(arm, [0.2; 0.1], [0.0; 0.0], [1.0; 0.0]);
qdd = forwardDynamics(arm, [0.2; 0.1], [0.0; 0.0], tau);
fprintf('FD/ID roundtrip err = %.2e\n', abs(qdd(1) - 1.0) + abs(qdd(2)));
