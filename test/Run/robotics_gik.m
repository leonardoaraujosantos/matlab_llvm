% Robotics Tier-3 follow-on — generalizedInverseKinematics (multi-constraint).
% Solve for a config of the 2-link planar arm that satisfies a position
% target + an orientation target as two separate weighted constraints.
arm = rigidBodyTree();
loadrobot(arm, 'planar2');
gik = generalizedInverseKinematics(arm);

% Constraint 1: end-effector position at (1.5, 0.5, 0).
pos = constraintPositionTarget([1.5, 0.5, 0.0]);

% Constraint 2: orientation = identity, weighted low so position dominates
% (a 2-link planar arm cannot satisfy an arbitrary position AND orientation).
ori = constraintOrientationTarget(eye(3), [0.02, 0.02, 0.02, 0, 0, 0]);

q0 = [0.1; 0.1];
res = matlab_robotics_gik_solve(gik, q0, pos, ori);
fprintf('gIK iters=%.0f flag=%.0f err=%.4f\n', res(3), res(4), res(5));
fprintf('gIK q = [%.3f, %.3f]\n', res(1), res(2));

% Verify FK at the solution reaches the position target.
qsol = [res(1); res(2)];
T = getTransform(arm, qsol);
fprintf('EE pos = [%.3f, %.3f]  (target 1.5, 0.5)\n', T(4), T(8));
fprintf('pos err = %.4f\n', abs(T(4) - 1.5) + abs(T(8) - 0.5));
