% Robotics Tier-2 + Tier-3 — load a 2-link planar arm, solve IK for a target
% end-effector pose, and verify the FK at the solution matches.
arm = rigidBodyTree();
loadrobot(arm, 'planar2');
fprintf('Loaded planar2 arm: %.0f bodies\n', arm.NumBodies);

% Home configuration.
qh = homeConfiguration(arm);
Th = getTransform(arm, qh);
fprintf('Home end-effector: [%.2f, %.2f]\n', Th(4), Th(8));

% Build an IK solver from the arm.
ik = inverseKinematics(arm);

% Target: end-effector at (1.5, 0.5) in the plane (reachable with link 1+1).
Tgt = trvec2tform([1.5, 0.5, 0.0]);
q0  = [0.1; 0.1];
res = matlab_robotics_ik_solve(ik, Tgt, q0, 1.0, 0.0);
N = arm.NumBodies;
fprintf('IK iters=%.0f exitflag=%.0f err=%.4f\n', res(3), res(4), res(5));
fprintf('IK q   = [%.3f, %.3f]\n', res(1), res(2));

% Verify FK at the solution.
qsol = [res(1); res(2)];
Tf = getTransform(arm, qsol);
fprintf('Final EE: [%.3f, %.3f]\n', Tf(4), Tf(8));
