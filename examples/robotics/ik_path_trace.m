% Robotics System Toolbox — headline tracer (Tiers 1–3).
% Solve inverseKinematics for each waypoint of a 2-D path so the end-effector
% of a 2-link planar arm traces it.  Closes the headline `ik_path_trace.m`
% per docs/robotics_toolbox_roadmap.md.

% Build the arm and IK solver.
arm = rigidBodyTree();
loadrobot(arm, 'planar2');
ik  = inverseKinematics(arm);

% 2-D path: a small horizontal segment in front of the arm.
xs = [1.4, 1.5, 1.6, 1.7];
ys = [0.4, 0.5, 0.6, 0.5];

fprintf('Tracing a 4-waypoint path with the planar 2-link arm:\n');

q_prev = [0.0; 0.0];
total_err = 0.0;
for k = 1:4
    Tgt = trvec2tform([xs(k), ys(k), 0.0]);
    res = matlab_robotics_ik_solve(ik, Tgt, q_prev, 1.0, 0.0);
    qsol = [res(1); res(2)];
    Tf = getTransform(arm, qsol);
    fprintf('  wp %d: target=(%.2f, %.2f)  EE=(%.3f, %.3f)  err=%.5f\n', ...
            k, xs(k), ys(k), Tf(4), Tf(8), res(5));
    total_err = total_err + res(5);
    q_prev = qsol;
end
fprintf('Sum of IK residuals over the 4 waypoints: %.5f\n', total_err);
