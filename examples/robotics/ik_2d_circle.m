% Robotics — 2-D inverse-kinematics circle trace (the MathWorks
% "2-D Inverse Kinematics Example" workflow).  Build a 3-link planar arm
% with addBody, then solve inverseKinematics for each point on a circular
% path so the end-effector traces it.
%
% NOTE on API surface: the verbatim MathWorks example builds the robot with
% per-body objects (rigidBody/rigidBodyJoint/setFixedTransform) and calls the
% callable solver `ik(ee, trvec2tform(pt), weights, q0)`.  This compiler ships
% the equivalent capability through the packed addBody form + the
% matlab_robotics_ik_solve entry; the kinematics + IK math are identical.

robot = rigidBodyTree();
% Three revolute links, each 0.5 m, via the packed DH addBody form
% (dh = [a alpha d theta], joint type 1 = revolute, limits ±pi).
addBody(robot, [0.5, 0, 0, 0], 1, -pi, pi);
addBody(robot, [0.5, 0, 0, 0], 1, -pi, pi);
addBody(robot, [0.5, 0, 0, 0], 1, -pi, pi);
fprintf('Built %.0f-link planar arm.\n', robot.NumBodies);

ik = inverseKinematics(robot);

% Trace a circle of radius 0.3 centred at (0.9, 0) with 8 waypoints.
q = [0.1; 0.1; 0.1];
maxerr = 0.0;
fprintf('Tracing a circle with inverseKinematics:\n');
for k = 1:8
    ang = 2*pi*(k-1)/8;
    px = 0.9 + 0.3*cos(ang);
    py = 0.3*sin(ang);
    T = trvec2tform([px, py, 0.0]);
    res = matlab_robotics_ik_solve(ik, T, q, 1.0, 0.0);
    q = [res(1); res(2); res(3)];
    Tf = getTransform(robot, q);
    e = abs(Tf(4) - px) + abs(Tf(8) - py);
    if e > maxerr
        maxerr = e;
    end
    fprintf('  wp %d: target=(%.2f, %.2f)  EE=(%.3f, %.3f)\n', k, px, py, Tf(4), Tf(8));
end
fprintf('Max position error over the circle: %.6f\n', maxerr);
