% Robotics Tier-6 — GJK collision (orientation-aware) + manipulatorRRT.
s1 = collisionSphere(1.0);
s2 = collisionSphere(1.0);
c1 = checkCollision(s1, s2);
fprintf('Two coincident spheres collide: %.0f\n', c1(1));

% Move s2 away by setting its Pose; should now be separated.
s2.Pose = trvec2tform([3.0, 0.0, 0.0]);
c2 = checkCollision(s1, s2);
fprintf('Spheres 3m apart collide: %.0f\n', c2(1));
g = matlab_robotics_gjk_collision(s1, s2);
fprintf('GJK separation = %.2f m\n', g(2));

% Box vs cylinder vs capsule at the origin all overlap.
b   = collisionBox(1.0, 1.0, 1.0);
cyl = collisionCylinder(0.4, 2.0);
cap = collisionCapsule(0.3, 1.5);
bc = checkCollision(b, cyl);
bp = checkCollision(b, cap);
fprintf('box-cyl collide: %.0f\n', bc(1));
fprintf('box-cap collide: %.0f\n', bp(1));

% manipulatorRRT planning on the 2-link planar arm around a sphere obstacle.
arm = rigidBodyTree();
loadrobot(arm, 'planar2');
centers = [3.0, 3.0, 0.0];
radii   = [0.2];
rrt = manipulatorRRT(arm, centers, radii);
plan_path = plan(rrt, [0.0; 0.0], [0.3; -0.3]);
fprintf('RRT plan rows=%.0f cols=%.0f\n', size(plan_path, 1), size(plan_path, 2));
