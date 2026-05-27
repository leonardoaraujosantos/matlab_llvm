% Robotics Tier-6 — collision primitives + manipulatorRRT planning.
s1 = collisionSphere(1.0);
s2 = collisionSphere(1.0);
c1 = checkCollision(s1, s2);
fprintf('Two coincident spheres collide: %.0f\n', c1(1));

b = collisionBox(1.0, 1.0, 1.0);
sph = collisionSphere(0.4);
c2 = checkCollision(b, sph);
fprintf('Box at origin vs sphere at origin collide: %.0f\n', c2(1));

% manipulatorRRT planning on the 2-link planar arm with one sphere obstacle
% well off the start/goal segment, so a direct connect should work.
arm = rigidBodyTree();
loadrobot(arm, 'planar2');
centers = [3.0, 3.0, 0.0];
radii   = [0.2];
rrt = manipulatorRRT(arm, centers, radii);

q_start = [0.0; 0.0];
q_goal  = [0.3; -0.3];
plan_path = plan(rrt, q_start, q_goal);
nr = size(plan_path, 1);
nc = size(plan_path, 2);
fprintf('RRT plan rows=%.0f cols=%.0f\n', nr, nc);
