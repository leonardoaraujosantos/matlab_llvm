% pde_multicylinder.m — Geometry primitives smoke test.
%
% Builds a voxelized cylinder + tests translate / rotate / scale.

% R = 0.5, H = 1.0, voxel size 0.15 -> ~7x7x7 cell grid -> a few
% hundred inside cells after the cylinder predicate.
mesh = pde_multicylinder(0.5, 1.0, 0.15);
nodes = pde_mesh_nodes(mesh);
faces = pde_mesh_faces(mesh);

% Translate by [1, 2, 3]; rotate 90 deg around y; scale by 2 in x.
mesh = pde_translate(mesh, 1.0, 2.0, 3.0);
mesh = pde_rotate(mesh, 2.0, 90.0);
mesh = pde_scale(mesh, 2.0, 1.0, 1.0);
nodes_after = pde_mesh_nodes(mesh);

% Sanity-check: nodes/faces count and AABB after transforms.
n_nodes = size(nodes, 1);
n_faces = size(faces, 1);

% Get the first node's coords post-transform to verify the affine ops.
x0 = nodes_after(1, 1);
y0 = nodes_after(1, 2);
z0 = nodes_after(1, 3);

% The first cell-corner node of the un-transformed cylinder is at
% roughly (-0.5, -0.5, 0).  After translate(1, 2, 3) -> (0.5, 1.5, 3).
% After rotate-y(90 deg) -> (3, 1.5, -0.5).
% After scale x by 2 -> (6, 1.5, -0.5).
fprintf('PDE geometry primitives:\n');
fprintf('  multicylinder nodes:   %.0f\n', n_nodes);
fprintf('  multicylinder faces:   %.0f\n', n_faces);
fprintf('  first node x post-xfm: %.1f\n', x0);
fprintf('  first node y post-xfm: %.1f\n', y0);
fprintf('  first node z post-xfm: %.1f\n', z0);
