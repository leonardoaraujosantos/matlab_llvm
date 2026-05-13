% stl_load_render.m — Save a mesh as STL, load it back, render with
% pdeplot3D coloured by node height.  Demonstrates the round-trip
% geometry import + visualisation loop the headline §10.2 + §10.4
% roadmap entries unlock.

% --- Step 1: build a mesh in MATLAB and save it as binary STL. ----
src = pde_mesh_cuboid_tet(0.4, 0.4, 0.1, 4, 4, 2);
ok  = pde_save_stl(src, "/tmp/pde_demo.stl");
fprintf('Wrote STL: ok=%.0f\n', ok);

% --- Step 2: load the STL back as a surface fegeometry. -----------
mesh  = pde_load_stl("/tmp/pde_demo.stl");
nodes = pde_mesh_nodes(mesh);
faces = pde_mesh_faces(mesh);

n_nodes = size(nodes, 1);
n_faces = size(faces, 1);
fprintf('Loaded STL: %.0f nodes / %.0f faces (round-tripped)\n', ...
        n_nodes, n_faces);

% --- Step 3: colour by node height (z), render to PNG. -----------
heights = zeros(n_nodes, 1);
for i = 1:n_nodes
    heights(i) = nodes(i, 3);
end

pdeplot3d(nodes, faces, heights);
title('STL load + pdeplot3D coloured by height');
saveas(gcf, '/tmp/pde_stl_demo.png');
fprintf('Rendered to /tmp/pde_stl_demo.png\n');
