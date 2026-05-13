% pde_rom_craig_bampton.m — Full Craig-Bampton ROM.
%
% Cantilever beam reduced via Craig-Bampton: master DOFs are all
% DOFs on face 6 (the +x end, the "interface" for substructure
% coupling); internal modes are the lowest 6 Lanczos modes of the
% K_ss-block (slave block with master DOFs fixed).
%
% The combined Ritz basis T has size NumDOFs × (n_master +
% n_internal).

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 8, 2, 2);

model = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed(model, 5);       % clamp x = 0
model = pde_set_interface_face(model, 6);   % interface at x = L
model = pde_set_num_modes(model, 6);
model = pde_generate_mesh(model);

Rred = pde_reduce_craig_bampton(model);

fprintf('PDE Tier-4 Craig-Bampton ROM:\n');
fprintf('  num master DOFs:      %.0f\n', Rred.nMaster);
fprintf('  num internal modes:   %.0f\n', Rred.nInternal);
fprintf('  num full DOFs:        %.0f\n', Rred.NumDOFs);
