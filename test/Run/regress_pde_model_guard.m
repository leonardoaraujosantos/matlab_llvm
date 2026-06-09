% regress_pde_model_guard.m — a PDE entry point handed a value that is NOT a
% real PDE model must not segfault. This happens when a femodel construction
% fails or a classdef value loses its class on a REPL cross-turn round-trip
% (#28/#116): the runtime then receives a matlab_mat* (or stale garbage) where
% it expects a matlab_struct*. Before the fix matlab_pde_generate_mesh /
% matlab_pde_solve walked the matrix's data-pointer halves as the struct's
% nfields/names fields and crashed with signal 11 (observed on
% examples/pde/clamped_plate_pressure.m under -repl).
%
% The entry points now validate the struct header invariants and return an
% empty struct for an implausible "model", so execution continues gracefully.
% A plain matrix is the deterministic stand-in for the bad pointer; reaching
% the disp after each call proves no crash. (isempty consumes the result so
% the value is used — the emit-c/cpp strict lanes reject an unused temp.)

bad = zeros(3, 3);

disp(isempty(generateMesh(bad)));        % 1-arg -> matlab_pde_generate_mesh
disp('mesh1 ok');

disp(isempty(generateMesh(bad, Hmax=0.02)));  % name=value -> ..._generate_mesh_kw
disp('mesh2 ok');
