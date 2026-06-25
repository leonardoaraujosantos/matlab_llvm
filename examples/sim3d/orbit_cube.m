% orbit_cube.m — Simulink-3D-Animation-style command-line demo.
%
% Builds a 3-D world with a single cube actor, drives it around a circular
% orbit by setting its Translation each step, and exports a self-contained
% Babylon.js HTML player. Run with the interpreted REPL or as a compiled
% script — both produce byte-identical output:
%
%     matlabc -repl < orbit_cube.m
%     # then open orbit_cube.html in a browser
%
% This is the command-line (sim3d.*) counterpart of the block-diagram
% examples/mflowlink/3d/orbit_cube.mflow; it renders through the same viewer.

w = sim3d.World();

a = sim3d.Actor('cube', 'box');
a.Color = [0.2 0.6 1.0];
w.add(a);

w.open();
for k = 1:60
    th = k * 0.1;
    a.Translation = [3*cos(th) 3*sin(th) 1.0];
    a.Rotation = [0 0 th];
    w.run(0.05);
end
w.close();

sim3d.export(w, 'orbit_cube.html');
disp('wrote orbit_cube.html');
