% sim3d command-line API — construct a world + actor, animate, read back.
% Exercises the parser fold (sim3d.World/sim3d.Actor), handle classdefs with
% Dependent transform properties forwarding to the C++ runtime, the
% open/run/close recording loop, and getter round-trips.
w = sim3d.World();
a = sim3d.Actor('cube', 'box');
w.add(a);
w.open();
a.Translation = [1 2 3];
w.run(0.1);
a.Translation = [4 5 6];
a.Rotation = [0 0 1.5];
a.Scale = [2 2 2];
w.run(0.1);
w.close();
disp(a.Translation);
disp(a.Rotation);
disp(a.Scale);
