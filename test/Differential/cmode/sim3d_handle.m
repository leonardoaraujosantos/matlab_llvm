% Handle classdef (sim3d.World/Actor) over the runtime object model.
w = sim3d.World();
a = sim3d.Actor('cube', 'box');
a.Translation = [1 2 3];
w.add(a);
w.open();
for k = 1:3
    a.Translation = [k 0 1];
    w.run(0.1);
end
w.close();
disp('sim3d handle ok');
