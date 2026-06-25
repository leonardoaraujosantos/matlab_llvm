% sim3d_classdefs.m — Simulink 3D Animation command-line surface.
%
% Auto-prepended by matlabc when the user input mentions a `sim3d.*` member
% (the parser folds `sim3d.World` -> `sim3d_World`, `sim3d.Actor` ->
% `sim3d_Actor`, `sim3d.export` -> `sim3d_export`; see lib/Parse/Parser.cpp).
%
% Faithful to MathWorks' sim3d framework: build a World, add Actors, set their
% Translation/Rotation/Scale in a loop, and export. All scene + timeline state
% lives in the C++ runtime (runtime/toolbox/sim3d/runtime_sim3d.cpp); these
% classes are thin `handle` wrappers whose one-line method bodies forward the
% receiver `obj` into the `matlab_sim3d_*` entries (the DSP System-Object
% convention). Transform properties are `Dependent` so assignment dispatches a
% set.* accessor that writes through to the runtime (plain properties would not).
%
% Rendering is accumulate-then-export: open(world) begins recording, each
% run(world, dt) records one keyframe of every actor's current transform, and
% sim3d.export(world, path) writes a self-contained Babylon.js HTML player.

classdef sim3d_World < handle
    properties
        Name
    end
    methods
        function obj = sim3d_World()
            matlab_sim3d_world_new(obj);
        end
        function add(obj, actor)
            matlab_sim3d_add(obj, actor);
        end
        function open(obj)
            matlab_sim3d_open(obj);
        end
        function run(obj, dt)
            matlab_sim3d_run(obj, dt);
        end
        function close(obj)
            matlab_sim3d_close(obj);
        end
    end
end

classdef sim3d_Actor < handle
    properties
        Name
        Shape
        Color
        Size
    end
    properties (Dependent)
        Translation
        Rotation
        Scale
    end
    methods
        function obj = sim3d_Actor(name, shape)
            obj.Name = name;
            obj.Shape = shape;
            matlab_sim3d_actor_new(obj, name, shape);
        end
        function set.Translation(obj, v)
            matlab_sim3d_set_translation(obj, v);
        end
        function v = get.Translation(obj)
            v = matlab_sim3d_get_translation(obj);
        end
        function set.Rotation(obj, v)
            matlab_sim3d_set_rotation(obj, v);
        end
        function v = get.Rotation(obj)
            v = matlab_sim3d_get_rotation(obj);
        end
        function set.Scale(obj, v)
            matlab_sim3d_set_scale(obj, v);
        end
        function v = get.Scale(obj)
            v = matlab_sim3d_get_scale(obj);
        end
        function setParent(obj, p)
            matlab_sim3d_set_parent(obj, p);
        end
    end
end
