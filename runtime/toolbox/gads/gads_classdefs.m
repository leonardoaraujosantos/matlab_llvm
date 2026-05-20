% Global Optimization Toolbox — Tier-2 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `MultiStart`,
% `GlobalSearch`, `createOptimProblem`, or a `run(` call.  The two solver
% objects are thin property holders; all orchestration lives in
% runtime/toolbox/gads/runtime_gads.cpp.  `run(solver, problem [, k])`
% and `createOptimProblem(...)` are NOT classdef methods — they are
% registered as builtins in Resolver.cpp and dispatched in Lowering.cpp
% (run keys on the solver's pinned class; createOptimProblem scans its
% name-value args).  The objective handle rides from createOptimProblem
% to run through a thread-local context in the runtime.

classdef MultiStart
    properties
        Display       % 'off' (default) / 'iter' / 'final'
        UseParallel   % scalar flag (Tier-6; ignored today)
    end
    methods
        function obj = MultiStart()
            obj.Display     = 0;
            obj.UseParallel = 0;
        end
    end
end

classdef GlobalSearch
    properties
        Display
        NumTrialPoints   % stage-1 scatter sample size (Tier-6 tuning)
    end
    methods
        function obj = GlobalSearch()
            obj.Display        = 0;
            obj.NumTrialPoints = 200;
        end
    end
end

% optimoptions(solver, 'Name', val, ...) — Tier-6 options carrier.  The
% Lowering intercept allocates this shell (zero-arg ctor) and writes the
% named fields; the runtime reads them (sentinel −1 = "use the solver
% default").  Wired into `ga` for Tier-6 (PopulationSize / MaxGenerations
% / IntCon); other solvers' options are a follow-on.
classdef optimoptions
    properties
        PopulationSize
        SwarmSize
        MaxGenerations
        MaxIterations
        FunctionTolerance
        IntCon matrix      % integer-variable indices (empty/0 = none)
    end
    methods
        function obj = optimoptions()
            obj.PopulationSize    = -1;
            obj.SwarmSize         = -1;
            obj.MaxGenerations    = -1;
            obj.MaxIterations     = -1;
            obj.FunctionTolerance = -1;
            obj.IntCon            = zeros(1, 1);
        end
    end
end
