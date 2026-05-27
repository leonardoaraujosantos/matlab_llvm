% Navigation Toolbox — classdef umbrella (Tiers 1–4).
% Auto-prepended by matlabc when the user input mentions any navigation symbol
% (`occupancyMap`, `stateSpaceSE2`, `stateSpaceDubins`, `validatorOccupancyMap`,
% `navPath`, `plannerRRT`, `plannerRRTStar`, `plannerAStarGrid`, `lidarScan`,
% `lidarSLAM`, `poseGraph`, ...).
%
% Every classdef ships a *neutral* zero-arg constructor only; the user-facing
% forms (e.g. `occupancyMap(10,10,5)`, `plannerRRT(ss,sv)`) are intercepted in
% Lowering.cpp's constructor path and populated by a `matlab_nav_*_init` runtime
% call.  Method bodies (`plan`, `inflate`, `isStateValid`, `matchScans`,
% `optimizePoseGraph`, ...) are likewise intercepted in Lowering.cpp.

% Tier-1 — occupancy map ---------------------------------------------------
classdef occupancyMap
    properties
        Grid matrix              % R×C occupancy probabilities in [0,1]
        Resolution               % cells per metre
        GridSize matrix          % 1×2 [rows cols]
        XWorldLimits matrix      % 1×2
        YWorldLimits matrix      % 1×2
        OccupiedThreshold
        FreeThreshold
    end
    methods
        function obj = occupancyMap()
            obj.Grid              = zeros(1, 1);
            obj.Resolution        = 1;
            obj.GridSize          = [1, 1];
            obj.XWorldLimits      = [0, 1];
            obj.YWorldLimits      = [0, 1];
            obj.OccupiedThreshold = 0.65;
            obj.FreeThreshold     = 0.2;
        end
    end
end

% Tier-1 — state spaces ----------------------------------------------------
% StateBounds is 3×2 [xmin xmax; ymin ymax; thmin thmax]; MinTurningRadius=0
% means a holonomic SE2 metric, >0 selects the Dubins-style turning penalty.
classdef stateSpaceSE2
    properties
        StateBounds matrix
        WeightTheta
        MinTurningRadius
    end
    methods
        function obj = stateSpaceSE2()
            obj.StateBounds      = [-100, 100; -100, 100; -pi, pi];
            obj.WeightTheta      = 1;
            obj.MinTurningRadius = 0;
        end
    end
end

classdef stateSpaceDubins
    properties
        StateBounds matrix
        WeightTheta
        MinTurningRadius
    end
    methods
        function obj = stateSpaceDubins()
            obj.StateBounds      = [-100, 100; -100, 100; -pi, pi];
            obj.WeightTheta      = 1;
            obj.MinTurningRadius = 1;
        end
    end
end

% Tier-1 — validator -------------------------------------------------------
% Carries a clone of the map grid + the state-space bounds for self-contained
% collision checks at plan time.
classdef validatorOccupancyMap
    properties
        Grid matrix
        Resolution
        OccupiedThreshold
        XWorldLimits matrix
        YWorldLimits matrix
        StateBounds matrix
        WeightTheta
        MinTurningRadius
        ValidationDistance
    end
    methods
        function obj = validatorOccupancyMap()
            obj.Grid               = zeros(1, 1);
            obj.Resolution         = 1;
            obj.OccupiedThreshold  = 0.65;
            obj.XWorldLimits       = [0, 1];
            obj.YWorldLimits       = [0, 1];
            obj.StateBounds        = [-100, 100; -100, 100; -pi, pi];
            obj.WeightTheta        = 1;
            obj.MinTurningRadius   = 0;
            obj.ValidationDistance = 0.1;
        end
    end
end

% Tier-1 — path container --------------------------------------------------
classdef navPath
    properties
        States matrix            % N×3 [x y θ]
    end
    methods
        function obj = navPath()
            obj.States = zeros(0, 3);
        end
    end
end

% Tier-2 — sampling planners ----------------------------------------------
% plannerRRT and plannerRRTStar share storage; IsStar discriminates.  They
% clone the validator's grid + bounds so plan() is self-contained.
classdef plannerRRT
    properties
        Grid matrix
        Resolution
        OccupiedThreshold
        StateBounds matrix
        MinTurningRadius
        ValidationDistance
        MaxConnectionDistance
        MaxIterations
        GoalBias
        IsStar
    end
    methods
        function obj = plannerRRT()
            obj.Grid                  = zeros(1, 1);
            obj.Resolution            = 1;
            obj.OccupiedThreshold     = 0.65;
            obj.StateBounds           = [-100, 100; -100, 100; -pi, pi];
            obj.MinTurningRadius      = 0;
            obj.ValidationDistance    = 0.1;
            obj.MaxConnectionDistance = 1;
            obj.MaxIterations         = 10000;
            obj.GoalBias              = 0.05;
            obj.IsStar                = 0;
        end
    end
end

classdef plannerRRTStar
    properties
        Grid matrix
        Resolution
        OccupiedThreshold
        StateBounds matrix
        MinTurningRadius
        ValidationDistance
        MaxConnectionDistance
        MaxIterations
        GoalBias
        IsStar
    end
    methods
        function obj = plannerRRTStar()
            obj.Grid                  = zeros(1, 1);
            obj.Resolution            = 1;
            obj.OccupiedThreshold     = 0.65;
            obj.StateBounds           = [-100, 100; -100, 100; -pi, pi];
            obj.MinTurningRadius      = 0;
            obj.ValidationDistance    = 0.1;
            obj.MaxConnectionDistance = 1;
            obj.MaxIterations         = 10000;
            obj.GoalBias              = 0.05;
            obj.IsStar                = 1;
        end
    end
end

classdef plannerAStarGrid
    properties
        Grid matrix
        OccupiedThreshold
    end
    methods
        function obj = plannerAStarGrid()
            obj.Grid              = zeros(1, 1);
            obj.OccupiedThreshold = 0.65;
        end
    end
end

% Tier-3 — lidar ----------------------------------------------------------
classdef lidarScan
    properties
        Ranges matrix
        Angles matrix
        Cartesian matrix         % N×2
    end
    methods
        function obj = lidarScan()
            obj.Ranges    = zeros(0, 1);
            obj.Angles    = zeros(0, 1);
            obj.Cartesian = zeros(0, 2);
        end
    end
end

classdef lidarSLAM
    properties
        Poses matrix             % N×3 absolute trajectory
        PrevCart matrix
        NumScans
    end
    methods
        function obj = lidarSLAM()
            obj.Poses    = zeros(0, 3);
            obj.PrevCart = zeros(0, 2);
            obj.NumScans = 0;
        end
    end
end

% Tier-4 — pose graph ------------------------------------------------------
classdef poseGraph
    properties
        NodeEstimates matrix     % N×3 [x y θ]
        Edges matrix             % M×6 [from to dx dy dθ infoScale]
    end
    methods
        function obj = poseGraph()
            obj.NodeEstimates = zeros(1, 3);
            obj.Edges         = zeros(0, 6);
        end
    end
end
