% Robotics System Toolbox — classdef umbrella (Tiers 1–6).
% Auto-prepended by matlabc when the user input mentions any robotics symbol
% (`se3`, `rigidBodyTree`, `inverseKinematics`, `binaryOccupancyMap`,
% `mobileRobotPRM`, `controllerPurePursuit`, `differentialDriveKinematics`,
% `collisionBox`, `manipulatorRRT`, ...).
%
% Every classdef ships a *neutral* zero-arg constructor only; the user-facing
% forms (e.g. `se3(T)`, `rigidBodyTree`, `inverseKinematics('RigidBodyTree',rb)`)
% are intercepted in Lowering.cpp's constructor path and populated by a
% `matlab_robotics_*_init` runtime call.  Method bodies forward to the
% `matlab_robotics_*` runtime entries.

% Tier-1 ------------------------------------------------------------------

classdef se3
    properties
        Data matrix   % 4×4 homogeneous transform
    end
    methods
        function obj = se3()
            obj.Data = zeros(4, 4);
            obj.Data(1, 1) = 1;
            obj.Data(2, 2) = 1;
            obj.Data(3, 3) = 1;
            obj.Data(4, 4) = 1;
        end
    end
end

classdef so3
    properties
        Data matrix   % 3×3 rotation matrix
    end
    methods
        function obj = so3()
            obj.Data = zeros(3, 3);
            obj.Data(1, 1) = 1;
            obj.Data(2, 2) = 1;
            obj.Data(3, 3) = 1;
        end
    end
end

% Tier-2 ------------------------------------------------------------------

% rigidBodyTree — kinematic tree as packed property matrices.
%   DH         : N×4 [a alpha d theta_offset] Denavit-Hartenberg per joint
%   JointTypes : N×1 (1=revolute, 2=prismatic, 0=fixed)
%   JointLimits: N×2 [low high]
%   NumBodies  : scalar
%   Representation : 0 = DH chain, 1 = fixed-transform + axis (URDF)
%   PreTransforms  : N×16 (4×4 parent->joint fixed transform, URDF form)
%   JointAxes      : N×3  unit joint axes (URDF form)
%   Mass / COM / Inertia : per-link dynamics (COM N×3 in link frame,
%                          Inertia N×6 [Ixx Iyy Izz Iyz Ixz Ixy] about COM)
%   Gravity        : 1×3  gravity vector (default [0 0 -9.81])
classdef rigidBodyTree
    properties
        DH matrix
        JointTypes matrix
        JointLimits matrix
        NumBodies
        Representation
        PreTransforms matrix
        JointAxes matrix
        Mass matrix
        COM matrix
        Inertia matrix
        Gravity matrix
    end
    methods
        function obj = rigidBodyTree()
            obj.DH             = zeros(0, 4);
            obj.JointTypes     = zeros(0, 1);
            obj.JointLimits    = zeros(0, 2);
            obj.NumBodies      = 0;
            obj.Representation = 0;
            obj.PreTransforms  = zeros(0, 16);
            obj.JointAxes      = zeros(0, 3);
            obj.Mass           = zeros(0, 1);
            obj.COM            = zeros(0, 3);
            obj.Inertia        = zeros(0, 6);
            obj.Gravity        = [0, 0, -9.81];
        end
    end
end

% Tier-3 ------------------------------------------------------------------

% inverseKinematics — solver carrier.  Holds a rigidBodyTree handle (stored
% as cloned matrices on this obj for runtime access) plus solver knobs.
classdef inverseKinematics
    properties
        DH matrix
        JointTypes matrix
        JointLimits matrix
        NumBodies
        MaxIterations
        SolutionTolerance
    end
    methods
        function obj = inverseKinematics()
            obj.DH                = zeros(0, 4);
            obj.JointTypes        = zeros(0, 1);
            obj.JointLimits       = zeros(0, 2);
            obj.NumBodies         = 0;
            obj.MaxIterations     = 200;
            obj.SolutionTolerance = 1e-6;
        end
    end
end

% Constraint containers.  Kind discriminant: 1=pose, 2=position, 3=orientation.
% TargetTransform (4×4) carries both the target rotation and translation;
% position/orientation constraints use only the relevant block.
classdef constraintPoseTarget
    properties
        Kind
        TargetTransform matrix   % 4×4
        Weights matrix           % 1×6  [orientation(1:3) position(4:6)]
    end
    methods
        function obj = constraintPoseTarget()
            obj.Kind = 1;
            obj.TargetTransform = zeros(4, 4);
            obj.TargetTransform(1, 1) = 1;
            obj.TargetTransform(2, 2) = 1;
            obj.TargetTransform(3, 3) = 1;
            obj.TargetTransform(4, 4) = 1;
            obj.Weights = ones(1, 6);
        end
    end
end

classdef constraintPositionTarget
    properties
        Kind
        TargetTransform matrix
        Weights matrix
    end
    methods
        function obj = constraintPositionTarget()
            obj.Kind = 2;
            obj.TargetTransform = zeros(4, 4);
            obj.TargetTransform(1, 1) = 1;
            obj.TargetTransform(2, 2) = 1;
            obj.TargetTransform(3, 3) = 1;
            obj.TargetTransform(4, 4) = 1;
            obj.Weights = ones(1, 6);
        end
    end
end

classdef constraintOrientationTarget
    properties
        Kind
        TargetTransform matrix
        Weights matrix
    end
    methods
        function obj = constraintOrientationTarget()
            obj.Kind = 3;
            obj.TargetTransform = zeros(4, 4);
            obj.TargetTransform(1, 1) = 1;
            obj.TargetTransform(2, 2) = 1;
            obj.TargetTransform(3, 3) = 1;
            obj.TargetTransform(4, 4) = 1;
            obj.Weights = ones(1, 6);
        end
    end
end

% generalizedInverseKinematics — multi-constraint solver carrier.
classdef generalizedInverseKinematics
    properties
        DH matrix
        JointTypes matrix
        JointLimits matrix
        NumBodies
        Representation
        PreTransforms matrix
        JointAxes matrix
        MaxIterations
        SolutionTolerance
    end
    methods
        function obj = generalizedInverseKinematics()
            obj.DH                = zeros(0, 4);
            obj.JointTypes        = zeros(0, 1);
            obj.JointLimits       = zeros(0, 2);
            obj.NumBodies         = 0;
            obj.Representation     = 0;
            obj.PreTransforms      = zeros(0, 16);
            obj.JointAxes          = zeros(0, 3);
            obj.MaxIterations     = 200;
            obj.SolutionTolerance = 1e-6;
        end
    end
end

% Tier-5 ------------------------------------------------------------------

% Differential-drive kinematics: two independently-driven wheels.
% State [x y theta]; command [v omega] (VehicleSpeedHeadingRate).
classdef differentialDriveKinematics
    properties
        WheelRadius
        TrackWidth
    end
    methods
        function obj = differentialDriveKinematics()
            obj.WheelRadius = 0.1;
            obj.TrackWidth  = 0.5;
        end
    end
end

% Unicycle: single rolling wheel.  State [x y theta]; command [v omega].
classdef unicycleKinematics
    properties
        WheelRadius
    end
    methods
        function obj = unicycleKinematics()
            obj.WheelRadius = 0.1;
        end
    end
end

% Bicycle: car-like with front steering angle.  State [x y theta];
% command [v psi] (psi = steering angle).  derivative uses WheelBase.
classdef bicycleKinematics
    properties
        WheelBase
        MaxSteeringAngle
    end
    methods
        function obj = bicycleKinematics()
            obj.WheelBase        = 1.0;
            obj.MaxSteeringAngle = pi/4;
        end
    end
end

% Ackermann: car-like with Ackermann steering.  State [x y theta psi];
% command [v psidot] (psi = steering angle, psidot its rate).
classdef ackermannKinematics
    properties
        WheelBase
        MaxSteeringAngle
    end
    methods
        function obj = ackermannKinematics()
            obj.WheelBase        = 1.0;
            obj.MaxSteeringAngle = pi/4;
        end
    end
end

% Binary occupancy grid map: a 2-D matrix of 0/1 cells + world-units metadata.
classdef binaryOccupancyMap
    properties
        Grid matrix
        Resolution                 % cells per metre
        GridSize matrix            % 1×2 [rows cols]
        XWorldLimits matrix        % 1×2
        YWorldLimits matrix        % 1×2
    end
    methods
        function obj = binaryOccupancyMap()
            obj.Grid         = zeros(10, 10);
            obj.Resolution   = 1.0;
            obj.GridSize     = [10, 10];
            obj.XWorldLimits = [0, 10];
            obj.YWorldLimits = [0, 10];
        end
    end
end

% Probabilistic roadmap: a sampled-node graph over an occupancy map.
classdef mobileRobotPRM
    properties
        % Map fields cloned in for the runtime (sidesteps cross-obj refs).
        Grid matrix
        Resolution
        GridSize matrix
        XWorldLimits matrix
        YWorldLimits matrix
        NumNodes
        ConnectionDistance
        Nodes matrix               % N×2 sampled (free) positions
        Edges matrix               % M×3 [a b cost]
    end
    methods
        function obj = mobileRobotPRM()
            obj.Grid               = zeros(10, 10);
            obj.Resolution         = 1.0;
            obj.GridSize           = [10, 10];
            obj.XWorldLimits       = [0, 10];
            obj.YWorldLimits       = [0, 10];
            obj.NumNodes           = 50;
            obj.ConnectionDistance = 2.5;
            obj.Nodes              = zeros(0, 2);
            obj.Edges              = zeros(0, 3);
        end
    end
end

% Pure-pursuit path follower.
classdef controllerPurePursuit
    properties
        Waypoints matrix           % N×2 (or N×3)
        LookaheadDistance
        DesiredLinearVelocity
        MaxAngularVelocity
        CurrentWaypointIdx
    end
    methods
        function obj = controllerPurePursuit()
            obj.Waypoints              = zeros(0, 2);
            obj.LookaheadDistance      = 0.3;
            obj.DesiredLinearVelocity  = 0.5;
            obj.MaxAngularVelocity     = 2.0;
            obj.CurrentWaypointIdx     = 1;
        end
    end
end

% Tier-6 ------------------------------------------------------------------
% Collision primitives.  ShapeKind: 1=box, 2=sphere, 3=cylinder, 4=capsule.
% Pose is a 4×4 world transform (orientation respected by the GJK support
% functions); X/Y/Z full side lengths for box, Radius/Length for the others.

classdef collisionBox
    properties
        ShapeKind
        X
        Y
        Z
        Pose matrix
    end
    methods
        function obj = collisionBox()
            obj.ShapeKind = 1;
            obj.X    = 1.0;
            obj.Y    = 1.0;
            obj.Z    = 1.0;
            obj.Pose = zeros(4, 4);
            obj.Pose(1, 1) = 1;
            obj.Pose(2, 2) = 1;
            obj.Pose(3, 3) = 1;
            obj.Pose(4, 4) = 1;
        end
    end
end

classdef collisionSphere
    properties
        ShapeKind
        Radius
        Pose matrix
    end
    methods
        function obj = collisionSphere()
            obj.ShapeKind = 2;
            obj.Radius = 0.5;
            obj.Pose   = zeros(4, 4);
            obj.Pose(1, 1) = 1;
            obj.Pose(2, 2) = 1;
            obj.Pose(3, 3) = 1;
            obj.Pose(4, 4) = 1;
        end
    end
end

classdef collisionCylinder
    properties
        ShapeKind
        Radius
        Length
        Pose matrix
    end
    methods
        function obj = collisionCylinder()
            obj.ShapeKind = 3;
            obj.Radius = 0.5;
            obj.Length = 1.0;
            obj.Pose   = zeros(4, 4);
            obj.Pose(1, 1) = 1;
            obj.Pose(2, 2) = 1;
            obj.Pose(3, 3) = 1;
            obj.Pose(4, 4) = 1;
        end
    end
end

classdef collisionCapsule
    properties
        ShapeKind
        Radius
        Length
        Pose matrix
    end
    methods
        function obj = collisionCapsule()
            obj.ShapeKind = 4;
            obj.Radius = 0.5;
            obj.Length = 1.0;
            obj.Pose   = zeros(4, 4);
            obj.Pose(1, 1) = 1;
            obj.Pose(2, 2) = 1;
            obj.Pose(3, 3) = 1;
            obj.Pose(4, 4) = 1;
        end
    end
end

% manipulatorRRT — sampling-based planner over a rigidBodyTree config space.
classdef manipulatorRRT
    properties
        DH matrix
        JointTypes matrix
        JointLimits matrix
        NumBodies
        ObstacleCenters matrix      % K×3 sphere obstacle centres
        ObstacleRadii matrix        % K×1
        MaxConnectionDistance
        MaxIterations
    end
    methods
        function obj = manipulatorRRT()
            obj.DH                    = zeros(0, 4);
            obj.JointTypes            = zeros(0, 1);
            obj.JointLimits           = zeros(0, 2);
            obj.NumBodies             = 0;
            obj.ObstacleCenters       = zeros(0, 3);
            obj.ObstacleRadii         = zeros(0, 1);
            obj.MaxConnectionDistance = 0.3;
            obj.MaxIterations         = 200;
        end
    end
end
