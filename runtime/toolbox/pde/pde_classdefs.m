% PDE Toolbox classdef façade — MathWorks-faithful API surface.
%
% Auto-prepended by matlabc when the user input mentions any of:
%   - femodel
%   - materialProperties
%   - faceBC / edgeBC / vertexBC
%   - faceLoad / edgeLoad / vertexLoad / cellLoad
%   - StaticStructuralResults / StationaryResults
%
% Every classdef ctor body is a no-op — the matlabc Lowering
% kwarg-ctor sugar (lib/MLIR/Lowering.cpp) intercepts calls of the
% form `ClassName('Name', value, ...)` and emits `matlab_obj_new` +
% one `matlab_obj_set_*` per pair at the call site.  The ctor body
% itself never runs.
%
% Setters and solve() are C runtime builtins (registered in
% lib/Sema/Resolver.cpp, dispatched in lib/MLIR/Passes/LowerTensorOps.cpp,
% implemented in runtime/runtime_pde.cpp).  The MATLAB-side surface
% the user actually writes is:
%
%   gm    = pde_mesh_cuboid_tet(3, 0.05, 2, 12, 1, 8);
%   model = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
%   model = pde_set_material(model, ...
%               materialProperties('YoungsModulus', 2e11, ...
%                                  'PoissonsRatio', 0.3,  ...
%                                  'MassDensity',   7850));
%   model = pde_set_face_fixed(model, 5);
%   model = pde_set_face_pressure(model, 3, 3543);
%   R     = solve(model);
%   vm    = R.VonMisesStress;
%   ux    = R.Displacement.ux;

classdef materialProperties
    properties
        YoungsModulus
        PoissonsRatio
        MassDensity
        ThermalConductivity
        ThermalCondCoeff
        SpecificHeat
        CTE
        RelativePermittivity
        RelativePermeability
        ElectricalConductivity
    end
    methods
        function obj = materialProperties()
        end
    end
end

classdef faceBC
    properties
        Constraint
        Displacement
        XDisplacement
        YDisplacement
        ZDisplacement
        Temperature
        Voltage
    end
    methods
        function obj = faceBC()
        end
    end
end

classdef edgeBC
    properties
        Constraint
        Displacement
        XDisplacement
        YDisplacement
        ZDisplacement
        Temperature
        Voltage
    end
    methods
        function obj = edgeBC()
        end
    end
end

classdef vertexBC
    properties
        Constraint
        Displacement
        XDisplacement
        YDisplacement
        ZDisplacement
        Temperature
        Voltage
    end
    methods
        function obj = vertexBC()
        end
    end
end

classdef faceLoad
    properties
        Pressure
        SurfaceTraction
        Force
        Heat
        Temperature
        CurrentDensity
        ChargeDensity
    end
    methods
        function obj = faceLoad()
        end
    end
end

classdef edgeLoad
    properties
        Pressure
        SurfaceTraction
        Force
        Heat
        Temperature
        CurrentDensity
        ChargeDensity
    end
    methods
        function obj = edgeLoad()
        end
    end
end

classdef vertexLoad
    properties
        Force
        Heat
        Temperature
        CurrentDensity
        ChargeDensity
    end
    methods
        function obj = vertexLoad()
        end
    end
end

classdef cellLoad
    properties
        Force
        Heat
        Temperature
        CurrentDensity
        ChargeDensity
    end
    methods
        function obj = cellLoad()
        end
    end
end

classdef femodel
    properties
        AnalysisType
        Geometry
        Mesh
        MaterialProperties
        % Flat field layout — see the C kernel
        % `matlab_pde_solve_femodel` which reads these directly.
        FixedFaces
        PressureFaces
        PlanarType
    end
    methods
        function obj = femodel()
        end
    end
end

classdef pdeDisplacement
    properties
        ux
        uy
        uz
        Magnitude
    end
    methods
        function obj = pdeDisplacement()
        end
    end
end

classdef StaticStructuralResults
    properties
        Displacement
        VonMisesStress
        Mesh
    end
    methods
        function obj = StaticStructuralResults()
        end
    end
end

classdef StationaryResults
    properties
        NodalSolution
        Mesh
    end
    methods
        function obj = StationaryResults()
        end
    end
end

classdef ThermalResults
    properties
        Temperature
        Mesh
    end
    methods
        function obj = ThermalResults()
        end
    end
end

classdef ElectrostaticResults
    properties
        Voltage
        Mesh
    end
    methods
        function obj = ElectrostaticResults()
        end
    end
end

classdef MagneticResults
    properties
        MagneticPotential
        Mesh
    end
    methods
        function obj = MagneticResults()
        end
    end
end

classdef DCConductionResults
    properties
        Voltage
        Mesh
    end
    methods
        function obj = DCConductionResults()
        end
    end
end

classdef TransientStructuralResults
    properties
        Displacement
        Uhist
        SolutionTimes
        Mesh
    end
    methods
        function obj = TransientStructuralResults()
        end
    end
end

classdef ModalStructuralResults
    properties
        NaturalFrequencies
        ModeShapes
        Mesh
    end
    methods
        function obj = ModalStructuralResults()
        end
    end
end

classdef FrequencyStructuralResults
    properties
        FrequencyList
        Displacement
        Mesh
    end
    methods
        function obj = FrequencyStructuralResults()
        end
    end
end

classdef HarmonicEMResults
    properties
        WaveField
        Mesh
    end
    methods
        function obj = HarmonicEMResults()
        end
    end
end
