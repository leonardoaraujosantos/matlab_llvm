% State-space model — auto-prepended when matlabc sees `ss(` or
% `ss =` in the user input. See cst_classdefs.m for the umbrella
% comment + the tf classdef.

classdef ss
    properties
        A
        B
        C
        D
    end
    methods
        function obj = ss(A, B, C, D)
            % `ss(A, B, C, D)` — state-space model:
            %   x' = A x + B u
            %   y  = C x + D u
            % All four are ordinary matrices (n×n, n×m, p×n, p×m).
            % v1 only stores them; operator overloads (ss + ss,
            % ss * ss, feedback(ss, ss)) need real control-system math
            % (see the matrix-arg primitives `feedback_ss` /
            % `series_ss` / `parallel_ss` in matlab_runtime.cpp) and
            % are a follow-on slice.
            if nargin >= 1, obj.A = A; end
            if nargin >= 2, obj.B = B; end
            if nargin >= 3, obj.C = C; end
            if nargin >= 4, obj.D = D; end
        end
    end
end
