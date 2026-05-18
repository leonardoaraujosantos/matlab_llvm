% Optimization Toolbox problem-based API — stdlib classdefs (Tier-4).
%
% Auto-prepended by matlabc when the user input mentions `optimvar`,
% `optimproblem`, or `OptimizationExpression` / `OptimizationProblem`
% — see the prelude table in `tools/matlabc/main.cpp`.
%
% The expression DAG lives entirely in the runtime
% (`runtime/runtime_optim.cpp`, the `matlab_optim_pb_*` family).  These
% classdefs are thin boxes: an `OptimizationExpression` carries a
% single `Id` property (a DAG node id); every operator overload
% forwards to a runtime builder and wraps the returned id.  The
% comparison operators (`<=` / `>=` / `==`) return the constraint
% node id as a bare scalar — `prob.Constraints.<name> = ...` stores it
% directly and `solve` reads it back.
%
% Tier-4 scope: scalar optimisation variables.  Vector/matrix
% `optimvar`, `eqnproblem`, `show` / `write`, `prob2struct` are
% deferred — see docs/optim_toolbox_roadmap.md §5.

classdef OptimizationExpression
    properties
        Id
    end
    methods
        function obj = OptimizationExpression(a, b)
            % One-arg form: `a` is a raw numeric constant arriving from
            % the operator-dispatch scalar-boxing — make a constant
            % node.  Two-arg form: `(dummy, node_id)` — wrap an
            % existing DAG node id produced by a runtime builder.
            if nargin == 1
                obj.Id = matlab_optim_pb_const(a);
            elseif nargin == 2
                obj.Id = b;
            end
        end

        function r = plus(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_add(a.Id, b.Id));
        end
        function r = minus(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_sub(a.Id, b.Id));
        end
        function r = uminus(a)
            r = OptimizationExpression(0, matlab_optim_pb_neg(a.Id));
        end
        function r = mtimes(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_mul(a.Id, b.Id));
        end
        function r = times(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_mul(a.Id, b.Id));
        end
        function r = mrdivide(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_div(a.Id, b.Id));
        end
        function r = rdivide(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_div(a.Id, b.Id));
        end
        function r = mpower(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_pow(a.Id, b.Id));
        end
        function r = power(a, b)
            r = OptimizationExpression(0, matlab_optim_pb_pow(a.Id, b.Id));
        end

        % Relational operators build constraint nodes; the result is
        % the bare node id (a scalar), which the problem stores
        % directly in its Constraints struct.
        function r = le(a, b)
            r = matlab_optim_pb_le(a.Id, b.Id);
        end
        function r = ge(a, b)
            r = matlab_optim_pb_ge(a.Id, b.Id);
        end
        function r = eq(a, b)
            r = matlab_optim_pb_eq(a.Id, b.Id);
        end
    end
end

classdef OptimizationProblem
    properties
        Objective
        Constraints
        Maximize
    end
    methods
        function obj = OptimizationProblem()
        end

        % `solve(prob)` — dispatched here when the first argument is an
        % OptimizationProblem (the class-pinned-first-arg routing also
        % used by step / bode).  Hands the whole classdef object to the
        % runtime, which reads Objective / Constraints / Maximize.
        function sol = solve(prob)
            sol = matlab_optim_pb_solve(prob);
        end
    end
end

% EquationProblem — the problem-based equation-solving counterpart of
% OptimizationProblem.  Equations are stored as named fields of the
% `Equations` struct property (`prob.Equations.e1 = lhs == rhs`); each
% relational `==` expression contributes a residual node.  `solve`
% hands the object to the runtime, which solves F(x) = 0 by
% Levenberg-Marquardt.
classdef EquationProblem
    properties
        Equations
    end
    methods
        function obj = EquationProblem()
        end

        function sol = solve(prob)
            sol = matlab_optim_pb_solve_eqn(prob);
        end
    end
end

% optimvar()     -> a new continuous scalar optimisation variable
% optimintvar()  -> a new integer    scalar optimisation variable
%
% Tier-4 scope: scalar variables, created with no arguments.  The
% MATLAB `optimvar('name', ...)` string/keyword form is deferred —
% `solve` returns the solution as a column vector in variable-creation
% order, so the name is purely cosmetic.  Bounds are expressed as
% ordinary `x >= lo` / `x <= hi` constraints.
function v = optimvar()
    v = OptimizationExpression(0, matlab_optim_pb_var(0));
end

function v = optimintvar()
    v = OptimizationExpression(0, matlab_optim_pb_var(1));
end

% optimproblem() -> an empty OptimizationProblem (minimisation by
% default; set prob.Maximize = 1 to maximise).
function p = optimproblem()
    p = OptimizationProblem();
end

% eqnproblem() -> an empty EquationProblem.  Set named equations via
% prob.Equations.<name> = (lhs == rhs); `solve` finds x with all
% equations satisfied by Levenberg-Marquardt on the residual.
function p = eqnproblem()
    p = EquationProblem();
end
