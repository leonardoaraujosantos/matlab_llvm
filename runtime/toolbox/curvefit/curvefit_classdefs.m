% Curve Fitting Toolbox — Tier-1 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `fit`, `cfit`,
% `fittype`, or `fitoptions`.  The fitted-model object is a thin descriptor;
% all numeric work (fitting, evaluation, goodness-of-fit) lives in
% runtime/toolbox/curvefit/runtime_curvefit.cpp.
%
% fit(...) is NOT a classdef method — it is a registered builtin dispatched
% in Lowering.cpp (it allocates this `cfit` shell then calls the runtime
% populate step matlab_curvefit_fit).  feval / coeffvalues / disp key on the
% object's class at runtime (REPL-safe), reading the fields below.  The
% `f(xq)` call-syntax routes to the `feval` method via the callable-instance
% sugar in Lowering.cpp.
%
% ModelType: 1 = polynomial (Tier-2+ adds exp / power / gauss / fourier).
% Coeffs ride MATLAB highest-power-first order in the centered-scaled domain;
% Mu / Sigma carry the center-and-scale conditioning so feval can rescale.

classdef cfit
    properties
        ModelType
        Degree
        Coeffs matrix
        Mu
        Sigma
        NumObs
        NumCoeffs
        SSE
        Rsquare
        DFE
        AdjRsquare
        RMSE
        Resid matrix
        Xdata matrix
        Expr
    end
    methods
        function obj = cfit()
            obj.ModelType  = 1;
            obj.Degree     = 1;
            obj.Coeffs     = zeros(1, 2);
            obj.Mu         = 0;
            obj.Sigma      = 1;
            obj.NumObs     = 0;
            obj.NumCoeffs  = 0;
            obj.SSE        = 0;
            obj.Rsquare    = 0;
            obj.DFE        = 0;
            obj.AdjRsquare = 0;
            obj.RMSE       = 0;
            obj.Resid      = zeros(1, 1);
            obj.Xdata      = zeros(1, 1);
            obj.Expr       = 0;
        end
        function y = feval(obj, xq)
            y = matlab_curvefit_feval(obj, xq);
        end
        function c = coeffvalues(obj)
            c = matlab_curvefit_coeffvalues(obj);
        end
        function d = differentiate(obj, xq)
            d = matlab_curvefit_differentiate(obj, xq);
        end
        function v = integrate(obj, xq)
            v = matlab_curvefit_integrate(obj, xq);
        end
        function ci = confint(obj)
            ci = matlab_curvefit_confint(obj, 0.95);
        end
        function n = numcoeffs(obj)
            n = matlab_curvefit_numcoeffs(obj);
        end
        function s = formula(obj)
            s = matlab_curvefit_formula(obj);
        end
        function disp(obj)
            matlab_curvefit_disp(obj);
        end
    end
end

% fittype — Tier-3 custom-equation descriptor.  fittype('a*exp(-b*x)+c') is
% intercepted in Lowering.cpp (constructor path) → matlab_curvefit_fittype_init
% stores the equation string + coefficient count.  fit(x,y,ft) reads it and
% runs the finite-difference Levenberg-Marquardt.
classdef fittype
    properties
        Expr
        NumCoeffs
        ModelType
    end
    methods
        function obj = fittype()
            obj.Expr      = 0;
            obj.NumCoeffs = 0;
            obj.ModelType = 100;
        end
    end
end

% fitoptions — Tier-2 solver-knob carrier for nonlinear fits.  Built by the
% name-value constructor fitoptions('StartPoint',sp,'Lower',lb,'Upper',ub,
% 'Weights',w,'Robust','Bisquare'), intercepted in Lowering.cpp (the
% optimoptions pattern).  fit(x,y,model,opts) reads these fields.  Robust is
% stored as a code (RobustCode: 0=off, 1=bisquare/on, 2=LAR) so the runtime
% needs no string compare.
classdef fitoptions
    properties
        StartPoint matrix
        Lower matrix
        Upper matrix
        Weights matrix
        RobustCode
        MaxIter
    end
    methods
        function obj = fitoptions()
            obj.StartPoint = zeros(1, 0);
            obj.Lower      = zeros(1, 0);
            obj.Upper      = zeros(1, 0);
            obj.Weights    = zeros(1, 0);
            obj.RobustCode = 0;
            obj.MaxIter    = 400;
        end
    end
end
