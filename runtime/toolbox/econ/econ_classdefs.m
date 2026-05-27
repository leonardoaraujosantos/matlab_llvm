% Econometrics Toolbox — classdef umbrella.
% Auto-prepended by matlabc when the user input mentions an Econometrics
% model constructor (arima/garch/egarch/gjr/varm/vecm/ssm/dssm/regARIMA/
% bayeslm/dtmc).  See tools/matlabc/main.cpp Want[].
%
% Heavy lifting (estimation, forecasting, filtering) lives in
% runtime/toolbox/econ/runtime_econ.cpp (matlab_econ_* C-ABI entries),
% dispatched by Lowering.cpp's class-pinned-first-arg path keyed on the
% receiver class name (so the names estimate/forecast/infer/simulate that
% collide with System-Identification's idpoly route correctly by class).
%
% Tier-1 is entirely function-form (autocorr/adftest/hpfilter/...) and
% needs no classdef.  Model objects arrive with Tiers 2+.

% ---------------------------------------------------------------------
% arima — conditional-mean ARIMA(p,D,q) model (Tier-2).  Carries the AR
% and MA polynomials (estimated coefficient vectors), the integration
% order D, the additive Constant, and the innovation Variance.  The
% estimate/forecast/infer/simulate methods are dispatched class-pinned
% in Lowering to the matlab_econ_arima_* runtime kernels (Hannan-
% Rissanen estimation + recursive forecasting).
% ---------------------------------------------------------------------
classdef arima
    properties
        P              % AR order
        D              % integration (differencing) order
        Q              % MA order
        AR matrix      % 1 x P estimated AR coefficients
        MA matrix      % 1 x Q estimated MA coefficients
        Constant       % additive constant
        Variance       % innovation variance
        ModelKind      % dispatch discriminant (0 = arima)
    end
    methods
        function obj = arima(p, D, q)
            if nargin < 1, p = 0; end
            if nargin < 2, D = 0; end
            if nargin < 3, q = 0; end
            obj.P = p;
            obj.D = D;
            obj.Q = q;
            obj.AR = zeros(1, 1);
            obj.MA = zeros(1, 1);
            obj.Constant = 0;
            obj.Variance = 1;
            obj.ModelKind = 0;
        end
    end
end

% ---------------------------------------------------------------------
% garch / egarch / gjr — conditional-variance models (Tier-3).  All three
% share a property set; the ModelKind discriminant (1=garch, 2=egarch,
% 3=gjr) routes the shared matlab_econ_garch_* kernels.  P = number of
% GARCH (lagged-variance) terms, Q = number of ARCH (lagged-squared-
% innovation) terms.  Estimation is Gaussian MLE over a Nelder-Mead
% simplex.
% ---------------------------------------------------------------------
classdef garch
    properties
        P; Q;
        Constant       % conditional-variance constant (kappa)
        GARCH matrix   % 1 x P lagged-variance coefficients
        ARCH matrix    % 1 x Q lagged-squared-innovation coefficients
        Leverage matrix% leverage coefficient (egarch/gjr)
        Offset         % conditional-mean offset (sample mean)
        Variance       % unconditional variance
        ModelKind
    end
    methods
        function obj = garch(P, Q)
            if nargin < 1, P = 1; end
            if nargin < 2, Q = 1; end
            obj.P = P; obj.Q = Q;
            obj.Constant = 0;
            obj.GARCH = zeros(1, 1);
            obj.ARCH = zeros(1, 1);
            obj.Leverage = zeros(1, 1);
            obj.Offset = 0;
            obj.Variance = 1;
            obj.ModelKind = 1;
        end
    end
end

classdef egarch
    properties
        P; Q;
        Constant; GARCH matrix; ARCH matrix; Leverage matrix;
        Offset; Variance; ModelKind
    end
    methods
        function obj = egarch(P, Q)
            if nargin < 1, P = 1; end
            if nargin < 2, Q = 1; end
            obj.P = P; obj.Q = Q;
            obj.Constant = 0;
            obj.GARCH = zeros(1, 1);
            obj.ARCH = zeros(1, 1);
            obj.Leverage = zeros(1, 1);
            obj.Offset = 0;
            obj.Variance = 1;
            obj.ModelKind = 2;
        end
    end
end

classdef gjr
    properties
        P; Q;
        Constant; GARCH matrix; ARCH matrix; Leverage matrix;
        Offset; Variance; ModelKind
    end
    methods
        function obj = gjr(P, Q)
            if nargin < 1, P = 1; end
            if nargin < 2, Q = 1; end
            obj.P = P; obj.Q = Q;
            obj.Constant = 0;
            obj.GARCH = zeros(1, 1);
            obj.ARCH = zeros(1, 1);
            obj.Leverage = zeros(1, 1);
            obj.Offset = 0;
            obj.Variance = 1;
            obj.ModelKind = 3;
        end
    end
end

% ---------------------------------------------------------------------
% varm — vector autoregression VAR(P) for k series (Tier-4).  Constant is
% k x 1; AR is the lag-coefficient matrices stacked horizontally k x (k*P);
% Covariance is the k x k residual covariance.  estimate runs equation-wise
% OLS; forecast/simulate/irf operate on the stacked coefficients.
% ---------------------------------------------------------------------
classdef varm
    properties
        NumSeries
        P
        Constant matrix
        AR matrix
        Covariance matrix
        ModelKind
    end
    methods
        function obj = varm(numSeries, numLags)
            if nargin < 1, numSeries = 1; end
            if nargin < 2, numLags = 1; end
            obj.NumSeries = numSeries;
            obj.P = numLags;
            obj.Constant = zeros(1, 1);
            obj.AR = zeros(1, 1);
            obj.Covariance = zeros(1, 1);
            obj.ModelKind = 4;
        end
    end
end

% ---------------------------------------------------------------------
% ssm / dssm — linear-Gaussian state-space models (Tier-5).
%   x_t = A x_{t-1} + B w_t,   y_t = C x_t + D v_t
% estimate runs Kalman-filter ML over the free B/D entries; filter/smooth
% return the (filtered / RTS-smoothed) latent states; forecast extrapolates
% the observation equation.  dssm is the diffuse-initialization variant
% (ModelKind 7); ssm is ModelKind 6.
% ---------------------------------------------------------------------
classdef ssm
    properties
        A matrix; B matrix; C matrix; D matrix; ModelKind
    end
    methods
        function obj = ssm(a, b, c, d)
            if nargin < 1, a = zeros(1, 1); end
            if nargin < 2, b = zeros(1, 1); end
            if nargin < 3, c = zeros(1, 1); end
            if nargin < 4, d = zeros(1, 1); end
            obj.A = a; obj.B = b; obj.C = c; obj.D = d;
            obj.ModelKind = 6;
        end
    end
end

classdef dssm
    properties
        A matrix; B matrix; C matrix; D matrix; ModelKind
    end
    methods
        function obj = dssm(a, b, c, d)
            if nargin < 1, a = zeros(1, 1); end
            if nargin < 2, b = zeros(1, 1); end
            if nargin < 3, c = zeros(1, 1); end
            if nargin < 4, d = zeros(1, 1); end
            obj.A = a; obj.B = b; obj.C = c; obj.D = d;
            obj.ModelKind = 7;
        end
    end
end
