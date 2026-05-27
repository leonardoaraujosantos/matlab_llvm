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
