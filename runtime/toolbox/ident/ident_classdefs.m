% System Identification Toolbox — Tier-1 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `iddata(`,
% `idpoly(`, `arx(`, `ar(`, `compare(`, etc.  See tools/matlabc/main.cpp
% userMentionsExtClasses.  All heavy lifting (regressor assembly, the
% QR least-squares solve, simulation, prediction, fit metrics) lives in
% runtime/toolbox/ident/runtime_ident.cpp; the estimator entry points
% (arx / ar / sim / predict / compare / ...) are NOT declared as MATLAB
% functions here — they are registered as builtins in Resolver.cpp and
% dispatched in Lowering.cpp's class-pinned-first-arg path directly to
% the `matlab_ident_*` C-ABI entries (same pattern as MPC's mpcmove /
% sim).  Routing through a classdef-method body would land the inner
% runtime call inside a function whose formal-parameter slots carry
% `none` types and defeat the type-based dispatch.

% ---------------------------------------------------------------------
% iddata — time-domain input/output data container.  The central object
% every estimator consumes.  SISO in Tier-1 (single output column /
% single input column); MIMO stacking is a Tier-2 follow-on.
% ---------------------------------------------------------------------
classdef iddata
    properties
        OutputData matrix   % N × ny  measured outputs
        InputData  matrix   % N × nu  measured inputs (0×0 for a time series)
        Ts                  % scalar sample time (seconds)
        Tstart              % scalar start time
    end
    methods
        function obj = iddata(y, u, Ts)
            % `iddata(y, u, Ts)` — wrap measured signals.  For a time
            % series with no input use `iddata(y, [], Ts)` or
            % `iddata(y)`.  Default Ts = 1.
            if nargin < 1, y = zeros(1, 1); end
            if nargin < 2, u = zeros(0, 0); end
            if nargin < 3, Ts = 1; end
            obj.OutputData = y;
            obj.InputData  = u;
            obj.Ts         = Ts;
            obj.Tstart     = 0;
        end
    end
end

% ---------------------------------------------------------------------
% idpoly — identified input/output polynomial model.  Carries the five
% MATLAB polynomials A/B/C/D/F (row vectors) of the general structure
%
%     A(q) y(t) = [B(q)/F(q)] u(t-nk) + [C(q)/D(q)] e(t)
%
% Tier-1 fills A/B (ARX) or A only (AR); C/D/F stay monic [1].  The
% scalar bookkeeping fields (NoiseVariance, Np, Ns) feed fpe / aic.
% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
% idss — identified state-space model.  Innovations form
%
%     x(t+1) = A x(t) + B u(t) + K e(t)
%     y(t)   = C x(t) + D u(t) + e(t)
%
% Populated by n4sid / ssest (subspace + Ho-Kalman/ERA realization).
% Zero-arg ctor (the estimators allocate via idss__idss() then write the
% matrices in place); every matrix property is `matrix`-annotated so the
% setter routes through matlab_obj_set_mat with a none-typed RHS.
% ---------------------------------------------------------------------
classdef idss
    properties
        A matrix      % nx × nx state matrix
        B matrix      % nx × nu input matrix
        C matrix      % ny × nx output matrix
        D matrix      % ny × nu feedthrough
        K matrix      % nx × ny Kalman/innovations gain (0 in Tier-3)
        x0 matrix     % nx × 1 initial state
        Ts            % scalar sample time
        NoiseVariance % scalar innovations variance
        Np            % scalar number of estimated parameters
        Ns            % scalar number of samples used
    end
    methods
        function obj = idss()
            obj.A  = zeros(1, 1);
            obj.B  = zeros(1, 1);
            obj.C  = zeros(1, 1);
            obj.D  = zeros(1, 1);
            obj.K  = zeros(1, 1);
            obj.x0 = zeros(1, 1);
            obj.Ts            = 1;
            obj.NoiseVariance = 0;
            obj.Np            = 0;
            obj.Ns            = 0;
        end
    end
end

% ---------------------------------------------------------------------
% extendedKalmanFilter / unscentedKalmanFilter — dynamic nonlinear state
% estimators.  Carry a mutable State + StateCovariance updated in place
% by predict(obj, @StateFcn) and correct(obj, @MeasFcn, y).  The Fcn
% handles are single-arg (x) and passed at call time (they don't store
% on the object).  Scalar measurement.
% ---------------------------------------------------------------------
% Zero-arg ctors with concrete defaults; the user-facing 4-arg form
% `extendedKalmanFilter(x0,P0,Q,R)` is intercepted in Lowering.cpp's
% constructor path, which allocates this shell then populates the fields
% via the runtime `matlab_ident_ekf_init` (the arx/n4sid alloc-then-
% populate pattern — bare-param matrix assignment in a ctor body is
% brittle).
classdef extendedKalmanFilter
    properties
        State matrix            % x, nx × 1
        StateCovariance matrix  % P, nx × nx
        ProcessNoise matrix     % Q, nx × nx
        MeasurementNoise matrix % R, 1 × 1
    end
    methods
        function obj = extendedKalmanFilter()
            obj.State            = zeros(1, 1);
            obj.StateCovariance  = zeros(1, 1);
            obj.ProcessNoise     = zeros(1, 1);
            obj.MeasurementNoise = zeros(1, 1);
        end
    end
end

classdef unscentedKalmanFilter
    properties
        State matrix
        StateCovariance matrix
        ProcessNoise matrix
        MeasurementNoise matrix
    end
    methods
        function obj = unscentedKalmanFilter()
            obj.State            = zeros(1, 1);
            obj.StateCovariance  = zeros(1, 1);
            obj.ProcessNoise     = zeros(1, 1);
            obj.MeasurementNoise = zeros(1, 1);
        end
    end
end

% ---------------------------------------------------------------------
% recursiveLS / recursiveARX — online (recursive) estimators.  Mutable
% Parameters + Covariance updated in place by step(obj, ...).  Zero-arg
% ctors; the user-facing recursiveLS(np) / recursiveARX([na nb nk]) forms
% are intercepted in Lowering.cpp's constructor path (alloc-then-populate
% via matlab_ident_rls_init / _rarx_init).
% ---------------------------------------------------------------------
classdef recursiveLS
    properties
        Parameters matrix       % θ, np × 1
        Covariance matrix       % P, np × np
        ForgettingFactor        % λ (1 = infinite history)
    end
    methods
        function obj = recursiveLS()
            obj.Parameters       = zeros(1, 1);
            obj.Covariance       = zeros(1, 1);
            obj.ForgettingFactor = 1;
        end
    end
end

classdef recursiveARX
    properties
        Parameters matrix       % θ = [a; b]
        Covariance matrix
        ForgettingFactor
        na
        nb
        nk
        yhist matrix            % buffered past outputs
        uhist matrix            % buffered past inputs
    end
    methods
        function obj = recursiveARX()
            obj.Parameters       = zeros(1, 1);
            obj.Covariance       = zeros(1, 1);
            obj.ForgettingFactor = 1;
            obj.na = 0;
            obj.nb = 0;
            obj.nk = 1;
            obj.yhist = zeros(1, 1);
            obj.uhist = zeros(1, 1);
        end
    end
end

% ---------------------------------------------------------------------
% idfrd — frequency-response data (non-parametric).  Produced by etfe /
% spa.  The complex response G(e^{jω}) is stored as real magnitude +
% phase arrays (sidestepping complex object storage); Frequency is in
% rad/s.
% ---------------------------------------------------------------------
classdef idfrd
    properties
        Frequency matrix       % Nf × 1 frequencies (rad/s)
        ResponseMag matrix     % Nf × 1 |G(e^{jω})|
        ResponsePhase matrix   % Nf × 1 angle(G) (rad)
        Ts                     % scalar sample time
    end
    methods
        function obj = idfrd()
            obj.Frequency     = zeros(1, 1);
            obj.ResponseMag   = zeros(1, 1);
            obj.ResponsePhase = zeros(1, 1);
            obj.Ts            = 1;
        end
    end
end

% ---------------------------------------------------------------------
% idgrey — linear grey-box model.  greyest fits the physical Parameters
% of a user structure function (par -> packed continuous [A B; C D]) by
% prediction-error minimization, then discretizes (ZOH) at the data Ts.
% Carries both the estimated Parameters and the realized discrete model.
% ---------------------------------------------------------------------
classdef idgrey
    properties
        Parameters matrix   % estimated physical parameter vector
        A matrix            % nx × nx discrete state matrix
        B matrix            % nx × nu discrete input matrix
        C matrix            % ny × nx output matrix
        D matrix            % ny × nu feedthrough
        Ts                  % scalar sample time
        NoiseVariance       % scalar innovations variance
        Np                  % scalar number of estimated parameters
        Ns                  % scalar number of samples used
    end
    methods
        function obj = idgrey()
            obj.Parameters = zeros(1, 1);
            obj.A = zeros(1, 1);
            obj.B = zeros(1, 1);
            obj.C = zeros(1, 1);
            obj.D = zeros(1, 1);
            obj.Ts            = 1;
            obj.NoiseVariance = 0;
            obj.Np            = 0;
            obj.Ns            = 0;
        end
    end
end

% ---------------------------------------------------------------------
% idnlgrey — nonlinear grey-box model.  nlgreyest fits the physical
% Parameters of a user ODE StateFcn (z=[x;u;par] -> dx/dt) by PEM over
% an Euler rollout.  Carries the estimated Parameters.
% ---------------------------------------------------------------------
classdef idnlgrey
    properties
        Parameters matrix
        Ts
        NoiseVariance
        Np
        Ns
    end
    methods
        function obj = idnlgrey()
            obj.Parameters    = zeros(1, 1);
            obj.Ts            = 1;
            obj.NoiseVariance = 0;
            obj.Np            = 0;
            obj.Ns            = 0;
        end
    end
end

% ---------------------------------------------------------------------
% arxOptions — Tier-6 option carrier for the 3-arg arx(data,orders,opt)
% regularized-LS form.  Set `Regularization` to a scalar λ ≥ 0 to ridge
% the Gram solve `(ΦᵀΦ + λI) θ = Φᵀy`.  Reading the field is enough; no
% factory function is required.
% ---------------------------------------------------------------------
classdef arxOptions
    properties
        Regularization        % scalar ridge λ
    end
    methods
        function obj = arxOptions()
            obj.Regularization = 0;
        end
    end
end

classdef idpoly
    properties
        A matrix      % 1 × (na+1) autoregressive polynomial, A(1) = 1
        B matrix      % 1 × (nk+nb) input polynomial (nk leading zeros)
        C matrix      % 1 × (nc+1) noise-numerator polynomial
        D matrix      % 1 × (nd+1) noise-denominator polynomial
        F matrix      % 1 × (nf+1) input-denominator polynomial
        nk matrix     % 1 × nu input-delay vector
        Gram matrix   % (na+nb) × (na+nb) regularized Gram cached by arx
                      % (T6 — feeds getcov; 0×0 for PEM-fit models)
        Ts            % scalar sample time
        NoiseVariance % scalar innovations variance V (loss function)
        Np            % scalar number of estimated parameters
        Ns            % scalar number of samples used in estimation
        Lambda        % scalar regularization factor used at fit time (T6)
        Fit           % scalar NRMSE fit % (filled by compare; 0 until then)
    end
    methods
        function obj = idpoly()
            % Zero-arg constructor — produces the monic empty shell that
            % arx / ar populate in place (matlab_ident_arx / _ar via the
            % Lowering class-pinned dispatch).  Every RHS is a concretely
            % typed `ones`/`zeros`/scalar literal: arx allocates this via
            % the zero-arg form `idpoly__idpoly()`, which forces the
            % *generic* ctor body to be lowered, where bare untyped
            % parameters would mis-route the property setter.  Direct
            % `idpoly(A,B,…)` construction is a Tier-2 follow-on.
            obj.A = ones(1, 1);     % monic [1]
            obj.B = zeros(1, 1);
            obj.C = ones(1, 1);
            obj.D = ones(1, 1);
            obj.F = ones(1, 1);
            obj.nk            = zeros(1, 1);
            obj.Gram          = zeros(1, 1);
            obj.Ts            = 1;
            obj.NoiseVariance = 0;
            obj.Np            = 0;
            obj.Ns            = 0;
            obj.Lambda        = 0;
            obj.Fit           = 0;
        end
    end
end
