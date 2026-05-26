% Financial Toolbox Tier-3 — Portfolio classdef. The mean-variance
% optimisation kernel lives in runtime_finance.cpp (matlab_portfolio_*
% entries); this file is the umbrella the parser pulls in whenever a
% Portfolio constructor or method name appears in the user input.

% creditscorecard — logistic-regression credit model. Carries the
% training predictors X (N×p), the default flags Y (N×1, 0/1), and
% the fitted Beta ((p+1)×1, intercept first) after fitmodel. WoE/IV
% binning is a documented follow-on; this fits logistic regression
% on the raw predictors.
classdef creditscorecard
    properties
        X matrix
        Y matrix
        Beta matrix
    end
    methods
        function obj = creditscorecard(X, y)
            if nargin < 1, X = zeros(1, 1); end
            if nargin < 2, y = zeros(1, 1); end
            obj.X = X;
            obj.Y = y;
            obj.Beta = zeros(1, 1);
        end
    end
end

classdef Portfolio
    properties
        NumAssets         % scalar (count)
        AssetMean matrix  % N x 1 expected return vector
        AssetCovar matrix % N x N covariance matrix
        LowerBound matrix % N x 1 lower bounds (default 0)
        UpperBound matrix % N x 1 upper bounds (default 1)
        LowerBudget       % sum-of-weights min (default 1)
        UpperBudget       % sum-of-weights max (default 1)
        RiskFreeRate      % scalar (default 0)
        RiskKind          % 0 = mean-variance (dispatch discriminant)
    end
    methods
        function obj = Portfolio()
            % Empty Portfolio. Populate via setters or setAssetMoments.
            obj.NumAssets    = 0;
            obj.AssetMean    = zeros(1, 1);
            obj.AssetCovar   = zeros(1, 1);
            obj.LowerBound   = zeros(1, 1);
            obj.UpperBound   = zeros(1, 1);
            obj.LowerBudget  = 1;
            obj.UpperBudget  = 1;
            obj.RiskFreeRate = 0;
            obj.RiskKind     = 0;
        end
    end
end

% PortfolioCVaR — scenario-based Conditional Value-at-Risk optimization.
% Carries a Scenarios matrix (S x N) + a ProbabilityLevel (alpha). The
% shared estimateFrontier / estimatePortRisk / setDefaultConstraints
% method names route on RiskKind = 1 at runtime.
classdef PortfolioCVaR
    properties
        NumAssets
        Scenarios matrix        % S x N scenario return matrix
        ProbabilityLevel        % alpha (e.g. 0.95)
        LowerBound matrix
        UpperBound matrix
        LowerBudget
        UpperBudget
        RiskKind                % 1 = CVaR
    end
    methods
        function obj = PortfolioCVaR()
            obj.NumAssets        = 0;
            obj.Scenarios        = zeros(1, 1);
            obj.ProbabilityLevel = 0.95;
            obj.LowerBound       = zeros(1, 1);
            obj.UpperBound       = zeros(1, 1);
            obj.LowerBudget      = 1;
            obj.UpperBudget      = 1;
            obj.RiskKind         = 1;
        end
    end
end

% PortfolioMAD — scenario-based Mean-Absolute-Deviation optimization.
% Same scenario surface as PortfolioCVaR; RiskKind = 2.
classdef PortfolioMAD
    properties
        NumAssets
        Scenarios matrix
        LowerBound matrix
        UpperBound matrix
        LowerBudget
        UpperBudget
        RiskKind                % 2 = MAD
    end
    methods
        function obj = PortfolioMAD()
            obj.NumAssets   = 0;
            obj.Scenarios   = zeros(1, 1);
            obj.LowerBound  = zeros(1, 1);
            obj.UpperBound  = zeros(1, 1);
            obj.LowerBudget = 1;
            obj.UpperBudget = 1;
            obj.RiskKind    = 2;
        end
    end
end

% ---------------------------------------------------------------------
% Stochastic Differential Equation models (Tier-6). Each carries the
% model parameters + a ModelType discriminant; simByEuler / simBySolution
% read them at simulation time. 1-D models.
% ---------------------------------------------------------------------

% Geometric Brownian motion: dX = mu X dt + sigma X dW.
classdef gbm
    properties
        Drift; Sigma; StartState; Speed; Level; ModelType
    end
    methods
        function obj = gbm(mu, sigma, S0)
            if nargin < 3, S0 = 1; end
            obj.Drift = mu; obj.Sigma = sigma; obj.StartState = S0;
            obj.Speed = 0; obj.Level = 0; obj.ModelType = 1;
        end
    end
end

% Brownian motion (arithmetic): dX = mu dt + sigma dW.
classdef bm
    properties
        Drift; Sigma; StartState; Speed; Level; ModelType
    end
    methods
        function obj = bm(mu, sigma, X0)
            if nargin < 3, X0 = 0; end
            obj.Drift = mu; obj.Sigma = sigma; obj.StartState = X0;
            obj.Speed = 0; obj.Level = 0; obj.ModelType = 0;
        end
    end
end

% Cox-Ingersoll-Ross: dX = speed(level - X) dt + sigma sqrt(X) dW.
classdef cir
    properties
        Drift; Sigma; StartState; Speed; Level; ModelType
    end
    methods
        function obj = cir(speed, level, sigma, X0)
            if nargin < 4, X0 = level; end
            obj.Speed = speed; obj.Level = level; obj.Sigma = sigma;
            obj.StartState = X0; obj.Drift = 0; obj.ModelType = 2;
        end
    end
end

% Hull-White / Vasicek: dX = speed(level - X) dt + sigma dW.
classdef hwv
    properties
        Drift; Sigma; StartState; Speed; Level; ModelType
    end
    methods
        function obj = hwv(speed, level, sigma, X0)
            if nargin < 4, X0 = level; end
            obj.Speed = speed; obj.Level = level; obj.Sigma = sigma;
            obj.StartState = X0; obj.Drift = 0; obj.ModelType = 3;
        end
    end
end
