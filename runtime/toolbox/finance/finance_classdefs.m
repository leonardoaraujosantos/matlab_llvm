% Financial Toolbox Tier-3 — Portfolio classdef. The mean-variance
% optimisation kernel lives in runtime_finance.cpp (matlab_portfolio_*
% entries); this file is the umbrella the parser pulls in whenever a
% Portfolio constructor or method name appears in the user input.

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
        end
    end
end
