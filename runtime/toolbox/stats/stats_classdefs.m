% Statistics and Machine Learning Toolbox — Tier-1 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `makedist`,
% `fitdist`, or `ProbDistUnivParam`.  The probability-distribution object
% is a thin parameter holder; all numeric work (pdf/cdf/icdf/random + the
% fitdist MLE) lives in runtime/toolbox/stats/runtime_stats.cpp.
%
% makedist(...) and fitdist(...) are NOT classdef methods — they are
% registered builtins dispatched in Lowering.cpp (makedist scans its
% name-value args; fitdist allocs this shell then calls the runtime
% populate step matlab_stats_fitdist_init).  pdf/cdf/icdf/random key on
% the object's class at runtime (REPL-safe), reading the fields below.
%
% DistCode: 1 = Normal, 2 = Exponential, 3 = Uniform.  Params ride as
% mu/sigma (Normal: mean/std; Exponential: mu=mean; Uniform: mu=lower,
% sigma=upper).

classdef ProbDistUnivParam
    properties
        DistCode
        mu
        sigma
    end
    methods
        function obj = ProbDistUnivParam()
            obj.DistCode = 1;
            obj.mu       = 0;
            obj.sigma    = 1;
        end
    end
end

% LinearModel / GeneralizedLinearModel (Tier-3) — fitted-regression holder.
% fitlm / fitglm allocate this shell and the runtime populates it
% (alloc-then-populate); predict(mdl, Xnew) is dispatched in Lowering.
% Coefficients is an (p+1)x4 matrix [Estimate SE tStat pValue]; ModelType
% 1 = OLS linear, 2 = logistic GLM.
classdef LinearModel
    properties
        Beta matrix
        Coefficients matrix
        Rsquared
        RsquaredAdj
        RMSE
        ModelType
        NumPred
    end
    methods
        function obj = LinearModel()
            obj.Beta         = zeros(1, 1);
            obj.Coefficients = zeros(1, 1);
            obj.Rsquared     = 0;
            obj.RsquaredAdj  = 0;
            obj.RMSE         = 0;
            obj.ModelType    = 1;
            obj.NumPred      = 0;
        end
    end
end

% ClassificationModel (Tier-5) — fitted-classifier holder.  fitcknn /
% fitcnb / fitcdiscr / fitctree / fitcsvm / fitcecoc allocate this shell
% and the runtime populates it (alloc-then-populate); predict(mdl, Xnew)
% is dispatched in Lowering.  ModelType 1=kNN, 2=naive Bayes, 3=LDA,
% 4=CART tree, 5=linear SVM, 6=ECOC.  kNN/NB/LDA carry the training set
% (Xtr/Ytr/Classes); tree/SVM/ECOC carry a compact Params matrix.
classdef ClassificationModel
    properties
        ModelType
        NumClass
        NumPred
        K
        Xtr matrix
        Ytr matrix
        Classes matrix
        Params matrix
        Offsets matrix
    end
    methods
        function obj = ClassificationModel()
            obj.ModelType = 1;
            obj.NumClass  = 0;
            obj.NumPred   = 0;
            obj.K         = 5;
            obj.Xtr       = zeros(1, 1);
            obj.Ytr       = zeros(1, 1);
            obj.Classes   = zeros(1, 1);
            obj.Params    = zeros(1, 1);
            obj.Offsets   = zeros(1, 1);
        end
    end
end
