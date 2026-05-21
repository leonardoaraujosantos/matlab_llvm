# Statistics and Machine Learning Toolbox — examples

Hand-coded Statistics-Toolbox subset over the shipped numeric base (no
external dependency). See [`docs/stats_ml_toolbox_roadmap.md`](../../docs/stats_ml_toolbox_roadmap.md).

## Tier-1 (shipped) — descriptive stats · distributions · RNG · fit

Descriptive statistics, covariance/correlation, probability distributions
(pdf/cdf/inverse), distribution random-number generators, and the
distribution-object workflow (`makedist` / `fitdist` + `pdf`/`cdf`/`icdf`/
`random`).

| Example | User's Guide | Notes |
|---|---|---|
| [`fit_normal.m`](fit_normal.m) | *Fit a Distribution* | **Tier-1 headline.** Draw a 500-sample Normal (true μ=100, σ=15), summarise it (`mean`/`median`/`std`/`iqr`/`skewness`), recover the parameters by maximum likelihood with `fitdist(data,'Normal')` (μ≈99.8, σ≈15.6), then query the fitted object: `cdf(pd,115)`, `icdf(pd,0.95)`, `pdf(pd,pd.mu)`. |
| [`exploratory_analysis.m`](exploratory_analysis.m) | *Exploratory Analysis of Data* | Location/spread/shape summary + quartiles (`prctile` with a vector of percentiles), then `corr`/`cov` of two related variables. |
| [`distribution_fitting.m`](distribution_fitting.m) | *Curve Fitting & Distribution Fitting* · *Maximum Likelihood Estimation* | `fitdist` Exponential + Normal by MLE on lifetime data, query `cdf`/`icdf`; least-squares polynomial curve fit with `polyfit`/`polyval`. |

## Tier-2 (shipped) — hypothesis tests + ANOVA

| Example | User's Guide | Notes |
|---|---|---|
| [`hypothesis_testing.m`](hypothesis_testing.m) | *Hypothesis Testing* | Two-sample `ttest2` with the full `[h,p,ci,stats]` report (t-statistic, df, CI on the mean difference), the equal-variance `vartest2` F-test, and the nonparametric `ranksum` cross-check. |

### Surface covered
`ttest` (one-sample / paired) · `ttest2` (two-sample pooled) · `vartest2`
(F-test) · `ztest` · `kstest` (vs standard normal) · `ranksum`
(Mann-Whitney) · `signrank` (Wilcoxon) · `signtest` · `anova1` (one-way).
All return the MATLAB `[h,p,ci,stats]` / `[p,h,stats]` multi-output via a
per-output splitter; p-values use hand-coded t / F / χ² CDFs (regularized
incomplete gamma + beta).

## Tier-3 (shipped) — regression

| Example | User's Guide | Notes |
|---|---|---|
| [`linear_regression.m`](linear_regression.m) | *Linear Regression* / regression-performance assessment | `fitlm` multiple linear model: R²/adjusted-R²/RMSE, the coefficient table (Estimate/SE/tStat/pValue), and `predict` on new data. |
| [`glm_logistic.m`](glm_logistic.m) | *Generalized Linear Models* | Logistic regression (binomial response, logit link) fitted by IRLS with `fitglm`; `predict` the class-1 probability. |

### Surface covered
`regress` (OLS, explicit intercept column) · `fitlm` (`LinearModel`:
coefficient table, R²/adj-R²/RMSE, `predict`) · `fitglm` (logistic GLM via
IRLS) · `ridge` (centered ridge regression) · `predict`.  Design matrices
are built with bracket horizontal concatenation `[x1 x2]` (the concat of
column-vector variables now lowers via `matlab_horzcat`/`matlab_vertcat`).

## Tier-4 (shipped) — PCA + clustering

Covered: `pca` (`[coeff,score,latent,~,explained]` via a hand-coded
symmetric Jacobi eigensolver) · `kmeans` (`[idx,C,sumd,D]`, Lloyd +
k-means++) · `pdist2` / `pdist` / `squareform` (euclidean) · `silhouette`
(cluster-quality score).  Exercised inside the headline below.

## Tier-5 (shipped) — supervised classification (HEADLINE)

| Example | User's Guide | Notes |
|---|---|---|
| [`iris_classify.m`](iris_classify.m) | the Fisher-iris pipeline | **Toolbox headline.** descriptive → `pca` (PC1 ≈ 90% variance) → `kmeans` + `silhouette` (unsupervised check) → `fitcecoc` (one-vs-one linear SVM) → `predict` → `confusionmat` + accuracy.  Recovers the real Fisher-iris behaviour — setosa perfectly separated, versicolor/virginica overlapping → **≈95% accuracy** on a 150×4 dataset generated from the real class means. |

### Surface covered
`fitcknn` (k-NN) · `fitcnb` (Gaussian naive Bayes) · `fitcdiscr` (LDA,
pooled covariance) · `fitctree` (hand-coded CART, Gini splits) · `fitcsvm`
(binary linear SVM, squared-hinge) · `fitcecoc` (one-vs-one multiclass) ·
`predict` (runtime-dispatched on the model class, REPL-safe) ·
`confusionmat`.  k-NN/NB/LDA carry the training set and re-derive at
predict; the tree and SVM carry a compact parameter matrix.

### Surface covered

- **Descriptive**: `prctile`, `quantile`, `iqr`, `range`, `mode`,
  `skewness`, `kurtosis`, `geomean`, `harmmean` (column-wise on matrices,
  whole-vector on vectors — matching `var`/`median`).
- **Covariance / correlation**: `cov`, `corr` (Pearson), `corrcoef`.
- **Distributions** (pdf / cdf / inverse): Normal (`normpdf`/`normcdf`/
  `norminv`, with 1-arg standard-normal forms), Exponential
  (`exppdf`/`expcdf`/`expinv`), Uniform (`unifpdf`/`unifcdf`/`unifinv`).
  The normal CDF rides libc `erf`/`erfc`; the inverse normal is Acklam's
  rational approximation (≈1e-9).
- **RNG** (all `rng`-reproducible over the shared PRNG): `normrnd`,
  `unifrnd`, `exprnd`.
- **Distribution objects**: `makedist('Normal'/'Exponential'/'Uniform',
  …)`, `fitdist(x, dist)` (closed-form MLE), and the `ProbDistUnivParam`
  methods `pdf` / `cdf` / `icdf` / `random` (runtime-dispatched on the
  object's class, so they work in the REPL too).

## Tier-6 (shipped) — ensembles · Bayesian optimization · Markov models

| Example | User's Guide | Notes |
|---|---|---|
| [`ensemble_classify.m`](ensemble_classify.m) | *Ensemble Learning* | A single CART tree vs `fitcensemble` (bagged trees) vs `TreeBagger` (random forest = bootstrap + √p random feature subset per split) on iris-like data. |
| [`hmm_markov.m`](hmm_markov.m) | *Hidden Markov Models* | The "occasionally dishonest casino": `hmmgenerate` a sequence, recover the hidden states with `hmmviterbi`, score with forward-backward `hmmdecode`, and re-learn the model from data with `hmmtrain` (Baum-Welch). |

Also: `bayesopt(fun, lb, ub)` — Gaussian-process surrogate + expected-
improvement minimization of an expensive black box (functional form over
the objective-handle ABI).

### Carve-downs (documented follow-ons)

T1: wider distribution library (binomial/Poisson/gamma/Weibull/beta — the
incomplete gamma/beta needed for them now exist), `grpstats`/`crosstab`,
`ksdensity`/`histfit`, `mean(pd)`/`std(pd)`.  T2: `anovan`/`multcompare`,
`kruskalwallis`/`friedman`, `chi2gof`/`jbtest`, Welch `ttest2`.  T3:
Wilkinson-formula `fitlm(tbl,'y~x1+x2')`, `stepwiselm`/`robustfit`/`lasso`/
`fitnlm`, regression CIs.  T4: `linkage`/`cluster`, `gmdistribution`/EM,
`evalclusters` object, `dbscan`, `tsne`.  T5: RBF/poly SVM kernels (linear
shipped), `crossval`/`cvpartition`/`perfcurve`, `loss`, `[label,score]`
multi-output predict.  T6: boosting (`AdaBoost`/`LogitBoost`; bagging
shipped), OOB error + feature importance, `bayesopt` `optimizableVariable`/
results-object API + `OptimizeHyperparameters`, `isolationForest`, feature
selection (`fscnca`/`relieff`), `dtmc`.  See the roadmap.

**All six tier cores are shipped — this completes the toolbox subset.**
