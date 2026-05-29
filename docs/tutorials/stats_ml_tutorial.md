# Statistics & Machine Learning Toolbox — Tutorial

A hand-coded Statistics-Toolbox subset built directly on the project's numeric base — no external dependency (no NumPy/BLAS, no MATLAB). It covers the descriptive → distribution → hypothesis-test → regression → unsupervised → supervised arc, with p-values computed from hand-coded t/F/χ² CDFs and a single generic `ClassificationModel` classdef backing every `fitc*` learner.

## Supported features

- **Descriptive:** `mean`, `median`, `std`, `iqr`, `range`, `skewness`, `kurtosis`, `prctile`, `cov`, `corr`.
- **Distributions:** `normrnd`, `exprnd`, `unifrnd`, `rng`, `makedist`, `fitdist` (`'Normal'`, `'Exponential'`), and the fitted-object queries `pdf` / `cdf` / `icdf` / `random`.
- **Hypothesis tests:** `ttest`, `ttest2`, `vartest2`, `ztest`, `kstest`, `ranksum`, `signrank`, `signtest`, `anova1` — all returning the full `[h,p,ci,stats]` multi-output.
- **Regression:** `regress`, `fitlm` (`LinearModel` with R²/adj-R²/RMSE + coefficient table), `fitglm` (logistic GLM via IRLS), `ridge`, `polyfit` / `polyval`, `predict`.
- **Unsupervised:** `pca`, `kmeans`, `pdist` / `pdist2` / `squareform`, `silhouette`, `tsne`.
- **Supervised:** `fitcknn`, `fitcnb`, `fitcdiscr`, `fitctree`, `fitcsvm`, `fitcecoc`, `predict`, `confusionmat`.
- **Ensembles & sequence models:** `fitcensemble` (bagged CART), `TreeBagger` (random forest), `bayesopt`, and the HMM family `hmmgenerate` / `hmmviterbi` / `hmmdecode` / `hmmtrain`.

## Build & run

```bash
build/matlabc -emit-llvm examples/stats_ml/iris_classify.m > /tmp/iris_classify.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/iris_classify.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/iris_classify
/tmp/iris_classify
```

Swap `iris_classify` for any other file under `examples/stats_ml/`.

## Worked examples

### Fisher-iris classification pipeline — HEADLINE  (`examples/stats_ml/iris_classify.m`)

The canonical end-to-end ML arc: summarise → reduce → cluster → classify → score. The 150×4 dataset is generated from the real Fisher-iris class means/spreads over the reproducible PRNG, so it is fully self-contained.

```matlab
X = [setosa; versicolor; virginica];
y = [ones(n,1); 2*ones(n,1); 3*ones(n,1)];

% ----- PCA: how much variance lives in the first two components -------
[coeff, score, latent, ts, explained] = pca(X);
fprintf('PCA explained : PC1 %.1f%%  PC2 %.1f%%\n', explained(1), explained(2));

% ----- unsupervised check: k-means into 3 clusters --------------------
idx = kmeans(X, 3);
fprintf('kmeans silhouette = %.3f\n', mean(silhouette(X, idx)));

% ----- supervised: multiclass SVM (ECOC), score on the data ----------
mdl = fitcecoc(X, y);
yp  = predict(mdl, X);
Cm  = confusionmat(y, yp);
acc = (Cm(1,1) + Cm(2,2) + Cm(3,3)) / (3*n);
fprintf('SVM accuracy   = %.1f%%\n', 100*acc);
```

`pca` runs a hand-coded symmetric Jacobi eigensolver and PC1 captures ≈90% of the variance; `kmeans` (Lloyd + k-means++) plus `silhouette` give the unsupervised sanity check; `fitcecoc` trains a one-vs-one linear SVM. Setosa separates perfectly while versicolor/virginica overlap, so the run lands at ≈95% accuracy — the genuine Fisher-iris behaviour.

### Fit a distribution  (`examples/stats_ml/fit_normal.m`)

The first-hour-with-data workflow: draw a Normal sample, summarise it, recover its parameters by maximum likelihood, then query the fitted object.

```matlab
data = normrnd(100, 15, 500, 1);            % true mean 100, sigma 15
pd = fitdist(data, 'Normal');
fprintf('fitdist(Normal): mu = %.2f, sigma = %.2f\n', pd.mu, pd.sigma);
fprintf('P(X <= 115)    = %.4f\n', cdf(pd, 115));
fprintf('95th percentile= %.2f\n', icdf(pd, 0.95));
fprintf('pdf at the mean= %.4f\n', pdf(pd, pd.mu));
```

`fitdist` returns a distribution object; the normal CDF uses libc `erf` and the inverse uses Acklam's rational approximation. Recovered parameters land near μ≈99.8, σ≈15.6.

### Hypothesis testing  (`examples/stats_ml/hypothesis_testing.m`)

Compare a control and treatment group with a two-sample t-test (full report), check the equal-variance assumption with an F-test, and cross-check nonparametrically.

```matlab
[h, p, ci, stats] = ttest2(control, treatment);
fprintf('  t-statistic           : %.3f\n', stats.tstat);
fprintf('  degrees of freedom    : %.0f\n', stats.df);
fprintf('  95%% CI on mean diff   : [%.2f, %.2f]\n', ci(1), ci(2));

[hv, pv] = vartest2(control, treatment);                 % F-test
[pr, hr] = ranksum(control, treatment);                  % Wilcoxon rank-sum
```

The `[h,p,ci,stats]` multi-output is split per-return; p-values come from regularized-incomplete-gamma/beta CDFs.

### Linear & logistic regression  (`linear_regression.m`, `glm_logistic.m`)

`fitlm` builds a `LinearModel` exposing `Rsquared`, `RsquaredAdj`, `RMSE`, `Beta`, and the `Coefficients` table (Estimate/SE/tStat/pValue), with `predict` on new rows:

```matlab
mdl = fitlm([x1 x2], y);
fprintf('R-squared = %.4f\n', mdl.Rsquared);
fprintf('Coef x1   = %.3f\n', mdl.Beta(2));
yhat = predict(mdl, [5 2]);
```

`glm_logistic.m` fits a binomial/logit GLM by IRLS with `fitglm` and predicts the class-1 probability via `predict`.

### Other examples (briefly)

- `exploratory_analysis.m` — location/spread/shape summary, `prctile` quartiles, and `corr`/`cov` of two related variables.
- `distribution_fitting.m` — `fitdist` Exponential + Normal by MLE, queried with `cdf`/`icdf`; plus a `polyfit`/`polyval` quadratic curve fit.
- `ensemble_classify.m` — single `fitctree` vs bagged `fitcensemble` vs random-forest `TreeBagger` on iris-like data.
- `hmm_markov.m` — the "occasionally dishonest casino" 2-state HMM: `hmmgenerate`, `hmmviterbi`, `hmmdecode`, and `hmmtrain` (Baum-Welch).
- `stats_tsne.m` — `tsne` embedding of three 4-D blobs into 2-D, verifying cluster separation.

## Limitations & carve-outs

- **Apps** (Classification/Regression Learner, Distribution Fitter) — the command-line `fitc*`/`fitdist` API is the whole surface.
- **Deep-learning-backed models** (`fitcnet`/`fitrnet`, LSTM/CNN feature extractors) — Deep Learning Toolbox dependency.
- **Simulink ML blocks** and **MATLAB-Coder codegen API** (`saveLearnerForCoder`).
- **Tall arrays / GPU / distributed** and **Python coexecution** — in-core matrix forms only.
- **Incremental learning**, **fairness/interpretability** (`lime`/`shapley`), **survival analysis** (`coxphfit`), **time-series forecasting**, the full **Gaussian-process** kernel library, and **design of experiments** — deferred follow-ons.

## See also

- Roadmap: [`stats_ml_toolbox_roadmap.md`](../stats_ml_toolbox_roadmap.md)
- Examples: `examples/stats_ml/`
