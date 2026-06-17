# Statistics and Machine Learning Toolbox Spec

## Purpose
Documents the shipped subset of the Statistics and Machine Learning Toolbox in the matlab_llvm compiler: hand-coded descriptive statistics and probability distributions, hypothesis tests and ANOVA, regression, PCA and clustering, classical classifiers and ensembles, and Bayesian optimization with hidden Markov models. Tiers 1-6 core are marked shipped (2026-05-20). (doc: docs/stats_ml_toolbox_roadmap.md) (src: runtime/toolbox/stats)

## Requirements

### Requirement: Descriptive statistics and probability distributions
The system SHALL provide descriptive reductions and parametric distribution functions/objects. (src: runtime/toolbox/stats/runtime_stats.cpp) (src: runtime/toolbox/stats/stats_classdefs.m)

#### Scenario: Summarize data and fit a distribution
- **WHEN** a program calls descriptive functions (`prctile`, `quantile`, `iqr`, `range`, `mode`, `skewness`, `kurtosis`, `geomean`, `harmmean`, `cov`, `corr`, `corrcoef`), distribution functions (`normpdf`/`normcdf`/`norminv`, `exppdf`/`expcdf`/`expinv`, `unifpdf`/`unifcdf`/`unifinv`, `normrnd`/`unifrnd`/`exprnd`), or builds a `ProbDistUnivParam` via `makedist`/`fitdist`
- **THEN** the system SHALL return the requested statistic, density/quantile/random value, or a fitted distribution object with `pdf`/`cdf`/`icdf`/`random` methods

### Requirement: Hypothesis tests and ANOVA
The system SHALL provide parametric and nonparametric tests and one-way ANOVA. (src: runtime/toolbox/stats/runtime_stats.cpp)

#### Scenario: Run a hypothesis test
- **WHEN** a program calls `ttest`, `ttest2`, `vartest2`, `ztest`, `kstest`, `ranksum`, `signrank`, `signtest`, or `anova1`
- **THEN** the system SHALL return the test decision and p-value, with confidence interval and statistics available via the result accessors

### Requirement: Regression models
The system SHALL provide linear, regularized, and generalized linear regression. (src: runtime/toolbox/stats/runtime_stats.cpp) (src: runtime/toolbox/stats/stats_classdefs.m)

#### Scenario: Fit and predict a regression model
- **WHEN** a program calls `regress`, `fitlm`, `ridge`, or `fitglm` and then `predict`
- **THEN** the system SHALL return a fitted model (e.g. `LinearModel` with coefficient table, R-squared, RMSE) and predictions for new inputs

### Requirement: PCA and clustering
The system SHALL provide dimensionality reduction, distance metrics, and clustering. (src: runtime/toolbox/stats/runtime_stats.cpp)

#### Scenario: Reduce, cluster, and evaluate
- **WHEN** a program calls `pca`, `pdist`/`pdist2`/`squareform`, `kmeans`, `silhouette`, or `tsne`
- **THEN** the system SHALL return principal components/scores, pairwise distances, cluster assignments (Lloyd + k-means++), silhouette values, or a 2-D embedding respectively

### Requirement: Classification and ensembles
The system SHALL provide classical classifiers, ensembles, and evaluation metrics. (src: runtime/toolbox/stats/runtime_stats.cpp) (src: runtime/toolbox/stats/stats_classdefs.m)

#### Scenario: Train, predict, and evaluate a classifier
- **WHEN** a program calls `fitcknn`, `fitcnb`, `fitcdiscr`, `fitctree`, `fitcsvm`, `fitcecoc`, or `fitensemble`, then `predict` and metrics (`confusionmat`, `accuracy`, `precision`, `recall`, `fscore`, `rocmetrics`, `aucroc`)
- **THEN** the system SHALL return a `ClassificationModel`, class predictions, and the requested evaluation metrics

### Requirement: Bayesian optimization and Markov models
The system SHALL provide Gaussian-process Bayesian optimization and hidden Markov model functions. (src: runtime/toolbox/stats/runtime_stats.cpp)

#### Scenario: Optimize and decode sequences
- **WHEN** a program calls `bayesopt`, or `hmmgenerate`/`hmmviterbi`/`hmmdecode`/`hmmtrain`
- **THEN** the system SHALL return the optimized point (GP surrogate + expected improvement), or the generated sequence / Viterbi path / likelihood / Baum-Welch re-estimated parameters
