# Statistics and Machine Learning Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Statistics-and-Machine-Learning-Toolbox
programs.

Source: *Statistics and Machine Learning Toolbox User's Guide* (R2026a,
34 chapters: Getting Started · Organizing Data · Descriptive Statistics ·
Statistical Visualization · Probability Distributions · Gaussian
Processes · Random Number Generation · Hypothesis Tests · ANOVA ·
Bayesian Optimization · Parametric Regression · Generalized Linear
Models · Nonlinear Regression · Time Series Forecasting · Survival
Analysis · Multivariate Methods · Cluster Analysis · Parametric
Classification · Nonparametric Supervised Learning · Decision Trees ·
Discriminant Analysis · Naive Bayes · Classification Learner · Regression
Learner · Support Vector Machines · Fairness · Interpretability ·
Incremental Learning · Markov Models · Design of Experiments · Code
Generation · Machine Learning in Simulink · ML Pipelines · Functions).

This is the **single biggest user-reach gap** in the project — "I have
data" is the universal MATLAB entry point, and today the runtime can't
fit a distribution, run a t-test, do PCA, or train a classifier. The
classical-statistics + shallow-ML core needs **no Deep Learning
dependency** and reuses the shipped Optimization Toolbox (MLE / SVM dual
/ GLM IRLS via `fminunc` / `quadprog` / `fmincon`), the matrix kernel
(`eig` / `svd` / `chol` / `mldivide` for PCA / regression / discriminant
analysis), and the shipped PRNG.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/stats/iris_classify.m`](../examples/stats/iris_classify.m):
*the canonical Fisher-iris pipeline — load the 150×4 dataset, summarise
it (`mean`/`std`/`corr`), reduce with `pca`, cluster with `kmeans`, then
train + score an SVM classifier (`fitcsvm` + `predict`) with a confusion
matrix*.  This exercises the descriptive → unsupervised → supervised arc
end-to-end; achieving it is what closes **Stats-Tier-5**.

Companion docs: [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)
(MLE / SVM / GLM lean on the shipped solvers),
[`global_optim_toolbox_roadmap.md`](global_optim_toolbox_roadmap.md)
(Bayesian optimization shares the surrogate machinery),
[`ident_toolbox_roadmap.md`](ident_toolbox_roadmap.md) (regression /
ARX overlap), [`plotting.md`](plotting.md) (statistical visualization),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the descriptive-statistics + probability-distribution + RNG core
  (`mean`/`var`/`median`/`quantile`/`corr`/`cov` matrix forms +
  `normpdf`/`normcdf`/`norminv` & friends + `makedist`/`pdf`/`cdf`/`random`
  + the distribution RNGs). **Tier-2** is hypothesis tests + ANOVA
  (`ttest`/`ttest2`/`kstest`/`anova1`/`ranksum`). **Tier-3** is
  regression (`fitlm` / `regress` / `polyfit`-CI / ridge / lasso /
  `fitglm` IRLS / `fitnlm`). **Tier-4** is unsupervised learning (`pca` /
  `kmeans` / linkage-`cluster` / `gmdistribution`). **Tier-5** closes the
  headline — supervised classification (`fitcsvm` / `fitctree` /
  `fitcknn` / `fitcnb` / `fitcdiscr` + `predict` + `confusionmat` +
  `crossval`). **Tier-6** is ensembles + Bayesian optimization +
  carve-down polish (`fitcensemble` / `TreeBagger` / `bayesopt`).
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: T1 ~3 wk,
  T2 ~1.5 wk, T3 ~3 wk, T4 ~2 wk, T5 ~3 wk, T6 ~3 wk (~15 wk full). This
  is the **largest** single-toolbox roadmap — but each tier is
  independently shippable and demoable.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Nothing ML-specific is shipped — every row below is 🔵.** A handful
  of descriptive primitives already exist (`mean`/`std`/`var`/`median`,
  `rand`/`randn`/`randi`, `rng`, `qfunc`/`erfc`, `corrcoef` is partial) —
  noted per row.
- **Data container**: most of this toolbox is **table-centric** (`fitlm`
  takes a `table` + a Wilkinson formula). The runtime **already ships
  `table`** (see `feature_status.md` §3) — Tier-3 leans on it. SISO
  matrix forms (`X`, `y` numeric matrices) are the Tier-1/3/5 default;
  the table+formula form is layered on where it matters.
- **Model-object pattern**: every fitted model is a classdef descriptor
  (`LinearModel`, `GeneralizedLinearModel`, `ClassificationSVM`,
  `ClassificationTree`, `ProbDistUnivParam`, …) carrying fitted params +
  a `predict` method — the **exact** alloc-then-populate + class-pinned
  dispatch pattern proven by `idss`/`idpoly` (Ident), `tf`/`ss` (CST),
  `mpc` (MPC). Auto-prepend `stats_classdefs.m` via the prelude tables.
- **No external dependencies**: matching the project's hand-coded
  precedent — **no LIBSVM, no LAPACK beyond the shipped kernel, no
  scikit-learn parity shim.** SVM dual via the shipped `quadprog`; GLM
  via hand-coded IRLS on `mldivide`; trees via hand-coded CART; PCA via
  the shipped `svd`/`eig`.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Stats/ML code yet)

| Group | Surface (already shipped) | Location | How Stats/ML uses it |
|---|---|---|---|
| Reductions | `mean`, `std`, `var`, `median`, `sum`, `prod`, `min`, `max` (matrix + dim forms) | `runtime/matlab_runtime.cpp` | Descriptive statistics core (Tier-1). |
| Special functions | `erf`/`erfc`, `qfunc`, `gamma`/`gammaln`, `beta` (check) | `matlab_runtime.cpp` | Normal/χ²/t/F CDFs (Tier-1); GLM link functions (Tier-3). |
| PRNG | `matlab_rng_state` (xorshift) + `rand`/`randn`/`randi`/`randperm` + `rng(seed)` | `runtime_comm.cpp`, `matlab_runtime.cpp` | Every distribution RNG, bootstrap, k-fold split, k-means++ init, ensemble bagging. |
| Dense linear algebra | `mldivide`, `qr`, `chol`, `eig` (sym + non-sym), `svd`, `pinv`, `inv` | `runtime/matlab_runtime.cpp` | OLS regression, PCA (`svd`/`eig`), LDA (generalized eig), Mahalanobis (`chol`), GLM normal equations. |
| Optim solvers | `fminunc` (BFGS), `fmincon`, `quadprog` (convex QP), `lsqnonlin` (LM), `lsqlin` | `runtime/toolbox/optim/runtime_optim.cpp` | MLE (`fminunc` on neg-loglik), **SVM dual** (`quadprog`), GLM IRLS fallback, `fitnlm` (LM), ridge/lasso (`lsqlin` / coordinate descent). |
| Function-handle ABI | `void *fn_p` → `matlab_mat*(*)(matlab_mat*)` + `LowerAnonCalls` retyping | `runtime_optim.cpp` | Custom-pdf MLE (`mle(@pdf)`), `fitnlm` model handles, `bayesopt` objective. |
| `table` container | `table` / column access / `T.Var` / row ops | `matlab_runtime.cpp` (see feature_status §3) | The `fitlm(tbl, 'y ~ x1 + x2')` form (Tier-3). |
| `categorical` | `categorical` arrays + categories | `matlab_runtime.cpp` | Class labels for classifiers; grouping variables for ANOVA / `grpstats`. |
| Sorting / search | `sort`, `unique`, `histcounts`, `accumarray` | `matlab_runtime.cpp` | Empirical CDF, histograms, tree split enumeration, confusion matrix, mode. |
| Classdef plumbing | `matlab_obj_new` / `_set_*`, kwarg-ctor sugar, class-pinned dispatch, REPL persist | `lib/MLIR/Lowering.cpp`, `runtime_debug.cpp` | Every fitted-model + distribution-object descriptor. |
| Plotting | Cairo `histogram`/`scatter`/`plot`/`bar`/`boxplot`(partial) | `runtime/plot/` | Statistical visualization (Tier-1.5 / cross-tier). |
| Sym / autodiff | SymPP analytic gradients | `lib/Sym/` | Optional analytic MLE gradients; `bayesopt` acquisition. |

**Net assessment**: the *numeric base* (linear algebra, optimisation, PRNG,
special functions, containers) is shipped. The genuinely new code is
(a) the **distribution library** (~25 distributions × pdf/cdf/inv/rnd/fit),
(b) the **hypothesis-test battery**, (c) the **regression model classes** +
IRLS/lasso, (d) the **clustering algorithms**, and (e) the **classifier
training cores** (SVM SMO/dual-QP, CART, kNN, naive Bayes, LDA). Each is a
self-contained hand-coded routine over the shipped base.

---

## 2. Tier-1 — Descriptive statistics + probability distributions + RNG 🔵

Goal: the everyday data-summary + distribution surface. Most of MATLAB's
"first hour with data" lives here.

| # | Surface | Algorithm / notes | Runtime entry |
|---|---|---|---|
| 1.1 | descriptive reductions | `mean`/`std`/`var`/`median` (✅ shipped) + `mode`/`range`/`iqr`/`prctile`/`quantile`/`skewness`/`kurtosis`/`geomean`/`harmmean`/`trimmean` | `matlab_stats_*` |
| 1.2 | covariance / correlation | `cov`, `corr` (Pearson/Spearman/Kendall), `corrcoef`, `partialcorr` | `matlab_stats_cov` / `_corr` |
| 1.3 | grouped stats | `grpstats`, `tabulate`, `crosstab`, `accumarray`-backed group reductions | `matlab_stats_grpstats` |
| 1.4 | normal-family pdf/cdf/inv | `normpdf`/`normcdf`/`norminv`, `tpdf`/`tcdf`/`tinv`, `chi2*`, `fpdf`/`fcdf`/`finv` (via shipped `erf`/`gammainc`/`betainc`) | `matlab_stats_normpdf` … |
| 1.5 | discrete + continuous family | `binopdf`/`poisspdf`/`geopdf`/`unifpdf`/`exppdf`/`gampdf`/`wblpdf`/`betapdf`/`lognpdf`/`raylpdf` + cdf/inv each | `matlab_stats_*pdf/cdf/inv` |
| 1.6 | distribution RNGs | `normrnd`/`unifrnd`/`exprnd`/`poissrnd`/`binornd`/`gamrnd`/`wblrnd`/`mvnrnd`/`randsample` (all on the shipped PRNG, `rng`-reproducible) | `matlab_stats_*rnd` |
| 1.7 | distribution objects | `makedist('Normal', …)` → `ProbDistUnivParam` classdef + `pdf`/`cdf`/`icdf`/`random`/`mean`/`std`/`truncate` methods | `stats_classdefs.m` |
| 1.8 | distribution fitting | `fitdist(x, 'Normal')` (MLE via closed-form or `fminunc`), `mle(x, 'pdf', @f)`, `histfit`, `ksdensity` (kernel density) | `matlab_stats_fitdist` / `mle` / `ksdensity` |
| 1.9 | statistical viz | `boxplot`, `histfit`, `qqplot`, `cdfplot`, `normplot`, `ecdf`, `scatterhist` (Cairo) | `runtime/plot/` |

**Headline-within-tier**: UG "Fit a Distribution" — `fitdist` a Weibull
to lifetime data, overlay `histfit`, read the MLE parameters + CIs.

**Compile/Execute wiring**: new `runtime/toolbox/stats/runtime_stats.cpp`
+ `stats_classdefs.m`; register names in `Resolver.cpp`; `pde_table`
loose-match entries in `LowerTensorOps.cpp`; prelude trigger set for
`makedist`/`fitdist`/`ProbDistUnivParam`.

---

## 3. Tier-2 — Hypothesis tests + ANOVA 🔵

Goal: the inferential-statistics battery — the second pillar of classical
stats.

| # | Surface | Algorithm / notes | Runtime entry |
|---|---|---|---|
| 2.1 | t-tests | `ttest` (one-sample/paired), `ttest2` (two-sample, pooled + Welch), `[h, p, ci, stats]` | `matlab_stats_ttest` / `_ttest2` |
| 2.2 | variance / distribution tests | `vartest`/`vartest2`/`vartestn`, `ztest`, `kstest`/`kstest2` (KS), `lillietest`, `jbtest`, `adtest`, `chi2gof` | `matlab_stats_*test` |
| 2.3 | nonparametric tests | `ranksum` (Mann-Whitney), `signrank` (Wilcoxon), `signtest`, `kruskalwallis`, `friedman` | `matlab_stats_ranksum` … |
| 2.4 | correlation test | `corr` p-values, `[r, p] = corrcoef(...)` | reuse 1.2 + t-dist |
| 2.5 | one-way ANOVA | `anova1(X, group)` → F-stat + p + table; `multcompare` (Tukey HSD) | `matlab_stats_anova1` |
| 2.6 | N-way / repeated ANOVA | `anovan`, `anova` object, `ranova` | `matlab_stats_anovan` |
| 2.7 | power / sample size | `sampsizepwr` | `matlab_stats_sampsizepwr` |

**Headline-within-tier**: UG "Hypothesis Testing with Two Samples" —
`ttest2` + `vartest2` + `ranksum` on a two-group dataset with the full
`[h, p, ci, stats]` decision report.

---

## 4. Tier-3 — Regression (linear / GLM / nonlinear / regularized) 🔵

Goal: the regression workhorses, both matrix-form and table+formula-form.
Strong overlap with the shipped System Identification ARX/LS machinery.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `regress(y, X)` | OLS via QR; `[b, bint, r, rint, stats]` (R², F, p) | `qr`/`mldivide` |
| 3.2 | `fitlm(X, y)` / `fitlm(tbl, formula)` | `LinearModel` object: coefficient table (Estimate/SE/tStat/pValue), R²/adjR², ANOVA, residuals, `predict`/`plotResiduals`. Wilkinson-notation formula parser (`y ~ x1 + x2*x3`). | `qr`, `table`, t/F dist |
| 3.3 | `stepwiselm` / `robustfit` | Stepwise term selection (AIC/p-value); robust IRLS (bisquare). | IRLS |
| 3.4 | `ridge` / `lasso` | Ridge `(XᵀX+λI)⁻¹Xᵀy` (the shipped Ident T6 pattern); lasso via coordinate descent + cross-validated `lambda` path; elastic net. | `mldivide`, CD loop |
| 3.5 | `fitglm` | `GeneralizedLinearModel`: IRLS over the exponential family (logit/probit/log/identity links; binomial/Poisson/gamma). | IRLS on `mldivide` |
| 3.6 | `fitnlm` | `NonLinearModel`: Levenberg-Marquardt over a user model handle; coefficient CIs from the Jacobian. | `lsqnonlin` |
| 3.7 | `predict` / `feval` | Prediction + CIs/PIs on any fitted model. | — |
| 3.8 | model utilities | `coefCI`, `anova(mdl)`, `plotResiduals`, `Rsquared`, `mdl.Coefficients` | classdef |

**Headline-within-tier**: UG "Linear Regression Workflow" — `fitlm` on a
table with interaction terms, read the coefficient table, `plotResiduals`,
`predict` on new data with confidence intervals.

---

## 5. Tier-4 — Unsupervised learning (PCA + clustering) 🔵

Goal: dimensionality reduction + clustering — the exploratory-ML pillar.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `pca(X)` | Principal components via `svd` of the centered data; `[coeff, score, latent, ~, explained]`. | `svd` |
| 4.2 | `pcacov` / `factoran` | PCA from a covariance matrix; factor analysis (ML factor extraction). | `eig`, `fminunc` |
| 4.3 | `kmeans(X, k)` | Lloyd's algorithm + k-means++ init; `[idx, C, sumd, D]`; replicates. | PRNG, `mldivide` |
| 4.4 | `linkage` + `cluster` | Hierarchical agglomerative clustering (single/complete/average/ward); `dendrogram`, `cophenet`, `clusterdata`. | distance matrix |
| 4.5 | `gmdistribution` / `fitgmdist` | Gaussian mixture via EM; `cluster`/`posterior`/`pdf` methods. | `chol`, EM loop |
| 4.6 | `pdist` / `pdist2` / `squareform` | Pairwise distances (euclidean/cityblock/cosine/mahalanobis/…). | matrix kernel |
| 4.7 | `dbscan` / `evalclusters` / `silhouette` | Density clustering; cluster-count evaluation (silhouette / CH / gap). | distances |
| 4.8 | `mdscale` / `cmdscale` / `tsne` | Multidimensional scaling; t-SNE (carve-down candidate — heavier). | `eig` |

**Headline-within-tier**: UG "Cluster Analysis" — `pca` to 2-D then
`kmeans` + `silhouette` on the iris data, or `fitgmdist` soft clustering.

---

## 6. Tier-5 — Supervised classification (closes the headline) 🔵

Goal: the shallow-ML classifier suite — the most-requested ML surface.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | `fitcsvm(X, y)` | `ClassificationSVM`: SMO or dual-QP (`quadprog`) over the kernel matrix (linear/RBF/polynomial); soft-margin C; `predict`/`score`. | **`quadprog`** |
| 5.2 | `fitctree(X, y)` / `fitrtree` | `ClassificationTree`/`RegressionTree`: hand-coded CART (Gini/deviance split, pruning, surrogate splits); `predict`/`view`. | sort, accumarray |
| 5.3 | `fitcknn(X, y)` | `ClassificationKNN`: k-nearest-neighbour with `pdist2`-backed search; distance weighting. | `pdist2`, sort |
| 5.4 | `fitcnb(X, y)` | `ClassificationNaiveBayes`: Gaussian / kernel / multinomial class-conditionals. | Tier-1 distributions |
| 5.5 | `fitcdiscr(X, y)` | `ClassificationDiscriminant`: linear/quadratic discriminant (pooled vs per-class covariance, Mahalanobis). | `chol`, generalized `eig` |
| 5.6 | `fitclinear` / `fitcecoc` | Linear classifier (logistic/SVM hinge via `fminunc`); error-correcting output codes for multiclass. | `fminunc` |
| 5.7 | scoring + validation | `predict`, `loss`, `confusionmat`, `crossval` / `cvpartition` (k-fold), `perfcurve` (ROC/AUC). | PRNG (folds) |
| 5.8 | regression learners | `fitrsvm`, `fitrlinear`, `fitrgp` (Gaussian-process regression) | `quadprog`, `chol` |

**🎯 Headline (closes Tier-5)**:
[`examples/stats/iris_classify.m`](../examples/stats/iris_classify.m) —
the full Fisher-iris pipeline: load → `mean`/`std`/`corr` summary → `pca`
to 2-D → `kmeans` (unsupervised check) → `fitcsvm` (one-vs-one ECOC) →
`predict` → `confusionmat` + accuracy. The descriptive → unsupervised →
supervised arc end-to-end.

---

## 7. Tier-6 — Ensembles + Bayesian optimization + carve-down polish 🔵

Goal: the advanced-ML layer + the cross-tier hyperparameter optimizer +
the deferred-options sweep.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `fitcensemble` / `fitrensemble` | Boosting (AdaBoost/LogitBoost/GentleBoost) + bagging over the Tier-5 tree learner. | Tier-5 trees, PRNG |
| 6.2 | `TreeBagger` | Random forest (bootstrap + random feature subset per split); OOB error + feature importance. | Tier-5 trees |
| 6.3 | `bayesopt` / `hyperparameters` | Bayesian optimization: Gaussian-process surrogate + expected-improvement acquisition (shares machinery with GADS `surrogateopt`). `OptimizeHyperparameters` 'auto' on the Tier-5 fitters. | GP, `fmincon` |
| 6.4 | `pdist`-based anomaly | `isolationForest` / `ocsvm` (one-class SVM via `quadprog`). | `quadprog` |
| 6.5 | feature selection | `sequentialfs`, `fscnca` (NCA), `relieff`, `fsrftest`. | Tier-5 fitters |
| 6.6 | name-value options surface | `templateSVM`/`templateTree` + the full `fitc*` name-value set (KernelFunction, Standardize, ClassNames, Cost, Prior, …). | classdef |
| 6.7 | multi-return + model introspection | `[label, score, cost] = predict(...)`, `mdl.compact`, `loss` variants. | — |
| 6.8 | Markov models | `hmmtrain`/`hmmviterbi`/`hmmdecode`/`hmmgenerate` (Baum-Welch + Viterbi); `dtmc` discrete Markov chains. | matrix kernel |

**Headline-within-tier**: UG "Moving Towards Automating Model Selection
Using Bayesian Optimization" — `fitcsvm(..., 'OptimizeHyperparameters',
'auto')` driving `bayesopt` over the box-constraint + kernel scale.

---

## 8. Carve-outs (explicitly out of scope)

Matching the established roadmap discipline (GUI / Simulink / DL / big-data
deps are always carved):

- **Classification Learner + Regression Learner apps** + **Distribution
  Fitter app** (Chapters 23–24, "Model Data Using the Distribution Fitter
  App") — interactive GUIs; the command-line `fitc*` / `fitdist` API is
  the whole surface here.
- **Deep-learning-backed models** — neural-network classifiers/regressors
  (`fitcnet` / `fitrnet` beyond a shallow MLP), any LSTM / CNN feature
  extractor — Deep Learning Toolbox dependency.
- **Machine Learning in Simulink** (Chapter 32) + the Simulink predict
  blocks — needs Simulink.
- **Code Generation for Statistics and ML Functions** (Chapter 31,
  `saveLearnerForCoder` / `loadLearnerForCoder` / `%#codegen`) — MATLAB
  Coder surface; note the project's *own* emit-c/cpp lanes could later
  cover a subset, but the MathWorks codegen API is carved.
- **Tall arrays / big data / GPU** (`tall`, `gpuArray`, distributed) —
  out-of-core execution; the in-core matrix forms are the scope.
- **Python coexecution** (`Predict Responses Using Custom Python Model`)
  — external interpreter dependency.
- **Incremental learning** (Chapter 28, `incrementalClassificationLinear`
  / `fit` / `updateMetrics`) — streaming-model surface; a follow-on after
  the batch fitters land.
- **Fairness + Interpretability** (Chapters 26–27: `fairnessMetrics`,
  `lime` / `shapley` / `partialDependence`) — a polish layer on top of
  shipped models; deferred.
- **Survival analysis** (Chapter 15: `coxphfit` / `fitcox` / `ecdf`
  Kaplan-Meier) — a specialised vertical; deferred to a follow-on.
- **Time Series Forecasting** (Chapter 14) — overlaps the shipped System
  Identification Toolbox; deferred to avoid duplication.
- **Gaussian Processes as a standalone chapter** (Chapter 6) beyond the
  `fitrgp` regressor + the `bayesopt` surrogate — the full GP kernel
  library is a follow-on.
- **Design of Experiments** (Chapter 30: `fullfact` / `bbdesign` /
  `lhsdesign` / `candexch`) — a self-contained vertical; cheap follow-on
  but not core.

---

## 9. Dependency summary

```
Tier-1 (descriptive + distributions + RNG)  ── needs: reductions, erf/gammainc/betainc, PRNG, svd
   ├─ Tier-2 (hypothesis tests + ANOVA)      ── needs: Tier-1 distributions (t/F/χ²)
   ├─ Tier-3 (regression: lm/glm/nlm/lasso)  ── needs: qr/mldivide, table, lsqnonlin, IRLS
   ├─ Tier-4 (PCA + clustering)              ── needs: svd/eig, chol, PRNG, distances
   │     └─ Tier-5 (SVM/tree/knn/nb/lda)     ── needs: quadprog, CART, pdist2, Tier-1 dists, Tier-4 PCA  ◀── HEADLINE: iris_classify
   └─ Tier-6 (ensembles + bayesopt + Markov + polish)  ── needs: Tier-5 trees, GP surrogate (shared w/ GADS surrogateopt)
```

**Critical new build (not reusable from elsewhere)**: (1) the
distribution library (~25 families × pdf/cdf/inv/rnd/fit), (2) the
hypothesis-test battery, (3) the regression model classes + IRLS + lasso
coordinate descent, (4) the clustering algorithms (Lloyd / linkage / EM),
(5) the classifier training cores (SVM dual-QP, CART, kNN, naive Bayes,
LDA). Everything else (linear algebra, optimisation, PRNG, special
functions, `table`/`categorical`, classdefs, plotting) is shipped.
**No external dependency** — SVM via the shipped `quadprog`, GLM via
hand-coded IRLS, trees via hand-coded CART, PCA via the shipped `svd`.

**Sequencing note**: Tier-1 → Tier-4 → Tier-5 is the critical path to the
`iris_classify` headline (descriptive → PCA/kmeans → SVM). Tiers 2 and 3
are independent and can ship in parallel / any order. Tier-6 (esp.
`bayesopt`) shares the GP-surrogate machinery with the Global Optimization
Toolbox `surrogateopt` — build whichever lands first and reuse.
