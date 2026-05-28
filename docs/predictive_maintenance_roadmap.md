# Predictive Maintenance Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug) needs
to ship in order to faithfully **compile and execute** Predictive-Maintenance-
Toolbox programs — feature extraction from condition-monitoring data,
fault-state classification, and **remaining-useful-life (RUL) estimation** —
plus the headline **LSTM-for-RUL** demo that closes the loop with the Deep
Learning Toolbox.

Sources: *Predictive Maintenance Toolbox™ User's Guide* (R2026a — chapters on
Sensor-Data Analysis · Diagnostic Feature Designer · Condition Indicators ·
RUL Estimators · Decision Models · Deployment); the MATLAB
Predictive-Maintenance video series ("Identifying Condition Indicators",
"Feature Extraction Using Diagnostic Feature Designer", "Estimating Remaining
Useful Life") + the NASA CMAPSS turbofan reference dataset.

The toolbox is the **canonical industrial application of the LSTM tier** —
nearly every real-world PdM deep-learning demo (engine RUL, bearing wear,
HVAC fault detection) is an LSTM trained on time-series condition indicators.
That makes this roadmap the natural sequel to
[`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md) T4.

---

## 1. The architectural picture

Predictive Maintenance has a **fortunate composition story** for this project:
its forward kernel is *already shipped* by adjacent toolboxes, and only the
prognostics-specific estimators are genuinely new.

**What the project already ships that PdM composes on**:

- **Stats & ML toolbox** ([`global_optim_and_stats_ml_plans.md`](global_optim_and_stats_ml_plans.md)) —
  the time-domain condition indicators (`mean`/`std`/`skewness`/`kurtosis`/
  `var`/`median`/`mode`) **all already ship** as Stats T1; the
  classification step (`fitctree`/`fitcensemble`/`TreeBagger`/`fitcsvm`/
  `fitcecoc` + `confusionmat`) **already ships** as Stats T5/T6; one-way
  ANOVA + `bayesopt`-based HP tuning **already ship** as Stats T2/T6.
- **Signal Processing + DSP** ([`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md),
  [`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md)) — the frequency-domain
  feature kernel (`fft`/`pwelch`/`spectrum`/AR model PSD estimation) is the
  spectral-feature backbone; the time-frequency lane (`spectrogram`/`cwt`/
  scalogram) **already ships** via the Wavelet roadmap.
- **Wavelet** ([`wavelet_toolbox_progress.md` pointer in MEMORY.md](wavelet_toolbox_roadmap.md)) —
  `cwt`/`modwt`/EMD/VMD ship; PdM's "time-frequency condition indicator" lane
  rides this directly.
- **Deep Learning T4** ([`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md)) —
  **`lstm(X, H0, C0, W, R, b)` + BPTT ✅** is the headline composition layer:
  RUL prediction at the state-of-the-art is "feed multi-sensor time-series
  through an LSTM, regress on cycles-to-failure" — the kernel for this **is
  already on the project** as of the DL T4 shipment.
- **Curve Fitting + Optim** — `fit`/`lsqnonlin`/`fminunc` are the parameter
  estimators for the exponential / linear / Wiener-process degradation models.
- **Econometrics T3** (state-space) — the underlying Kalman lane for
  `covariateSurvivalModel` and Bayesian degradation tracking.

**What is genuinely new**: the **`ensemble` datastore container** (rows =
machine instances, columns = signals + fault codes + lifetime labels), the
**Diagnostic Feature Designer** headless API (feature enumeration + ANOVA
ranking over an ensemble), and the three **RUL estimator families** —
similarity, survival, and degradation — each with `fit` + `predictRUL`
methods over the ensemble container.

**No external dependency** — every estimator is hand-coded over the shipped
kernel.  The CMAPSS dataset is a `.mat` / CSV input the user provides.

---

## 2. Reading guide

- **Tier** = priority + dependency band.  Tiers 1–3 are the feature lane
  (compose almost entirely on Stats + Signal), Tier 4 is fault
  classification (rides Stats), Tier 5 is the **new** RUL estimator
  infrastructure, Tier 6 is the **LSTM-for-RUL headline** (rides DL T4 ✅).
- **Effort** in Phase-5.6.x cadence (one focused session ≈ a half-day; a
  "week" ≈ 5 sessions).  This is a **mid-sized roadmap (~6 wk total)** — small
  for a top-level toolbox because so much of its kernel is already shipped by
  Stats/Signal/DL.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.  All tiers are
  currently 🔵 not started; the *underlying compositional substrate* (Stats T1,
  Signal `pwelch`, DL T4 LSTM) is fully shipped.
- **No external dependencies** — matching project precedent.

---

## 3. Tier-1 — Condition indicators: time-domain features 🔵

*Per-window statistical summaries of a sensor signal — the entry point
into the diagnostic-feature pipeline.* Composes from Stats T1 ✅.

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| 1.1 | Statistical features | `mean`/`std`/`var`/`median`/`mode`/`rms`/`peak2peak`/`max`/`min` over a sliding or full-frame window | Stats T1 ✅ |
| 1.2 | Distribution-shape features | `skewness`/`kurtosis` + crest factor `peak/rms` + clearance factor `peak / mean(sqrt(|x|))²` + shape factor `rms / mean(|x|)` + impulse factor `peak / mean(|x|)` | Stats T1 ✅ |
| 1.3 | `pmEnsemble`-lite container | a thin table-like carrier: rows = machine instances, columns = `(signalName, sample-vector)` cells + a `FaultCode` / `Lifetime` label column; built on the shipped `matlab_table` / cell-array carrier | classdef carrier |
| 1.4 | `generateFeatures(pmEnsemble, "time")` | enumerate the §1.1+§1.2 features across every numeric column of every ensemble row; emit a 2-D feature matrix (rows = instances × windows, columns = feature names) suitable for Stats `fitctree`/`fitcensemble` | Stats T1 + 1.3 |

**Headline-within-tier**: `pm_timefeats_demo.m` — given a triplex-pump ensemble
with 30 healthy + 30 faulty `flow` measurements, generate the time-domain
feature matrix and box-plot `kurtosis` vs `FaultCode` — the boxes for
healthy/blocked-inlet **do not overlap**, proving the kurtosis condition
indicator alone discriminates that fault class.

---

## 4. Tier-2 — Condition indicators: frequency-domain + time-frequency features 🔵

*Spectral and time-frequency summaries — the discriminator for rotating-
machinery fault types (bearing wear, gear-mesh, blade-pass).* Composes from
Signal/DSP/Wavelet.

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| 2.1 | Spectral features | `fft`/`pwelch`/AR-model PSD → spectral peaks (top-K frequencies + amplitudes), spectral kurtosis, band power over [f_lo, f_hi], modal coefficients of an AR(p) fit | Signal + DSP |
| 2.2 | `generateFeatures(pmEnsemble, "spectral")` | enumerate §2.1 features across every numeric column; same row-major output as §1.4 | Signal + 1.3 |
| 2.3 | Time-frequency features | scalogram-derived band energies (max/mean/std of CWT magnitude per frequency band) + spectrogram band statistics | Wavelet + DSP |
| 2.4 | EMD / VMD energy features | per-IMF energy + per-IMF dominant frequency from the shipped Wavelet T6 `emd`/`vmd` | Wavelet ✅ |

**Headline-within-tier**: `pm_freqfeats_demo.m` — given a vibration signal
from a healthy + bearing-wear bearing, extract the top-5 spectral peaks via
`pwelch` and visualise: the **outer-race fault** signature at the
characteristic outer-race-defect frequency emerges as the dominant peak in
the faulty class but is absent in healthy — the spectral kurtosis feature
alone separates them by > 3σ.

---

## 5. Tier-3 — Diagnostic Feature Designer (headless API) 🔵

*The programmatic surface under the Diagnostic Feature Designer **App** (the
GUI is carved as a visual app, same precedent as DL's Deep Network Designer
in `deep_learning_toolbox_roadmap.md` §10).* The headless feature-design
pipeline is **fully in scope** because it is the engine that machine
learning rides on.

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| 3.1 | `pmFeatureSet(ensemble, signals, feats)` | enumerate the requested feature list (`"time"`/`"spectral"`/`"timefrequency"`/`"all"`) over the requested signal columns; return a feature-matrix + name vector + label-column | T1.4 + T2.2 + T2.3 |
| 3.2 | `rankFeatures(featMat, labels, "anova")` | one-way ANOVA F-statistic per column → normalised score → ranked feature list (the `"Rank Features"` button in the app); reuses Stats T2's anova1 ✅ | Stats T2 ✅ |
| 3.3 | `rankFeatures(…, "ttest")` / `"bhattacharyya"` / `"roc"` | additional ranking criteria for two-class problems; trivial scalar-per-feature loops | Stats T1 |
| 3.4 | `selectTopK(rankedFeats, K)` | drop the bottom-(N-K) features — the actionable knob the App labels "Select Features" | trivial |
| 3.5 | Feature-matrix export | round-trip the §3.1 output into the Stats `Classification Learner` input format (a numeric matrix + categorical label vector) | Stats T5/T6 ✅ |

**Headline-within-tier**: `pm_dfd_demo.m` — given the §4-headline pump
ensemble, run `pmFeatureSet(ens, "flow", "all")` → `rankFeatures` →
`selectTopK(5)` and emit a 5-column feature matrix; pipe to `fitcensemble`
(Stats T6 ✅) → confusion matrix shows **> 95% per-class accuracy** with
just 5 ANOVA-selected features out of the ~50 candidates.

---

## 6. Tier-4 — Fault classification 🔵 (rides Stats ML ✅)

*The classifier-training step — composes entirely on Stats T5/T6, no new
infrastructure.*

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| 4.1 | `pmClassify(featMat, labels, "tree")` | thin wrapper around `fitctree` with PdM-conventional defaults (CART, gini split, min-leaf-size=5) | Stats T5 ✅ |
| 4.2 | `pmClassify(…, "ensemble")` | wrapper around `fitcensemble` (bagged trees) — the PdM default for multi-class fault classification | Stats T6 ✅ |
| 4.3 | `pmClassify(…, "svm")` / `"knn"` / `"discriminant"` | wrappers around `fitcsvm`/`fitcknn`/`fitcdiscr` | Stats T5 ✅ |
| 4.4 | `confusionchart`-style table output | round-trip the predicted-vs-actual into the shipped `confusionmat` + ASCII pretty-printer | Stats T5 ✅ |
| 4.5 | Cross-validation harness | k-fold CV over the §4.1–4.3 pipelines using the shipped `cvpartition` | Stats T2 ✅ |

**Headline-within-tier**: `pm_classify_pump_faults.m` — train an ensemble
classifier on the §3-headline triplex-pump feature matrix; 5-fold CV
reports per-class precision/recall ≥ 90% across `{healthy, seal-leak,
blocked-inlet, worn-bearing}` and their pairwise combinations.

---

## 7. Tier-5 — RUL estimators 🔵 (the **new** infrastructure)

*Remaining-useful-life estimation — three model families, each fit to a
different data-availability regime.*

| # | Surface | Notes |
|---|---------|-------|
| 5.1 | **Similarity models** | `similarityModel(trainEnsemble)` / `hashSimilarityModel` / `pairwiseDifferenceModel` — fit to **run-to-failure** trajectories.  At `predictRUL(model, testTrajectory)`: find the K nearest training trajectories by trajectory-distance (`pdist2` over a window-sliding alignment), fit a probability distribution to their failure-times, return the **median** as the RUL estimate + the full distribution as the uncertainty band. |
| 5.2 | **Survival models** | `reliabilitySurvivalModel(failureTimes)` / `covariateSurvivalModel(failureTimes, covariates)` — fit to **failure-time-only** data (no degradation trajectory).  The first is a Weibull / lognormal / exponential fit (composes on Stats T1's `fitdist` ✅); the second is a Cox proportional-hazards model with linear covariate terms (regress on `pmFeatureSet` outputs).  `predictRUL` returns the conditional expected lifetime given the elapsed cycles. |
| 5.3 | **Degradation models** | `exponentialDegradationModel` / `linearDegradationModel` — fit to **degradation trajectories with a known safety threshold** (no failure data needed).  Linear: ordinary least squares on `condition(t) = a + b·t`; exponential: log-linear regression on `log(condition(t)) = a + b·t` (then back-transform).  `predictRUL` solves `condition(t* ) = threshold` for `t*` and returns `t* - t_current`. |
| 5.4 | `predictRUL` ABI | uniform return signature across the three families: `(rul, ciLow, ciHigh)` — the point estimate + 5%/95% bands.  CIs come from the §5.1 distribution percentiles, §5.2 hazard-function quantiles, and §5.3 parameter-uncertainty propagation respectively. |
| 5.5 | Trajectory-distance kernel | windowed Euclidean / DTW (`dtw` is a stretch follow-on; Euclidean is the default) over the multi-sensor condition-indicator stream; this is the inner loop §5.1 calls millions of times per `predictRUL` so it lives in `runtime_pm.cpp` as a tight C loop | Stats T4 `pdist2` ✅ |

**Headline-within-tier**: `pm_cmapss_rul.m` — train a similarity model on
80% of the NASA CMAPSS turbofan dataset (218 run-to-failure trajectories,
21 sensors), `predictRUL` on the held-out 20%; report median absolute
error < 25 cycles on engines with > 100 cycles remaining (matching the
published baseline). Decision tree of which model to use:

```
                                    ┌──── run-to-failure data?
                                    │       └─── yes ─→ §5.1 similarity
                                    │
                                    ├──── only failure times (no traj)?
                                    │       └─── yes ─→ §5.2 survival
                                    │
                                    └──── degradation traj + safety thresh?
                                            └─── yes ─→ §5.3 degradation
```

---

## 8. Tier-6 — LSTM-for-RUL ✅ (headline; rides DL T4 ✅)

*The deep-learning headline of the toolbox — train an LSTM regressor on the
multi-sensor time-series + cycle index → continuous RUL output.* **Composes
end-to-end on the shipped DL T4 functional `lstm`/`gru`** — no new
infrastructure beyond a thin example wrapper.

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| 6.1 | LSTM-RUL forward | per-timestep multi-sensor input → `lstm(X, H0, C0, W, R, b)` → FC head → scalar RUL prediction; the FC head is plain `mtimes` + bias add | DL T4 ✅ |
| 6.2 | LSTM-RUL training loop | the shipped DL T3.5 custom-loop pattern: `dlfeval`+`dlgradient`+`adamupdate` over MSE loss vs the cycles-to-failure label.  No new compiler infrastructure | DL T3.5 + T4 ✅ |
| 6.3 | Sequence-length normalisation | clip / pad each trajectory to a fixed window (the published baseline: last-30-cycles window); standard pre-processing | matrix kernel |
| 6.4 | Feature-engineered + LSTM hybrid | feed `pmFeatureSet` outputs (T3.1) as the LSTM inputs (rather than raw sensors) — the technique that wins on the CMAPSS benchmark | T3.1 + 6.2 |

**Headline-within-tier (the roadmap headline)**: `pm_lstm_cmapss_rul.m` —
train a 64-unit LSTM on CMAPSS FD001, predict RUL on the test set, plot
predicted-vs-true cycles-to-failure; reach RMSE < 18 cycles on the
"healthy regime" (engines with > 50 cycles remaining), matching the
published LSTM-RUL state-of-the-art baseline.  **This is the
deep-learning-meets-prognostics demo and the single most-watched PdM
workflow in the field.**

---

## 9. Status / wiring / examples / tests

### 9.1 Compile / Execute

- **Runtime**: `runtime/toolbox/pm/runtime_pm.cpp` (feature enumerators +
  ANOVA ranker + similarity-model trajectory matcher + survival/degradation
  fitters + `predictRUL` ABI) + `runtime/toolbox/pm/pm_classdefs.m` (the
  `similarityModel` / `reliabilitySurvivalModel` /
  `exponentialDegradationModel` / `pmEnsemble` classdefs).  Add to the
  strict no-C-cast list.
- **Wiring**: the six-place pattern (the
  [`navigation_toolbox_roadmap.md`](navigation_toolbox_roadmap.md) §8.1 /
  Robotics §8.1 map applies verbatim — `kToolboxDirs` ×2, prelude `Cls[]` +
  AOT `Names[]` + `findToolboxClassdef`, `Resolver.cpp`, `Lowering.cpp` ctor
  intercepts + arg-0-class method dispatch, `LowerTensorOps.cpp` pde_table,
  `run_tests.sh` + `run_sweep.sh`).  **Critical reuse-trap**: every raw
  `matlab_pm_*` runtime symbol emitted as a `call_builtin` callee MUST get a
  `pde_table` signature row.  Estimator constructors are classdef-ctor
  intercepts; `predictRUL`/`fit` are method dispatch on arg-0-class.
- **Backends**: LLVM JIT + native are primary.  `-emit-c`/`-emit-cpp` parity
  ports cleanly (no autodiff tape in this toolbox — the LSTM-RUL headline
  pushes that requirement into DL T2 which already has it).

### 9.2 Debug / REPL

A `similarityModel` persists across REPL inputs and renders its
training-trajectory count + fitted-distribution summary in the DAP
inspector; a paused `predictRUL` call shows the K nearest neighbours and
their failure-times.  `pmEnsemble` renders as a table with the
fault-code column highlighted.

### 9.3 Examples (`examples/pm/`)

| Example | Closes |
|---|---|
| `pm_timefeats_demo.m` | T1 — time-domain features + box-plot discriminability |
| `pm_freqfeats_demo.m` | T2 — spectral features + bearing-fault signature |
| `pm_dfd_demo.m` | T3 — feature designer headless: enumerate → rank → select-top-K |
| `pm_classify_pump_faults.m` | T4 — fault classifier on the triplex pump |
| `pm_cmapss_rul.m` | T5 — similarity-model RUL on NASA CMAPSS |
| `pm_lstm_cmapss_rul.m` | **T6 headline** — LSTM-for-RUL on CMAPSS, RMSE < 18 |

### 9.4 Tests (`test/Run/`)

`pm_{timefeats,freqfeats,rank,classify,similarity,survival,degradation,lstm}.m`
gating tests.  The CMAPSS-headline tests assert per-trajectory error bounds
(seeded `rng` for reproducibility, per the Navigation precedent).  Full
regression stays green; badge bumps to **26 toolboxes** on completion.

### 9.5 Effort summary

| Tier | Scope | Est. | New infra |
|---|---|---|---|
| T1 | time-domain condition indicators | ~2 sess | feature enumerators over `pmEnsemble` |
| T2 | spectral + time-frequency features | ~3 sess | spectral feature library |
| T3 | Diagnostic Feature Designer headless | ~3 sess | feature-set + ANOVA ranker |
| T4 | fault classification wrapper | ~1 sess | thin Stats-ML shim |
| T5 | RUL estimators | ~1.5 wk | **similarity/survival/degradation infra (keystone)** |
| T6 | LSTM-for-RUL headline | ~2 sess | example only (rides DL T4 ✅) |

**Total ~6 wk** — mid-sized for a top-level toolbox because Stats T1–T6 +
DL T4 + Signal `pwelch` + Wavelet `cwt` ship the entire compositional
substrate.  **T5 is the keystone** (~1.5 wk on its own) and the only tier
that requires meaningfully new runtime code.

---

## 10. Carve-outs (explicitly out of scope)

- **Diagnostic Feature Designer App** + **Classification Learner App** —
  GUI products; the programmatic surfaces (T3 + T4) ship, the visual apps do
  not.  Matches the project's pattern of "engine ships, app carves" (Deep
  Network Designer, Experiment Manager, Signal Analyzer App).
- **Simulink Predictive-Maintenance blocks** (`Predict Remaining Useful Life`,
  `Pareto Front`, the `Industrial Communication`-driven OPC UA sources) —
  the `mflowLink` lane is the project's block-diagram answer.
- **Live-streamed condition monitoring** (ThingSpeak / IoT / OPC UA /
  Modbus / MQTT live ingest) — file-based ensembles + offline batch
  prognostics ship; live-streaming-from-PLC pipelines are
  Industrial-Communication-Toolbox territory and carved.
- **Domain-specific PdM apps** — battery-state-of-health (`BatteryProductionStateOfHealthEstimator`,
  Battery-Toolbox dependency), Rotor-balancing, motor-fault-signature
  libraries — these belong to domain-specific peer toolboxes and carve.
- **The full RUL `Iterations` / online-update API** — `update(model,
  newData)` incremental refit; the static `fit` → `predictRUL` pipeline ships
  T5, the online lane is a Tier-7 stretch.
- **Anomaly detection** (`isolationForest`, `localOutlierFactor`,
  `oneClassSVM`, autoencoder-based) — overlaps the Stats-ML carve-outs
  (Stats T6 already carved isolationForest); the autoencoder lane rides DL
  T5.5 (carved as a follow-on there).  An explicit PdM "anomaly-flag"
  example over `oneClassSVM` is a future tier.
- **Digital twin / Simscape model-in-the-loop** for fault injection — fault
  data is provided as ensemble inputs (file-based); generating synthetic
  fault trajectories from a Simscape physical model is Simscape territory.

Companion docs:
[`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md) (the
T6 LSTM-for-RUL headline rides DL T4 ✅),
[`global_optim_and_stats_ml_plans.md`](global_optim_and_stats_ml_plans.md)
(Stats-ML kernel for T1 condition indicators + T3 ANOVA ranker + T4
classifiers + T5.2 survival fits),
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) /
[`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) (spectral-feature kernel
for T2),
[`wavelet_toolbox_progress.md` pointer](wavelet_toolbox_roadmap.md) (CWT +
EMD/VMD for T2.3/T2.4 time-frequency features),
[`embedded_coder_roadmap.md`](embedded_coder_roadmap.md) (the `mflowLink`
answer for Simulink PdM blocks),
[`feature_status.md`](feature_status.md).
