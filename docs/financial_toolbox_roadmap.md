# Financial Toolbox — Compatibility Roadmap

> **STATUS (2026-05-26): Tiers 1–7 SHIPPED.** This document is the
> original *plan*; the as-built surface, headline demos, gating tests,
> and the deliberate carve-outs are tracked in
> [`financial_toolbox_status.md`](financial_toolbox_status.md). The
> "🔵 not started" / "clean slate" language below reflects the state
> *before* implementation — read it as the design rationale, not the
> current status. Shipped across PRs #57 (timetables) / #58 (T1–T3) /
> #59 (T4–T6) / #60 (T7). Runtime in
> `runtime/toolbox/finance/runtime_finance.cpp` +
> `finance_classdefs.m`; 20 gating tests in `test/Run/finance_*.m` +
> `portfolio_*.m`.

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Financial-Toolbox programs.

Source: *Financial Toolbox User's Guide* (R2026a, 15 chapters: Getting
Started · Performing Common Financial Tasks · Portfolio Analysis ·
Mean-Variance Portfolio Optimization Tools · CVaR Portfolio
Optimization Tools · MAD Portfolio Optimization Tools · Investment
Performance Metrics · Credit Risk Analysis · Regression with Missing
Data · Solving Sample Problems · Using Financial Timetables · Trading
Date Utilities · Technical Analysis · Stochastic Differential
Equations · Functions). The book is **3,510 pages** — the deepest
single toolbox we have planned to date, but most of it is *workflow
glue* (date arithmetic, performance metrics, technical indicators,
classdef carriers) on top of solvers we already ship.

This is the **single highest-leverage non-shipped toolbox** for the
quantitative-finance audience — "I have prices, weights, and a return
target, give me an efficient frontier and a backtest" is the canonical
MATLAB-in-finance workflow, and nearly the entire numeric base is
already in the runtime. Portfolio optimisation rides the shipped
`quadprog` / `linprog` / `fmincon` / `intlinprog` (Optim) plus the
generic `ClassificationModel`-style classdef carrier (Stats T5);
multivariate normal regression with missing data rides the shipped
`svd` / `mldivide` / `chol` plus the EM machinery; credit scorecards
ride the shipped `fitglm` (Stats T3 logistic IRLS); SDE Monte Carlo
rides the shipped RNG + linear algebra; performance metrics are
arithmetic. **No external dependency** (no QuantLib, no SciPy
finance, no statsmodels shim) — every core is hand-coded on top of
shipped linear-algebra + optimisation + statistics kernels.

The headline tracer-bullet (the gating example for the whole roadmap)
is [`examples/finance/efficient_frontier.m`](../examples/finance/efficient_frontier.m):
*the canonical Markowitz demo — load monthly returns for a handful of
assets, build `p = Portfolio('AssetMean', m, 'AssetCovar', C)`, call
`p = setDefaultConstraints(p)` then `pwgt = estimateFrontier(p, 20)`,
read `[risk, ret] = estimatePortMoments(p, pwgt)`, and
`plotFrontier(p)` against the data*. This exercises the `Portfolio`
classdef → solver-dispatch → frontier loop → plot arc end-to-end;
achieving it closes **Fin-Tier-3**. The companion
`examples/finance/black_litterman.m` (views-based posterior + frontier)
is the **Fin-Tier-5** tracer-bullet, and
`examples/finance/monte_carlo_blsprice.m` (GBM Euler under a vol
surface, compared against closed-form `blsprice`) is the
**Fin-Tier-6** tracer-bullet.

Companion docs: [`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md)
(every Portfolio*/PortfolioCVaR*/PortfolioMAD* frontier call leans on
shipped `quadprog` / `linprog` / `fmincon` / `intlinprog` / problem-
based API), [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)
(`creditscorecard`'s WoE + IV + `fitmodel` ride the shipped
`fitglm` logistic-IRLS; CAPM with missing data and ECM ride the same
mvn machinery as `fitdist`/MVN regression),
[`global_optim_toolbox_roadmap.md`](global_optim_toolbox_roadmap.md)
(MINLP cardinality-constrained portfolios reuse `gads_ga_core` IntCon
+ `surrogateopt`),
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) (technical
indicators reuse `filter` / `movmean` / `movstd`),
[`plotting.md`](plotting.md) (`plotFrontier`, `bolling`, `candle`,
`movavg` plots route through the Cairo backend),
[`mpc_toolbox_roadmap.md`](mpc_toolbox_roadmap.md) (KWIK active-set QP
already shipped — same solver fronts the MV Portfolio object on the
"in-house" path),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is *common financial tasks*: date arithmetic + cash-flow analysis +
  fixed-income pricing + technical indicators (`yearfrac`,
  `bndprice`, `irr`, `sma`, `bolling`, …). **Tier-2** is *investment
  performance metrics + equity-option Greeks*: `sharpe`, `inforatio`,
  `maxdrawdown`, `lpm`, `blsprice`/`blsdelta`/`blsimpv`. **Tier-3** is
  the **Mean-Variance `Portfolio` classdef** (the headline) over
  shipped `quadprog`/`fmincon` — sufficient to plot an efficient
  frontier, hit a target return, or maximise Sharpe. **Tier-4** is
  *credit risk* (`transprob`, `creditscorecard`, CDS bootstrap) plus
  *regression with missing data* (`mvnrmle`, `mvnrstd`, `ecmlsrmle`,
  CAPM with NaNs). **Tier-5** is `PortfolioCVaR` + `PortfolioMAD`
  (mirror the MV API on top of `linprog`/`fmincon`) plus the
  `backtestEngine` / `backtestStrategy` classdef pair. **Tier-6** is
  the **SDE Monte Carlo lane** (`bm`/`gbm`/`cev`/`cir`/`hwv`/`heston`/
  `sde` class hierarchy, `simByEuler`, stratified sampling, QMC) plus
  the Chapter-10 *Solving Sample Problems* canonical demos
  (Greek-neutral portfolios, term-structure swaps, hedge with
  Monte-Carlo). **Tier-7** is carve-down polish: financial timetables
  (`fints` → `timetable`), trading-date utilities, the Sharpe-ratio
  helper functions, Black-Litterman + risk-parity + Brinson
  attribution examples, mixed-integer / semi-continuous /
  cardinality-constrained portfolios.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1
  ~2 wk · T2 ~1 wk · T3 ~2 wk · T4 ~2 wk · T5 ~2 wk · T6 ~2.5 wk · T7
  ~2 wk (~13.5 wk full)**. Each tier is independently shippable and
  demoable; T1–T3 alone close the 80% everyday workflow (efficient
  frontier, bond pricing, Sharpe ratio); T4 + T5 close the
  credit-risk + backtest workflows that drive most institutional
  demand.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Everything below is 🔵 not started** — clean slate. There is no
  `Portfolio` / `creditscorecard` / `bndprice` / `blsprice` / `gbm` /
  `backtestEngine` / `sharpe` / `mvnrmle` in the runtime today. The
  deep shipped base (`quadprog`, `linprog`, `fmincon`, `intlinprog`,
  `lsqlin`, `fitglm`, `svd`, `mldivide`, `chol`, `eig`, `mvnrnd`,
  RNG, Stats `ClassificationModel`-style classdef carrier, plot via
  Cairo, `datetime` arithmetic) makes this **mostly composition, not
  net-new numerics**.

---

## 1. Already-shipped pieces this roadmap leans on

The Financial Toolbox sits on a **deep stack of shipped numerics**.
Counter to first impressions, **<15% of the roadmap is net-new
mathematics** — the rest is glue, classdef carriers, financial
conventions (day counts, coupon dates), and the dozens of small
arithmetic helpers (`sharpe`, `inforatio`, `maxdrawdown`, …) that
constitute the toolbox surface.

| Need                                | Shipped in                       |
|-------------------------------------|----------------------------------|
| Quadratic programming               | Optim `quadprog` (T2) + MPC KWIK |
| Linear programming                  | Optim `linprog` (T1)             |
| Mixed-integer LP                    | Optim `intlinprog` (T3)          |
| Mixed-integer NLP (cardinality)     | Global Optim `ga` `IntCon` (T6)  |
| Generic nonlinear constrained       | Optim `fmincon` (T2)             |
| Bounded linear LS                   | Optim `lsqlin` / `lsqnonneg`     |
| Logistic regression (scorecards)    | Stats `fitglm` (T3)              |
| Linear regression / CIs / R²        | Stats `fitlm` / `regress` (T3)   |
| Multivariate normal sampling        | Stats `mvnrnd` + `mvncdf` (T1)   |
| Cholesky / SVD / QR / `mldivide`    | LAPACK lane (Phases 1-3)         |
| `eig` symmetric + non-symmetric     | Stats T4 + CST T1                |
| RNG with seed (`rng`, `randn`)      | Stats T1                         |
| Generic classdef carrier            | Stats T5 `ClassificationModel`   |
| Handle ABI (anon objectives)        | Ident T5 EKF / Bayes-opt         |
| `datetime` + `calendarDuration`     | Datetime infra (any-shape lane)  |
| `timetable` indexing                | (gap — see §11)                  |
| Plot (line / scatter / fill)        | Cairo plot backend               |
| `interp1` linear / spline / pchip   | Curve-Fitting T4                 |
| `filter`, `movmean`, `movstd`       | DSP T1 / Signal Toolbox          |
| `ode45` / `ode23s`                  | Numerics (~80% shipped)          |
| `polyfit` / `polyval`               | Curve-Fitting T1                 |

What that table buys: **Tiers 1, 2, 3, 4 are essentially routing +
classdef wrappers + financial-convention bookkeeping**. The actually
new numerics are confined to:

- **§T4** *ECM (Expectation-Conditional-Maximisation) algorithm* for
  multivariate normal regression with missing data — pattern-
  partitioned likelihood, ~150 LOC over shipped `chol`/`mldivide`.
- **§T4** *CDS bootstrap* — iterative root-finding over hazard rates,
  ~100 LOC.
- **§T5** *Concentration-graph CVaR optimisation* via the LP
  formulation — a thin wrapper around `linprog`; the cutting-plane
  variant is a per-iteration LP loop. ~200 LOC.
- **§T6** *SDE simulation engines* — Euler/Milstein/quadratic-exp
  + correlated Brownian increments (`chol` of correlation matrix per
  step). ~400 LOC across the engine + the 6 standard model classes.
- **§T6** *Stratified sampling + Sobol/Halton quasi-MC* — classical
  algorithms, ~200 LOC.

Everything else (bond pricing, yields, Greek formulas, technical
indicators, performance metrics, transition probabilities, scorecard
binning) is closed-form arithmetic over shipped primitives.

---

## 2. Tier-1 — Common Financial Tasks (~2 wk)

**Goal.** Close Chapter 2 of the User's Guide *except* the equity-
derivatives + ML-statistical-arbitrage sections (which Tier-2 and
Tier-7 own). After Tier-1, a script that reads bond cash flows, prices
them, computes a yield curve, and chart-plots a moving-average +
Bollinger band over closing prices is end-to-end compilable.

### 2.1 Date arithmetic + calendar utilities

These are the **gating dependencies** for everything else — every
fixed-income function takes settle/maturity dates and a day-count
convention. We piggyback on shipped `datetime`, but the day-count
machinery and business-day calendar are new.

Functions to wire: `yearfrac(d1, d2, basis)` with all 13 basis codes
(0..13: actual/actual, 30/360 ISDA, 30/360 SIA, 30/360 PSA, 30/360 BMA,
actual/360, actual/365, actual/365 ISDA, BUS/252, …), `days360`,
`days360e`, `days360isda`, `days360psa`, `days365`, `daysact`,
`daysadd`, `daysdif`, `wrkdydif`, `busdate`, `isbusday`, `holidays`
(NYSE schedule baked in), `lbusdate`, `fbusdate`, `eomdate`, `lweekdate`,
`fweekdate`, `m2xdate`, `x2mdate`, `datenum`/`datestr` already exist —
just add the `'mm/dd/yyyy'`, `'dd-mmm-yyyy'` etc. format-spec
shortcuts.

### 2.2 Cash-flow / time-value-of-money

Pure arithmetic over the time-value formulas. `pvfix`, `fvfix`,
`pvvar`, `fvvar`, `irr`, `mirr`, `payper`, `payadv`, `payodd`,
`payuni`, `amortize`, `annurate`, `annuterm`, `nomrr`/`effrr`,
`taxedrr`/`xirr`.

### 2.3 Depreciation

`depfixdb`, `depgendb`, `depsoyd`, `depstln`, `depvardb` — six small
closed-form helpers.

### 2.4 Fixed-income pricing + yield

The big block. Pricing/yield/duration/convexity for plain bonds and
T-bills, plus the coupon-date / accrual machinery:

- Bond pricing/yield: `bndprice`, `bndyield`, `bndtotalreturn`
- Sensitivities: `bnddurp`, `bnddury`, `bndconvp`, `bndconvy`,
  `bndkrduration`, `bndspread`
- Coupon dates: `cpndatenq`, `cpndatep`, `cpndaten`, `cpndatesm`,
  `cpndaysn`, `cpndaysp`, `cpncount`, `cfdates`, `cfamounts`,
  `cfport`, `cfspread`, `cfyield`, `cftimes`
- Accrued interest: `accrfrac`
- Treasury bills: `prdisc`, `prtbill`, `ytbill`, `beytbill`,
  `discrate`, `prmat`, `ylddisc`, `yldmat`, `ytmat`, `tbl2bond`,
  `bond2tbl`, `tbillval01`, `tbillyield`, `tbillrepo`, `tbillprice`
- Yield curves (term-structure): `termfit`, `zbtprice`, `zbtyield`,
  `tr2bonds`, `bonds2tr`, `tfutbyprice`, `tfutbyyield`

### 2.5 Returns with negative prices

`negvolidx`, `negvolidx`, the `tick2ret` / `ret2tick` /
`isutirreg` lane — short section in the User's Guide, ~6 helpers.

### 2.6 Technical indicators

Three dozen indicator functions — almost all are `filter` /
`movmean` / `movstd` rearrangements. Shipped Signal Toolbox `filter`
+ DSP `movmean`/`movstd` cover the kernels; this tier just wires the
financial-convention wrappers.

Functions: `sma` (= `movmean`), `tsmovavg`, `medprice`, `typprice`,
`wclose`, `bolling`, `bollinger`, `ema`, `rsindex`, `macd`,
`stochosc`, `willad`, `willpctr`, `adline`, `chaikvolat`,
`chaikosc`, `posvolidx`, `negvolidx`, `onbalvol`, `volroc`,
`prcroc`, `volat`, `hhigh`, `llow`, `pvtrend`, `pointfig`,
`candle`, `highlow`, `priceandvol`, `volarea`, `kagi`, `linebreak`,
`renko`, `pointfig`, `movavg`, `fpctkd`, `spctkd`.

A handful (`candle`, `bolling`, `pointfig`, `kagi`, `linebreak`,
`renko`) are *chart* functions that route through the Cairo backend
— they reuse the `plot`/`fill` primitives already in `plotting.md`,
no new graphics primitive needed.

### 2.7 Life tables

Tiny section in Ch 2 — `lifetableconv`, `lifetablefit`,
`lifetablegen`. ~5 helpers, closed-form actuarial formulas.

### 2.8 Headline demo

`examples/finance/bond_yield_curve.m` — read a vector of bond
prices/coupons/maturities, bootstrap the zero curve via `zbtprice`,
plot it with the shipped Cairo backend. Closes Tier-1.

---

## 3. Tier-2 — Performance Metrics + Equity-Option Greeks (~1 wk)

### 3.1 Investment performance metrics (Ch 7)

Closed-form arithmetic. ~12 helpers:
- Risk-adjusted: `sharpe`, `sortino`, `inforatio`, `ret2tick`,
  `tick2ret`, `portalpha` (Jensen's α / M² / Treynor)
- Lower partial moments: `lpm`, `elpm` (sample + expected)
- Tracking error: `tracking` (just a wrapper around `std` of active
  returns)
- Drawdown: `maxdrawdown`, `emaxdrawdown`, `drawdown`. The expected-
  maximum-drawdown is a known closed-form approximation (Magdon-
  Ismail) — ~30 LOC.
- Risk-adjusted return: `ulcerindex`, `martinratio`,
  `painratio` (carry-down polish).

### 3.2 Equity-option pricing — Black-Scholes Greeks (§2.39 in PDF)

Closed-form. The whole Black-Scholes family is `erf`-based
(shipped):
- Pricing: `blsprice`, `blkprice` (forwards), `blsdelta`,
  `blsgamma`, `blsvega`, `blsrho`, `blstheta`, `blslambda`,
  `blspsi`
- Implied vol: `blsimpv`, `impvbybjs` — Newton-Raphson over
  `blsprice`, ~50 LOC including bracketing
- Binomial / trinomial trees: `binprice` (CRR), `crrprice`,
  `crrsens`, `eqpprice` (Equal Probability), `eqpsens`,
  `lrprice`/`lrsens` (Leisen-Reimer) — small recursive lattice
  evaluators

### 3.3 Headline demo

`examples/finance/option_greeks.m` — compute the full
delta/gamma/vega/rho/theta surface for a grid of strike/maturity and
plot via `surf`. Closes Tier-2.

---

## 4. Tier-3 — Mean-Variance `Portfolio` Object (~2 wk) — **HEADLINE**

This is the **headline tier**. Closes Ch 3 + Ch 4 of the User's
Guide. After Tier-3, the canonical efficient-frontier demo and the
asset-allocation case-study compile end-to-end.

### 4.1 `Portfolio` classdef

Carries: `NumAssets`, `AssetList`, `AssetMean`, `AssetCovar`,
`RiskFreeRate`, `LowerBudget`, `UpperBudget`, `LowerBound`,
`UpperBound`, `Aequality`/`bequality`, `Ainequality`/`binequality`,
`GroupMatrix`/`LowerGroup`/`UpperGroup`, `MinNumAssets`/
`MaxNumAssets`, `BoundType`, `TrackingError`/`TrackingPort`,
`Turnover`/`BuyTurnover`/`SellTurnover`, `BuyCost`/`SellCost`,
`InitPort`. Mirror the **alloc-then-populate constructor + class-
pinned dispatch** classdef pattern proven by `ClassificationModel`,
`tf`/`ss`, `idpoly`/`idss`, `LinearModel`.

The setter-API is *vast* but each method is a one-line property
write + validation guard (the validation is the same `mv_optim_transform`
input-bounds-check the User's Guide §4 documents):
- `setAssetMoments`, `setAssetList`, `setInitPort`, `setBudget`,
  `setBounds`, `setEquality`, `setInequality`, `addEquality`,
  `addInequality`, `setGroups`, `addGroups`, `setGroupRatio`,
  `setMinMaxNumAssets`, `setTurnover`, `setOneWayTurnover`,
  `setCosts`, `setTrackingError`, `setDefaultConstraints`,
  `setSolver`, `setSolverMINLP`

### 4.2 Solver dispatch

A `Portfolio` object hands off to:
- `quadprog` (shipped) for the unconstrained / bound-/group-/
  budget-/equality-/inequality-constrained MV problem
- `fmincon` (shipped) for the custom-objective path
  (`estimateCustomObjectivePortfolio`)
- `intlinprog` + branch-and-bound (shipped Optim T3) for the MINLP
  path (cardinality / semi-continuous bounds via `BoundType =
  'Conditional'`, `MinNumAssets`/`MaxNumAssets`)
- `linprog` (shipped) for the one-way / average-turnover linearised
  step

The MINLP solver-guideline path in the PDF (§4.140) is **already
implemented as the Global-Optim `gads_ga_core` IntCon lane** — the
roadmap reuses it.

### 4.3 Frontier methods

- `estimateFrontier(p, n)` — `n`-point sweep over target return
- `estimateFrontierByReturn(p, target)` — single-point
- `estimateFrontierByRisk(p, target)` — root-find in σ
- `estimateMaxSharpeRatio(p)` — closed-form transform to QP
- `estimateBounds(p)` — endpoint risks
- `estimatePortMoments(p, w)` — `[risk, ret] = (w'·Cw)^½, w·m`
- `estimatePortReturn(p, w)`, `estimatePortRisk(p, w)`,
  `estimatePortStd(p, w)`
- `estimatePortVaR(p, w)` (parametric Normal — closed-form)
- `checkFeasibility(p, w)`
- `plotFrontier(p, n)` — routes through the Cairo backend

### 4.4 Asset-moments estimation

- `estimateAssetMoments(p, AssetReturns)` (no-missing-data path)
- `estimateAssetMoments(p, AssetReturns, 'MissingData', true)`
  — uses the ECM machinery wired in **Tier-4 §5.1**

### 4.5 Headline demo

`examples/finance/efficient_frontier.m` — Markowitz on 5 assets,
plot the frontier, mark the max-Sharpe point. **Gating example for
the whole Financial Toolbox roadmap.**

Companion: `examples/finance/asset_allocation_case.m` —
mirrors the User's Guide §4.181 case study.

---

## 5. Tier-4 — Credit Risk + Regression with Missing Data (~2 wk)

### 5.1 Multivariate Normal Regression — ECM

The Ch-9 *Regression with Missing Data* numerical core. Functions:
- `mvnrmle(Data, Design)` — multivariate normal MLE without missing
  data (closed-form Σ̂ = X'X / n, β̂ = (X'X)⁻¹X'y) — ~50 LOC
- `mvnrmle(Data, Design, 'MissingData', true)` — ECM algorithm
  (pattern-partitioned likelihood, iterates E[β|θ_k] / E[Σ|θ_k]
  conditioning on observed components per row) — ~150 LOC over
  shipped `chol`/`mldivide`. *New numerical contribution.*
- `mvnrstd` (standard errors via observed Fisher information)
- `mvnrobj` (log-likelihood)
- `ecmlsrmle` / `ecmlsrobj` / `ecmlsrtd` — least-squares variant
  (same code path with Σ = σ²·I)
- `ecmmvnrmle` / `ecmmvnrobj` / `ecmmvnrtd` — covariance-weighted
  variant
- `ecmnmle` / `ecmnpdf` / `ecmnstd` — mean+covariance estimation
  with missing data
- Support: `nancov`, `nanvar`, `nanmean` already shipped via
  Stats T1

### 5.2 CAPM with missing data

Wraps `mvnrmle` with the `[α; β]` design matrix per asset; closed-
form sensitivities. ~30 LOC.

### 5.3 Transition probabilities

Closed-form counting / cohort / duration estimators:
- `transprob(data)` — cohort + duration paths
- `transprobbycohort`, `transprobbyduration`,
  `transprobbytotals`, `transprobgrouptotals`
- `transprobfromthresholds`, `transprobtothresholds`
- `transprobprep` — counting helper

Bootstrap CI via shipped Stats `bootstrp` (~Stats T1).

### 5.4 Credit scorecards (classdef)

The `creditscorecard` classdef carrier (~User's Guide §8.49). Methods:
- Construction: `creditscorecard(data, ...)`
- Automatic binning: `autobinning` (monotone-adjacent-pooling-
  algorithm — MAPA — for numeric; equal-frequency for categorical)
- WoE / IV: `bininfo`, `setbininfo`, `predictorinfo`,
  `plotbins` (Cairo)
- Modeling: `fitmodel` — rides shipped `fitglm` (Stats T3) logistic
  IRLS
- Scoring: `formatpoints`, `displaypoints`, `score`, `probdefault`
- Validation: `validatemodel` — CAP / ROC / KS / IV stats, all
  shipped Stats T4 utilities

The headline `examples/finance/credit_scorecard.m` is the §8.73
case-study demo.

### 5.5 Credit Default Swap (CDS)

Closed-form pricing + bootstrap calibration:
- `cdsbootstrap(ZeroData, MarketData, Settle)` — iterative hazard-
  rate bootstrap. *New numerical contribution* (~100 LOC root-find
  per maturity over `cdsspread`).
- `cdsprice`, `cdsspread`, `cdsoptprice`
- `cdsrpv01` — risky-PV01 helper

### 5.6 Carbon emission / Ollama integration (§9.40)

**Carve-out** — the PDF's "Extract Corporate Carbon Emission Target
Metrics Using Ollama" example depends on an external LLM endpoint
that is out of scope for the static-compile lane.

---

## 6. Tier-5 — CVaR + MAD Portfolios + Backtest Engine (~2 wk)

### 6.1 `PortfolioCVaR` classdef

Mirror of `Portfolio` (T3) over **scenarios** rather than
mean/covariance:
- `PortfolioCVaR(...)`, `setScenarios(p, ReturnScenarios)`,
  `setProbabilityLevel(p, alpha)`, `simulateNormalScenariosFromReturns`,
  `simulateNormalScenariosFromMoments`
- Frontier: `estimateFrontier`, `estimateFrontierByReturn`,
  `estimateFrontierByRisk`, `estimatePortVaR`, `estimatePortRisk`,
  `estimateMaxSharpeRatio`, `plotFrontier`
- Solver dispatch:
  - `'TrustRegionCP'` / `'ExtendedCP'` / `'cuttingplane'` — LP-based
    cutting plane (Rockafellar-Uryasev formulation); LP via shipped
    `linprog`. *New numerical contribution* (~150 LOC outer loop).
  - `'fmincon'` — direct nonlinear path (shipped)
  - MINLP (cardinality / semi-continuous) — reuses Optim
    `intlinprog` (shipped)

### 6.2 `PortfolioMAD` classdef

Mirror of `PortfolioCVaR` with Mean-Absolute-Deviation risk proxy
(Konno-Yamazaki LP formulation). Same scenario surface; risk proxy
is `E|w'r - E[w'r]|` → LP form. ~200 LOC including
`estimatePortStd` (returns sqrt of variance from scenarios).

### 6.3 Backtest engine

The §4.239 *Backtest Investment Strategies* infrastructure:
- `backtestStrategy(name, rebalanceFcn, ...)` classdef carrier —
  function-handle rebalance callback (reuses shipped Ident T5 +
  Bayes-opt handle-ABI machinery)
- `backtestEngine(strategies)` classdef — `runBacktest(eng,
  AssetReturns, Signals)`
- Reports: `equityCurve(eng)`, `summary(eng)`, `plotEquity(eng)`,
  `plotPositions(eng)`
- Brinson attribution: `brinsonAttribution(...)` — closed-form
  arithmetic decomposition (User's Guide §4.316–§4.324)

### 6.4 Headline demo

`examples/finance/black_litterman.m` — User's Guide §4.223 verbatim
(market-cap-weighted prior + view matrix → posterior moments →
`Portfolio` frontier). Closes Tier-5.

Companion: `examples/finance/backtest_strategies.m` — the §4.252
"with-trading-signals" demo.

---

## 7. Tier-6 — Stochastic Differential Equations (~2.5 wk)

The Chapter-14 *SDE Monte Carlo* lane. Self-contained — no
dependency on T3/T5. Could be slotted in any order.

### 7.1 SDE class hierarchy

Mirror MathWorks' inheritance chain as a flat-classdef family
sharing a `ModelType` discriminant (same pattern as Stats T5
`ClassificationModel`):

```
sdeddo  — drift-diffusion only           (ModelType 0)
sdeld   — linear-drift                   (ModelType 1)
bm      — Brownian motion                (ModelType 2)
gbm     — geometric Brownian motion      (ModelType 3)
cev     — constant-elasticity-of-variance(ModelType 4)
sdemrd  — mean-reverting drift           (ModelType 5)
cir     — Cox-Ingersoll-Ross             (ModelType 6)
hwv     — Hull-White / Vasicek           (ModelType 7)
heston  — Heston stochastic volatility   (ModelType 8)
sde     — general (user drift/diffusion handles)
```

Each carries: drift parameters (`Mu`, `Theta`, `Alpha`, …), diffusion
parameters (`Sigma`, `Eta`, …), correlation matrix, start state,
start time. Constructors validate dimension consistency.

### 7.2 Simulation engines

The `simulate` / `simByEuler` / `simBySolution` / `simByMilstein` /
`simByQuadExp` (Heston-specific) methods — one engine, dispatched
on `ModelType`. *New numerical contribution* (~400 LOC):

- Euler-Maruyama scheme over `NSteps` per `NPeriods`, `NTrials`
  paths
- Correlated Brownian increments via shipped `chol` of correlation
  matrix
- `simBySolution` closed-form transition for `bm`/`gbm`/`hwv` (no
  discretisation error)
- `simByMilstein` for `gbm`/`cev`/`cir`
- `simByQuadExp` for Heston (Andersen QE scheme)
- `interpolate(t, X, ...)` for path interpolation

### 7.3 Stratified sampling + Quasi-Monte Carlo

- Stratified sampling (Latin-hypercube + brownian-bridge variance
  reduction)
- Sobol / Halton low-discrepancy sequences (classical generators,
  ~150 LOC each)
- `qrandstream` / `qrandset` / `scramble` carriers

### 7.4 Solving sample problems — Chapter 10

The Ch-10 demos that anchor T1–T6:
- §10.2 *Sensitivity of bond prices to interest rates*
- §10.6 *Bond portfolio for hedging duration and convexity*
- §10.9 *Bond prices and yield curve parallel shifts*
- §10.14 *Greek-neutral portfolios of European stock options*
- §10.18 *Term-structure analysis and interest-rate swaps*
- §10.22 *Plotting an efficient frontier using portopt*
- §10.30 *Bond portfolio optimisation using Portfolio object*
- §10.49 *Hedge using Monte Carlo simulation*

Each becomes a gated test in `runtime/toolbox/finance/test/`.

### 7.5 Headline demo

`examples/finance/monte_carlo_blsprice.m` — simulate 10,000 GBM
paths for a vanilla European call under Black-Scholes, compare the
sample mean discounted payoff to the closed-form `blsprice` from
Tier-2. Achieves <1% relative error at 10k paths and is bit-exact
to MathWorks reference at fixed `rng(0)`. **Gating example for
Tier-6.**

Companion: `examples/finance/cir_short_rate.m` — Vasicek/CIR short-
rate simulation, fit a zero curve.

---

## 8. Tier-7 — Polish + carve-down items (~2 wk)

### 8.1 Financial Timetables (Ch 11)

The **dependency-heavy** chapter. Requires the matlab_timetable
infrastructure that **is not yet shipped** in the runtime (datetime
exists; `timetable` indexing does not). Defer the chapter wholesale
until the timetable lane lands as a separate cross-toolbox
prerequisite. The User's Guide §11 even acknowledges this is just
a migration shim for the legacy `fints` object.

### 8.2 Trading Date Utilities (Ch 12)

`uicalendar` / `cfui` — *GUI* dialogs. **Carve out** (matches the
project-wide carve-out for App-Designer / Live-Editor surfaces).
The non-GUI date helpers ship in Tier-1.

### 8.3 Advanced portfolio examples

- §4.232 *Portfolio optimisation using factor models* — needs Stats
  `pca` + Optim `quadprog`, both shipped. Wire as `examples/finance/
  factor_model_portfolio.m`.
- §4.265 *Portfolio optimisation using social performance measure*
  — ESG screening; small.
- §4.287 *Risk-budgeting portfolio* + §4.297 *Hierarchical risk
  parity* — reuses Stats T4 `linkage`/agglomerative clustering
  (currently a Stats carve-down — promote it here).
- §4.403 *Mixed-integer mean-variance portfolio optimisation* —
  rides Tier-3 §4.2 MINLP path; demo only.

### 8.4 Reinforcement-learning examples

- §4.380 *Multiperiod GBM via RL* — depends on RL Toolbox.
  **Carve out** until RL Toolbox roadmap exists.
- §4.408 *Deep RL for optimal trade execution* — same. **Carve out.**

### 8.5 Deep-Learning examples

- §4.303 *Backtest strategies using deep learning* — depends on DL
  Toolbox. **Carve out.**
- §2.78 *Backtest deep-learning model for algorithmic trading of
  limit-order-book data* — same. **Carve out.**
- §2.48–§2.78 *Machine Learning for Statistical Arbitrage I-III* —
  these chain Statistics + Deep Learning + financial-timetable
  operations. **Carve out** until DL + timetable lanes land.

### 8.6 Misc carve-outs

- Parallel-computing Monte Carlo (§14.109) — depends on parfor
  scope expansion; carve out beyond shipped parfor capture lane
  (memory note: [Parfor capture beyond v1](parfor_capture_phase3.md))
- Volatility modelling for soft commodities (§14.89) — needs
  multi-factor GBM; trace as a Tier-7 polish demo.
- Robust portfolio with ellipsoidal uncertainty (§4.352) — needs
  SOCP via `coneprog` (shipped) — small.
- Convex denoising vs factor model (§4.395) — combines Stats `pca`
  and shipped `cov`/`chol`; small.

---

## 9. Execution order

A natural cadence that minimises mid-tier dependency stalls:

1. **T1 §2.1 date arithmetic** (~3 sessions) — unblocks everything
   downstream
2. **T1 §2.2–§2.3 cash flows + depreciation** (~2 sessions)
3. **T1 §2.4 fixed-income pricing** (~5 sessions) — the big block
4. **T1 §2.5–§2.7 returns + indicators + life tables** (~2 sessions)
5. **T1 §2.8 Tier-1 headline demo `bond_yield_curve.m`**
   — *first shippable green*
6. **T2 §3.1 performance metrics** (~3 sessions)
7. **T2 §3.2 Black-Scholes Greeks** (~2 sessions)
8. **T2 §3.3 Tier-2 headline demo `option_greeks.m`**
9. **T3 §4.1–§4.4 `Portfolio` classdef + solver dispatch +
   frontier + moments** (~8 sessions)
10. **T3 §4.5 Tier-3 headline demo `efficient_frontier.m` →
    PR / merge → README badge bump → run-tests++**
11. **T4 §5.1–§5.2 mvnrmle / ECM / CAPM** (~4 sessions)
12. **T4 §5.3–§5.4 transprob + creditscorecard** (~5 sessions)
13. **T4 §5.5 CDS bootstrap** (~2 sessions)
14. **T5 §6.1–§6.2 PortfolioCVaR + PortfolioMAD** (~6 sessions)
15. **T5 §6.3 backtestEngine** (~3 sessions)
16. **T5 §6.4 Tier-5 headline demo `black_litterman.m`**
17. **T6 §7.1–§7.2 SDE classdefs + Euler/Milstein** (~6 sessions)
18. **T6 §7.3 stratified + QMC** (~3 sessions)
19. **T6 §7.4 Chapter-10 demos**
20. **T6 §7.5 Tier-6 headline demo `monte_carlo_blsprice.m`**
21. **T7 carve-downs and polish**

Total: ~13.5 wk for a full end-to-end Financial Toolbox lane;
**T1–T3 alone (the 80% workflow) is ~5 wk** and unlocks the iconic
demo.

---

## 10. Layout in the repo

```
runtime/toolbox/finance/
├── runtime_finance_dates.cpp         (T1.1)
├── runtime_finance_cashflow.cpp      (T1.2 + T1.3)
├── runtime_finance_bond.cpp          (T1.4)
├── runtime_finance_tbill.cpp         (T1.4)
├── runtime_finance_yieldcurve.cpp    (T1.4)
├── runtime_finance_indicators.cpp    (T1.6)
├── runtime_finance_lifetable.cpp     (T1.7)
├── runtime_finance_perfmetrics.cpp   (T2.1)
├── runtime_finance_options.cpp       (T2.2)
├── runtime_finance_portfolio.cpp     (T3 — MV solver dispatch)
├── runtime_finance_portfolio_cvar.cpp(T5.1)
├── runtime_finance_portfolio_mad.cpp (T5.2)
├── runtime_finance_credit.cpp        (T4.3 + T4.5 CDS)
├── runtime_finance_scorecard.cpp     (T4.4)
├── runtime_finance_mvnreg.cpp        (T4.1 ECM)
├── runtime_finance_backtest.cpp      (T5.3)
├── runtime_finance_sde.cpp           (T6.1 + T6.2)
├── runtime_finance_qmc.cpp           (T6.3)
└── finance_classdefs.m               (Portfolio*, creditscorecard,
                                        backtestStrategy, gbm/bm/...)

examples/finance/
├── bond_yield_curve.m                (T1 headline)
├── option_greeks.m                   (T2 headline)
├── efficient_frontier.m              (T3 headline — overall)
├── asset_allocation_case.m
├── credit_scorecard.m                (T4 headline)
├── capm_with_missing.m
├── cds_bootstrap.m
├── black_litterman.m                 (T5 headline)
├── backtest_strategies.m
├── monte_carlo_blsprice.m            (T6 headline)
└── cir_short_rate.m

runtime/toolbox/finance/test/
├── (one gating .m per headline + per Ch-10 sample problem)
```

This mirrors the `runtime/toolbox/{ident,images,stats,dsp,gads,…}/`
layout already in tree.

---

## 11. Known gaps and risks

- **`timetable` infrastructure is not shipped.** Ch 11 of the
  Financial Toolbox User's Guide is essentially a tutorial on
  finance-flavoured `timetable` slicing/joining. The roadmap
  defers it to a cross-toolbox prerequisite (own roadmap entry
  pending) — the rest of the toolbox does not need timetable;
  `datetime` + matrix arithmetic is sufficient.
- **`fints` is documented as deprecated.** The PDF Ch 11 §11.1
  is "Convert Financial Time Series Objects (fints) to Timetables".
  Skip `fints` entirely — go straight to the timetable target when
  that lane lands.
- **Solver guarantees.** The User's Guide §4.140 promises specific
  exit-flag values from `quadprog`/`fmincon`/`intlinprog`. The
  Optim Tier-1 carve-down already flagged exitflag/output multi-
  return as a follow-on; coordinate with that work so Portfolio
  frontier methods return matching diagnostics.
- **MISO/MIQP cardinality solver.** `Portfolio` with
  `setMinMaxNumAssets` triggers a mixed-integer QP. The shipped
  Optim `intlinprog` is integer-LP only; the MIQP path needs to
  reuse the Global-Optim `gads_ga_core` IntCon lane (proven in
  Stats T6 bayesopt) — verify before Tier-3 §4.2.
- **RNG determinism for SDE Monte Carlo.** `rng(seed, 'twister')`
  is shipped (Stats T1). The SDE engines need to consume RNG
  *deterministically* across (NTrials × NPeriods × NSteps × dim) —
  follow the Bayes-opt `OutputBufferEx`-style pattern to pre-
  generate increments in a known order.
- **Bond callable + amortising schedules.** Ch 2 §2.4 documents
  but the *amortising callable* path needs a tree pricer. Carve
  out of T1 (mark as Tier-7 polish) — vanilla bullet bonds
  cover >95% of the demo surface.
- **Plotting backends.** `bolling`, `candle`, `kagi`, `linebreak`,
  `pointfig`, `renko` need OHLC chart primitives. The Cairo
  backend's `fill` + `line` cover all of these; add a thin
  `ohlc_bar` helper. No new graphics primitive needed.

---

## 12. Why this toolbox now

The user has now shipped 19 toolboxes (DSP being the most recent —
see [DSP progress](dsp_toolbox_progress.md)). Of the documented
remaining priorities ([Next toolboxes ranking](next_toolboxes_ranking.md)
ranks Curve Fitting > Computer Vision > Sensor Fusion), Curve Fitting
already shipped 2026-05-23. The **Financial Toolbox is the largest
unclaimed compute-heavy classdef surface** — every demo MathWorks
shows in their MATLAB-in-finance pitch (efficient frontier, CAPM,
credit scorecards, GBM Monte Carlo, Black-Scholes, backtest) is a
direct beneficiary, and the **prerequisite stack is already shipped**.
This is a "rebuild the demo surface on top of finished numerics"
roadmap, not an "invent the numerics" roadmap — net new mathematics
is confined to ECM, CDS bootstrap, CVaR cutting-plane, Heston QE,
and Sobol/Halton (collectively ~1 kLOC over a 13.5-wk total).

The audience overlap with the existing user persona (quant /
financial-engineer / risk-manager — see the User's Guide §1.3
"Expected Users" page, lifted verbatim) is exceptionally tight: it
hits the same "I have data, give me a model" workflow that drove
Stats/ML, Curve Fitting, Optim, and Ident to completion.
