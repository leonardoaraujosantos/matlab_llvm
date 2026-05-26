# Financial Toolbox — Implementation Status

Tracks what shipped against `docs/financial_toolbox_roadmap.md` (the
7-tier plan). Companion to the runtime in `runtime/toolbox/finance/`.

**Status: Tiers 1–7 implemented.** The everyday quant-finance surface —
dates, cash flows, bond pricing, performance metrics, Black-Scholes,
mean-variance / CVaR / MAD portfolios, credit risk, regression with
missing data, SDE Monte Carlo, Black-Litterman, risk parity — compiles,
runs, and is gated by stdout-golden tests.

Shipped across four PRs:
- **#57** — timetable lane (T7 §8.1: financial timetables)
- **#58** — T1 (dates/cash-flows/bonds/indicators) + T2 (perf metrics +
  Black-Scholes) + T3 (Portfolio classdef + frontier)
- **#59** — T4 (credit risk + ECM) + T5 (CVaR/MAD + backtest) + T6 (SDE
  Monte Carlo)
- **#60** — T7 (frontier-method gaps, Black-Litterman, risk parity)

## Shipped surface

### Tier-1 — Common Financial Tasks
- Dates: `yearfrac` (bases 0/1/2/3/6/12), `daysdif`, `daysadd`, `daysact`,
  `days360`, `days365`, `busdate`, `isbusday`, `eomdate`, `lweekdate`,
  `fweekdate`, `m2xdate`, `x2mdate`
- Cash flows: `pvfix`, `fvfix`, `pvvar`, `fvvar`, `irr`, `payper`,
  `amortize`, `nomrr`, `effrr`
- Depreciation: `depstln`, `depsoyd`, `depfixdb`
- Bonds: `bndprice`, `bndyield`, `bnddurp`, `bnddury`, `bndconvp`,
  `accrfrac`; T-bills `prdisc`, `prtbill`, `ytbill`, `beytbill`
- Returns + indicators: `tick2ret`, `ret2tick`, `sma`, `bolling`,
  `rsindex`
- (timetable-form `movavg`/`macd` shipped in #57)

### Tier-2 — Performance Metrics + Black-Scholes
- `sharpe`, `sortino`, `inforatio`, `tracking`, `maxdrawdown`, `lpm`,
  `portalpha`
- `blsprice`, `blsdelta`, `blsgamma`, `blsvega`, `blsrho`, `blstheta`,
  `blslambda`, `blsimpv`

### Tier-3 — Mean-Variance Portfolio
- `Portfolio` classdef + `setAssetMoments`, `setBounds`, `setBudget`,
  `setDefaultConstraints`
- `estimateFrontier`, `estimateFrontierByReturn`, `estimateFrontierByRisk`,
  `estimateMaxSharpeRatio`, `estimateBounds`, `estimatePortFrontier`,
  `estimatePortMoments`, `estimatePortReturn`, `estimatePortRisk`,
  `estimateAssetMoments`, `plotFrontier`

### Tier-4 — Credit Risk + Regression with Missing Data
- `ecmnmle`, `ecmncov` (ECM for missing data), `mvnrmle`, `capm`
- `transprob`, `cdsbootstrap`, `cdsspread`, `cdsprice`
- `creditscorecard` classdef + `fitmodel`, `probdefault`, `score`

### Tier-5 — CVaR + MAD Portfolios + Backtest
- `PortfolioCVaR` + `setScenarios`, `setProbabilityLevel`,
  `estimatePortVaR` (shared frontier/risk methods dispatch on RiskKind)
- `PortfolioMAD`
- `backtest`, `backtestSummary`

### Tier-6 — SDE Monte Carlo
- `bm`, `gbm`, `cir`, `hwv` classdefs + `simByEuler`, `simBySolution`
- `haltonseq` (QMC), `optpricemc`

### Tier-7 — Polish
- Black-Litterman: `blacklitterman`
- Risk parity / budgeting: `riskparity`, `riskbudget`, `riskcontribution`
- Financial timetables (#57)

## Headline demos (`examples/finance/`)
- `efficient_frontier.m` — Markowitz frontier + max-Sharpe (T3)
- `credit_scorecard.m` — logistic scorecard (T4)
- `backtest_strategies.m` — rebalanced vs buy-and-hold (T5)
- `monte_carlo_blsprice.m` — GBM MC vs closed-form Black-Scholes (T6)
- `black_litterman.m` — views-tilted posterior into a frontier (T7)
- `using_timetables_in_finance.m` — full doc-page timetable workflow (#57)

## Carve-outs (deliberately not implemented)

| Area | Reason |
|------|--------|
| `uicalendar` / `cfui` trading-date GUI | App-Designer surface; project-wide GUI carve-out |
| RL examples (DRL trade execution, multiperiod GBM via RL) | depends on Reinforcement Learning Toolbox |
| DL examples (LOB backtest, deep-learning strategies, ML stat-arb) | depends on Deep Learning Toolbox |
| Heston 2-D stochastic vol; correlated multi-asset SDE baskets; Sobol; stratified sampling | SDE follow-ons beyond the 1-D Euler/QMC core |
| Handle-driven `backtestStrategy`/`backtestEngine` callbacks; Brinson attribution | needs anon-function-handle ABI through a per-period callback |
| LP/quadprog-backed CVaR/MAD | uses projected subgradient (no LP-solver dependency) |
| Full WoE/IV `creditscorecard` autobinning | scorecard fits logistic regression on raw predictors |
| Robust portfolio (ellipsoidal uncertainty, SOCP) | `coneprog` is shipped; the wrapper is a follow-on |
| Factor-model portfolio construction | `pca` is shipped; the covariance-reconstruction wrapper is a follow-on |
| Mixed-integer / cardinality-constrained portfolios | needs the Global-Optim `gads_ga_core` IntCon path wired to Portfolio |
| Bond date-based API (settle/maturity/basis) + `cfdates`/`cfamounts` | bond pricing uses the periods+freq form |
| Yield-curve bootstrap (`zbtprice`/`zbtyield`/`termfit`) | not yet wired |
| Multi-return `[c, p] = blsprice(...)` | `blsprice` returns the call; put is a follow-on |
| ECM `mvnrstd` standard errors | only point estimates |
| Life tables (`lifetableconv`/`lifetablefit`/`lifetablegen`) | not yet wired |
| Most chart indicators (`candle`/`highlow`/`pointfig`/`kagi`/`renko`) | need OHLC chart primitives |

## Tests

16 finance gating tests in `test/Run/finance_*.m` + `portfolio_*.m`:
`finance_dates`, `finance_cashflows`, `finance_bonds`,
`finance_indicators`, `finance_perfmetrics`, `finance_blsgreeks`,
`portfolio_basic`, `efficient_frontier`, `finance_mvnreg`,
`finance_credit`, `finance_scorecard`, `portfolio_cvar`,
`portfolio_mad`, `finance_backtest`, `finance_sde`,
`finance_montecarlo`, `finance_blacklitterman`, `finance_riskparity`,
`portfolio_frontier_ext`, `portfolio_plotfrontier` (+ the timetable
suite from #57).
