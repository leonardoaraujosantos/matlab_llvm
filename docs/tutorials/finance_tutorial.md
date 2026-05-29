# Financial Toolbox — Tutorial

The Financial Toolbox runtime compiles quantitative-finance workflows to native code: mean-variance portfolio optimization, Black-Scholes option pricing with Greeks, Black-Litterman view blending, credit scorecards, strategy backtesting, SDE Monte-Carlo pricing, and financial timetables with technical indicators. The headline workflows are the efficient frontier, Black-Litterman, the backtest engine, the credit scorecard, and Monte-Carlo Black-Scholes pricing.

## Supported features

- **Portfolio (mean-variance)**: `Portfolio()` classdef, `setAssetMoments`, `setDefaultConstraints`, `setBounds`, `setBudget`, `estimateAssetMoments`, `estimateFrontier`, `estimateFrontierByReturn`, `estimatePortMoments`, `estimateMaxSharpeRatio`, `estimateBounds`, `estimatePortFrontier`, `plotFrontier`.
- **Option pricing + Greeks**: `blsprice`, `blsdelta`, `blsgamma`, `blsvega`, `blsrho`, `blstheta`, `blslambda`, `blsimpv`; Black-Litterman `blacklitterman`.
- **Credit risk**: `creditscorecard(X, y)` classdef, `fitmodel`, `probdefault`, `score`.
- **Backtesting + performance**: `backtest(R, w, rebalance)`, `backtestSummary`; metrics `sharpe`, `sortino`, `inforatio`, `tracking`, `maxdrawdown`, `lpm`.
- **SDE Monte-Carlo**: `bm`, `gbm`, `cir`, `hwv` classdefs with `simByEuler` / `simBySolution`; `optpricemc` (MC option price), `haltonseq` (quasi-MC low-discrepancy sequence).
- **Timetables + indicators**: `timetable`, `summary`, `fillmissing`, `timerange`, `head`, `retime`, `synchronize`, `movavg`, `macd`, `sma`, `bolling`; returns `tick2ret` / `ret2tick`; fixed-income `prdisc`, `prtbill`, `ytbill`, `accrfrac`.

## Build & run

```bash
build/matlabc -emit-llvm examples/finance/efficient_frontier.m > /tmp/efficient_frontier.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/efficient_frontier.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/efficient_frontier
/tmp/efficient_frontier
```

## Worked examples

### Markowitz efficient frontier  (`examples/finance/efficient_frontier.m`)

The Tier-3 headline: build a 5-asset `Portfolio`, sweep the mean-variance frontier, and locate the tangency (max-Sharpe) portfolio.

```matlab
p = Portfolio();
p = setAssetMoments(p, m, C);
p = setDefaultConstraints(p);

W = estimateFrontier(p, 20);                       % assets x 20 frontier
rm_lo = estimatePortMoments(p, W(:, 1));           % min-variance endpoint
rm_hi = estimatePortMoments(p, W(:, 20));          % max-return endpoint

w_ms  = estimateMaxSharpeRatio(p);                 % tangency portfolio
rm_ms = estimatePortMoments(p, w_ms);
sharpe_ratio = rm_ms(2) / rm_ms(1);

w_target  = estimateFrontierByReturn(p, 0.10);     % reverse lookup
moments   = estimateAssetMoments(p, returns);      % from sample returns
```

`estimatePortMoments` returns `[risk, return]`. The frontier endpoints bracket the achievable risk/return; `estimateMaxSharpeRatio` finds the tangency weights, and `estimateFrontierByReturn` does the reverse lookup for a target return.

### Black-Litterman view blending  (`examples/finance/black_litterman.m`)

The Tier-7 headline: combine the market-equilibrium prior with an investor view via `blacklitterman`, then feed the posterior returns into a `Portfolio`.

```matlab
delta = 2.5; tau = 0.025;
P = [1 0 -1];          % view: asset 1 outperforms asset 3 ...
Q = [0.06];            % ... by 6%

mu_bl = blacklitterman(Sigma, wmkt, P, Q, tau, delta);
Pi    = delta * Sigma * wmkt;            % equilibrium prior

p = Portfolio();
p = setAssetMoments(p, mu_bl, Sigma);
p = setDefaultConstraints(p);
w = estimateMaxSharpeRatio(p);
rm = estimatePortMoments(p, w);
```

The `P`/`Q` pair encodes the view (`+1` on asset 1, `−1` on asset 3, spread `Q`); `blacklitterman` returns posterior expected returns that tilt asset 1 up and asset 3 down relative to the equilibrium prior `Π = δ·Σ·w_mkt`. Those posterior returns then drive a standard max-Sharpe optimization.

### Monte-Carlo Black-Scholes pricing  (`examples/finance/monte_carlo_blsprice.m`)

The Tier-6 headline: price a European call by GBM Monte-Carlo under the risk-neutral measure and check it against the closed-form Black-Scholes price.

```matlab
ref = blsprice(S0, K, r, T, sigma);     % closed-form reference

g = gbm(r, sigma, S0);                   % risk-neutral GBM (drift = r)
P  = simBySolution(g, 252, T/252, 20000);
ST = P(253, :);                          % terminal prices

mcPrice = optpricemc(ST, K, r, T);       % discounted mean payoff
fprintf('abs error vs closed form = %.4f\n', abs(mcPrice - ref));

H = haltonseq(8, 2);                     % low-discrepancy quasi-MC draws
```

`simBySolution` uses the exact GBM transition (no Euler discretization error in the price process). `optpricemc` discounts the sample-mean payoff and sidesteps the elementwise `max(matrix, scalar)` lowering gap. The MC price converges to the closed-form `blsprice` value.

### Credit scorecard  (`examples/finance/credit_scorecard.m`)

The Tier-4 logistic credit core: fit a scorecard on labelled borrowers, then predict default probabilities and scores.

```matlab
sc = creditscorecard(X, y);
sc = fitmodel(sc);

pd = probdefault(sc, X);
fprintf('PD(good row 1) = %.3f\n', pd(1));   % near 0
fprintf('PD(bad  row 5) = %.3f\n', pd(5));   % near 1

s = score(sc, X);                            % log-odds scores
pdn = probdefault(sc, [7.2 0.9]);            % fresh applicant
```

`probdefault` returns the per-row default probability; `score` returns the log-odds, with riskier borrowers scoring higher. A fresh "good" applicant gets a low PD.

### Backtest engine  (`examples/finance/backtest_strategies.m`)

The Tier-5 backtest: run a return matrix through both a rebalanced and a buy-and-hold strategy, then summarize.

```matlab
eqR = backtest(R, w, 1);                 % rebalance to target each period
eqH = backtest(R, w, 0);                 % buy-and-hold (weights drift)
fprintf('rebal final equity = %.4f\n', eqR(7));

s = backtestSummary(eqR);
fprintf('total return = %.4f\n', s(1));
fprintf('ann sharpe   = %.4f\n', s(2));
fprintf('max drawdown = %.4f\n', s(3));
```

`backtest` returns the equity curve (length = periods + 1); the third argument toggles rebalancing. `backtestSummary` returns `[total return, annualized Sharpe, max drawdown]`.

### Financial timetables  (`examples/finance/using_timetables_in_finance.m`)

A port of the MathWorks "Using Timetables in Finance" doc page: build an OHLCV `timetable`, repair NaNs, subscript by time range, compute indicators, and aggregate weekly.

```matlab
TMW = timetable(open, high, low, close, vol, ...
                'VariableNames', {'Open','High','Low','Close','Volume'}, ...
                'RowTimes', dates);
TMW = fillmissing(TMW, 'linear');                  % repair sprinkled NaNs

idx = timerange(datetime(2014,1,15), datetime(2014,2,15), 'closed');
head(TMW(idx, :), 4);

ema15 = movavg(TMW(:, 'Close'), 'exponential', 15);
mline = macd(TMW(:, 'Close'));

wo = retime(TMW(:, 'Open'), 'weekly', 'firstvalue');
weeklyAll = synchronize([wo wh wl wc], TMW(:, 'Volume'), 'weekly', 'sum');
```

This exercises the full timetable surface: `summary`, `fillmissing`, `timerange` row subscripts, column subscripts, `movavg`/`macd` indicators, `retime` weekly aggregation, horizontal concatenation, and `synchronize` — finishing with a datetime-axis `plot` saved to PNG.

## Limitations & carve-outs

From [`docs/financial_toolbox_roadmap.md`](../financial_toolbox_roadmap.md) and [`docs/financial_toolbox_status.md`](../financial_toolbox_status.md):

- `blsprice` returns the call price; the multi-return `[c, p] = blsprice(...)` put form is a follow-on.
- SDE engine ships the 1-D Euler/QMC core (`bm`/`gbm`/`cir`/`hwv`). Heston 2-D stochastic vol, correlated multi-asset baskets, Sobol, and stratified sampling are SDE follow-ons.
- **Out of scope**: GUI dialogs (`uicalendar`/`cfui`), Reinforcement-Learning trade-execution demos, parallel-computing Monte-Carlo beyond the shipped parfor lane, and the corporate carbon-emission timetable extraction demo (static-compile lane).
- Examples synthesize data deterministically (e.g. `randn` / `cumsum`) in place of `.mat` fixtures, since there is no MAT-file decoder on this lane.

## See also

- Roadmap: [`docs/financial_toolbox_roadmap.md`](../financial_toolbox_roadmap.md)
- Status: [`docs/financial_toolbox_status.md`](../financial_toolbox_status.md)
- Examples: `examples/finance/`
