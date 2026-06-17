# Financial Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Financial Toolbox in `matlab_llvm` (Tiers 1–7): date/calendar utilities, cash-flow and bond pricing, performance metrics, Black-Scholes option pricing, mean-variance/CVaR/MAD portfolio optimization, credit risk, regression with missing data, and SDE Monte Carlo.

## Requirements

### Requirement: Date and calendar utilities
The system SHALL provide trading-date and day-count utilities.

#### Scenario: Compute trading dates
- **WHEN** a program calls `busdate`, `isbusday`, `eomdate`, `daysdif`/`daysact`/`days360`/`days365`, `fweekdate`/`lweekdate`, or `m2xdate`
- **THEN** the system SHALL return the computed date or day count (matlab_busdate, matlab_isbusday, matlab_eomdate, matlab_daysdif, matlab_days360) (doc: docs/financial_toolbox_roadmap.md, docs/financial_toolbox_status.md) (src: runtime/toolbox/finance/runtime_finance.cpp, runtime/toolbox/finance/finance_classdefs.m)

### Requirement: Cash flow and bond pricing
The system SHALL price bonds and amortize cash flows.

#### Scenario: Price a bond
- **WHEN** a program calls `bndprice`/`bndyield`, `bnddurp`/`bnddury`, `bndconvp`, `amortize`, `irr`, `payper`, future-value, or depreciation functions
- **THEN** the system SHALL return the price, yield, duration, or cash-flow metric (matlab_bndprice, matlab_bndyield, matlab_bnddurp, matlab_amortize, matlab_irr, matlab_depstln) (doc: docs/financial_toolbox_roadmap.md, docs/financial_toolbox_status.md) (src: runtime/toolbox/finance/runtime_finance.cpp, runtime/toolbox/finance/finance_classdefs.m)

### Requirement: Option pricing and performance metrics
The system SHALL price options (Black-Scholes) and compute investment-performance metrics.

#### Scenario: Price an option and measure performance
- **WHEN** a program calls `blsprice`/`blsdelta`/`blsgamma`/`blsvega`/`blsimpv`, or performance metrics (`maxdrawdown`, `inforatio`, `portalpha`, `capm`, `lpm`)
- **THEN** the system SHALL return the option Greek/price or the performance metric (matlab_blsprice, matlab_blsdelta, matlab_blsimpv, matlab_maxdrawdown, matlab_inforatio, matlab_capm) (doc: docs/financial_toolbox_roadmap.md, docs/financial_toolbox_status.md) (src: runtime/toolbox/finance/runtime_finance.cpp, runtime/toolbox/finance/finance_classdefs.m)

### Requirement: Portfolio optimization
The system SHALL optimize portfolios under mean-variance, CVaR, and MAD objectives.

#### Scenario: Estimate an efficient frontier
- **WHEN** a program builds a `Portfolio`/`PortfolioCVaR`/`PortfolioMAD`, sets assets/scenarios, and estimates the frontier or port risk/return
- **THEN** the system SHALL return frontier weights, risks, and returns (matlab_portfolio_estimate_frontier, matlab_portfolio_estimate_max_sharpe, matlab_portfoliocvar_estimate_frontier, matlab_portfoliomad_estimate_frontier) (doc: docs/financial_toolbox_roadmap.md, docs/financial_toolbox_status.md) (src: runtime/toolbox/finance/runtime_finance.cpp, runtime/toolbox/finance/finance_classdefs.m)

### Requirement: Credit risk and regression with missing data
The system SHALL score credit, price credit default swaps, and estimate moments under missing data.

#### Scenario: Score credit and estimate moments
- **WHEN** a program builds a `creditscorecard`, scores/predicts default, prices a CDS, or runs ECM estimation (`ecmnmle`/`ecmncov`/`mvnrmle`)
- **THEN** the system SHALL return the credit score/default probability, CDS price, or estimated moments (matlab_creditscorecard_score, matlab_creditscorecard_probdefault, matlab_cdsprice, matlab_ecmnmle, matlab_ecmncov) (doc: docs/financial_toolbox_roadmap.md, docs/financial_toolbox_status.md) (src: runtime/toolbox/finance/runtime_finance.cpp, runtime/toolbox/finance/finance_classdefs.m)

### Requirement: SDE Monte Carlo and asset-allocation models
The system SHALL simulate stochastic differential equations and apply Black-Litterman / risk parity.

#### Scenario: Simulate an SDE path
- **WHEN** a program builds a `gbm`/`bm`/`cir`/`hwv` model and simulates, runs Monte-Carlo option pricing, or computes Black-Litterman views
- **THEN** the system SHALL return the simulated paths, MC price, or posterior allocation (matlab_optpricemc, matlab_blacklitterman, matlab_haltonseq) (doc: docs/financial_toolbox_roadmap.md, docs/financial_toolbox_status.md) (src: runtime/toolbox/finance/runtime_finance.cpp, runtime/toolbox/finance/finance_classdefs.m)
