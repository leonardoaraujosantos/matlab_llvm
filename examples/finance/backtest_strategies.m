% Financial Toolbox Tier-5 §3 — backtest engine (function-form).

% 6 periods of 2-asset returns.
R = [ 0.02  0.01
      0.01 -0.01
     -0.02  0.03
      0.03  0.00
      0.01  0.02
     -0.01  0.01 ];
w = [0.6; 0.4];

% Rebalanced backtest (rebalance to target each period).
eqR = backtest(R, w, 1);
fprintf('rebal equity length = %.0f\n', length(eqR));
fprintf('rebal final equity  = %.4f\n', eqR(7));

% Buy-and-hold backtest (weights drift).
eqH = backtest(R, w, 0);
fprintf('hold final equity   = %.4f\n', eqH(7));

% Summary stats from the rebalanced equity curve.
s = backtestSummary(eqR);
fprintf('total return = %.4f\n', s(1));
fprintf('ann sharpe   = %.4f\n', s(2));
fprintf('max drawdown = %.4f\n', s(3));
