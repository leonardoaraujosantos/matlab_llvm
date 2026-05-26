% Financial Toolbox Tier-2 §1 — performance metrics.

% Synthetic monthly returns: 12 months of a mildly positive series.
r = [0.02; 0.01; -0.005; 0.015; 0.025; -0.01; 0.03; 0.005; 0.01; -0.02; 0.018; 0.012];
b = [0.015; 0.012; 0.0; 0.012; 0.018; -0.005; 0.022; 0.008; 0.009; -0.015; 0.014; 0.010];

fprintf('mean(r) = %.4f\n', mean(r));
fprintf('std(r)  = %.4f\n', std(r));

% Sharpe with rf=0: mean/std
sh = sharpe(r, 0);
fprintf('sharpe(r, 0)    = %.4f\n', sh);

% Sortino: only downside std denominator (looks at r<MAR)
so = sortino(r, 0);
fprintf('sortino(r, 0)   = %.4f\n', so);

% Information ratio: active-return mean / tracking error
ir = inforatio(r, b);
fprintf('inforatio(r,b)  = %.4f\n', ir);

% Tracking error: std of active returns
te = tracking(r, b);
fprintf('tracking(r,b)   = %.4f\n', te);

% Maximum drawdown on a price series with a known dip.
p = [100; 110; 105; 95; 100; 120; 115; 130];
% Peak 110 -> trough 95 -> dd = 15/110 = 0.1364
% Peak 130 in last position
mdd = maxdrawdown(p);
fprintf('maxdrawdown(p)  = %.4f\n', mdd);

% Lower partial moment, order=2: target = 0.
lp = lpm(r, 0, 2);
fprintf('lpm(r,0,2)      = %.6f\n', lp);

% Jensen's alpha against benchmark b, rf = 0.
pa = portalpha(r, b, 0);
fprintf('portalpha       = %.4f\n', pa);
