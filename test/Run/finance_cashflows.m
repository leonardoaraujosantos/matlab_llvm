% Financial Toolbox Tier-1 §2 — cash flows + depreciation.
% Stays in the f64-scalar lane (each helper takes/returns scalars,
% except pvvar/fvvar/irr/amortize/dep* which take/return matrices).

% pvfix(rate, n, pmt): PV of paying 100/period for 10 years at 5%.
% Expected ≈ -772.17 (annuity-PV factor 7.7217).
fprintf('pvfix(.05, 10, 100) = %.2f\n', pvfix(0.05, 10, 100));

% fvfix(rate, n, pmt): FV of 100/period for 10 years at 5%.
% Expected ≈ -1257.79.
fprintf('fvfix(.05, 10, 100) = %.2f\n', fvfix(0.05, 10, 100));

% pvvar / fvvar on a column-vector cash flow at 10%.
% Cash flow: -1000 at t=0, +200 in years 1..5, +300 at t=5.
cf = [-1000; 200; 200; 200; 200; 500];
fprintf('pvvar(cf, .10) = %.2f\n', pvvar(cf, 0.10));
fprintf('fvvar(cf, .10) = %.2f\n', fvvar(cf, 0.10));

% irr: a 5-year project of -1000 returning 300/yr (last year 400).
% Expected ~14.5%.
cf2 = [-1000; 300; 300; 300; 300; 400];
fprintf('irr(cf2) = %.4f\n', irr(cf2));

% payper: 30-year mortgage of $200k @ 6% APR (monthly).
% monthly rate = 0.005, n = 360. Expected payment ~ -1199.10
% (positive ≈ outflow). MATLAB returns the loan-payment sign (-1199).
fprintf('payper(.005, 360, 200000) = %.2f\n', payper(0.005, 360, 200000));

% nomrr/effrr round-trip: 6% nominal compounded monthly -> 6.168% effective.
fprintf('effrr(.06, 12) = %.4f\n', effrr(0.06, 12));
fprintf('nomrr(.061678, 12) = %.4f\n', nomrr(0.061678, 12));

% amortize: 4 cols x N rows = principal / interest / balance / cum-interest.
A = amortize(0.01, 6, 1000);   % 1% monthly, 6 periods, $1000 loan
fprintf('amortize size: rows=%.0f cols=%.0f\n', size(A,1), size(A,2));
% First-period interest = 1000 * 0.01 = 10.0
fprintf('A(1, interest) = %.4f\n', A(1, 2));
% Last-period balance ≈ 0.
fprintf('A(6, balance)  = %.4f\n', A(6, 3));

% Depreciation: straight-line on a $10k asset, $1k salvage, 5-yr life.
d = depstln(10000, 1000, 5);
fprintf('depstln(1)..(5) = %.0f %.0f %.0f %.0f %.0f\n', ...
        d(1), d(2), d(3), d(4), d(5));   % all = 1800

% Sum-of-years-digits.
d2 = depsoyd(10000, 1000, 5);
fprintf('depsoyd(1)..(5) = %.0f %.0f %.0f %.0f %.0f\n', ...
        d2(1), d2(2), d2(3), d2(4), d2(5));
% Sum = 9000; values are 3000, 2400, 1800, 1200, 600.

% Fixed declining-balance.
d3 = depfixdb(10000, 1000, 5);
fprintf('depfixdb(1) = %.2f\n', d3(1));
% Rate = 1 - (1000/10000)^(1/5) ≈ 0.3690 -> first dep = 3690.
