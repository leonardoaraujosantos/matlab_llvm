% Financial Toolbox Tier-1 §4 — bond pricing + T-bills.
%
% Bond pricing API uses (yield, coupon, periods, freq) directly rather
% than settle/maturity dates. Face value 100; coupon paid every 1/freq
% years. Standard semi-annual = freq 2.

% Par bond: 5% coupon, 5% yield, 10 years semi-annual.
% Price should be exactly 100.
p = bndprice(0.05, 0.05, 20, 2);
fprintf('par bond price = %.4f\n', p);

% Premium: 6% coupon, 4% yield, 10 years semi-annual.
% Approximate price by 10y annuity at 2%/period.
p2 = bndprice(0.04, 0.06, 20, 2);
fprintf('premium bond  = %.4f\n', p2);   % ~116.35

% Discount: 4% coupon, 6% yield, 10 years semi-annual. ~85.12
p3 = bndprice(0.06, 0.04, 20, 2);
fprintf('discount bond = %.4f\n', p3);

% Yield from price (round-trip).
y_back = bndyield(p2, 0.06, 20, 2);
fprintf('bndyield(premium) = %.4f\n', y_back);   % ~0.04

% Duration: 4% coupon, 4% yield, 10 years.  Macaulay should be ~8.4
% years for a par bond at 4% over 10 years.
d = bnddurp(0.04, 0.04, 20, 2);
fprintf('Macaulay = %.4f, Modified = %.4f\n', d(1), d(2));

% Convexity: ~80 years² for a 10y 4% bond at 4%.
c = bndconvp(0.04, 0.04, 20, 2);
fprintf('convexity = %.2f\n', c);

% Accrued-interest fraction: 30 days into a 180-day coupon period.
fprintf('accrfrac(30, 180) = %.4f\n', accrfrac(30, 180));

% T-bill: 91-day bill at 5% discount.
% Price = 100 * (1 - 0.05 * 91/360) = 100 - 1.2639 = 98.74
fprintf('prdisc(5%%, 91d) = %.4f\n', prdisc(0.05, 91));

% T-bill yield: from price 98.74 over 91 days.
% y = (100-98.74) * 365 / (98.74 * 91) = ~0.0513
fprintf('ytbill(98.74, 91) = %.4f\n', ytbill(98.74, 91));
