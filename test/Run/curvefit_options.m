% Curve Fitting Toolbox Tier-2 — the fitoptions surface.
% StartPoint seeding, Lower/Upper coefficient bounds (binding + non-binding),
% and Robust bisquare fitting that rejects a gross outlier.
x = (0:0.5:5)';
y = 3.0 * exp(-0.8 * x);

% StartPoint + non-binding bounds: recovers the truth.
o1 = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', [1 -0.1], ...
                'Lower', [0 -5], 'Upper', [10 0]);
[f1, g1] = fit(x, y, 'exp1', o1);
c1 = coeffvalues(f1);
fprintf('start  a=%.3f b=%.3f r2=%.5f\n', c1(1), c1(2), g1.rsquare);

% Binding bound: clamp the rate b >= -0.5 (the true -0.8 lies outside).
o2 = fitoptions('StartPoint', [3 -0.3], 'Lower', [0 -0.5], 'Upper', [10 0]);
[f2, g2] = fit(x, y, 'exp1', o2);
c2 = coeffvalues(f2);
fprintf('bound  b=%.3f\n', c2(2));

% Robust bisquare ignores a gross outlier.
yr = 3.0 * exp(-0.8 * x);
yr(3) = yr(3) + 5;
o3 = fitoptions('Robust', 'Bisquare', 'StartPoint', [3 -0.8]);
[f3, g3] = fit(x, yr, 'exp1', o3);
c3 = coeffvalues(f3);
fprintf('robust a=%.2f b=%.2f\n', c3(1), c3(2));
