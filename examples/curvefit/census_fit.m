% census_fit.m — Curve Fitting Toolbox HEADLINE (Tier-1).
% ----------------------------------------------------------------------
% The canonical US-census demo, exercising the everyday curve-fitting arc
% end to end:
%   fit(cdate, pop, 'poly2')  ->  cfit object
%                             ->  [f, gof]  (R^2 / RMSE goodness-of-fit)
%                             ->  f(2030)   (call-syntax forecast)
%                             ->  feval + plot  (overlay on the data).
%
% Decennial US resident population (millions), 1790-1990 — the same series
% as MATLAB's built-in census.mat.  fit() centers-and-scales the predictor
% internally for conditioning (a raw Vandermonde in calendar years is
% hopeless at degree 2), so disp(f) reports the normalization.  No external
% dependency: the quadratic least-squares rides the shipped polyfit/polyval.
cdate = (1790:10:1990)';
pop = [  3.929;   5.308;   7.240;   9.638;  12.866;  17.069;  23.192; ...
        31.443;  38.558;  50.156;  62.948;  75.995;  91.972; 105.711; ...
       122.775; 131.669; 150.697; 179.323; 203.212; 226.546; 248.710];

% ----- fit a quadratic + read goodness-of-fit -------------------------
[f, gof] = fit(cdate, pop, 'poly2');
disp(f);
fprintf('R-squared      = %.4f\n', gof.rsquare);
fprintf('adj. R-squared = %.4f\n', gof.adjrsquare);
fprintf('RMSE           = %.4f\n', gof.rmse);

% ----- forecast future censuses ---------------------------------------
% (evaluate on a query vector and index — feval is used uniformly with
% vector inputs throughout, which keeps the model-evaluation return type
% consistent across the script.)
yrs = (2000:10:2030)';
pf  = feval(f, yrs);
fprintf('forecast 2000  = %.1f million\n', pf(1));
fprintf('forecast 2030  = %.1f million\n', pf(4));

% ----- overlay the fitted curve on the observed data ------------------
xq = (1790:5:2030)';
yq = feval(f, xq);
figure;
plot(cdate, pop, 'o', xq, yq, '-'); grid on;
xlabel('year'); ylabel('US resident population (millions)');
title('US census — poly2 fit + 2030 forecast');
saveas(gcf, '/tmp/census_fit.png');
