% Curve Fitting Toolbox Tier-5 — polynomial surface fitting (sfit).
% fit([x y], z, 'poly22') recovers a known bivariate quadratic exactly;
% sf(xq,yq) evaluates the fitted surface.
np = 0;
x = zeros(36, 1); y = zeros(36, 1); z = zeros(36, 1);
for i = 0:5
    for j = 0:5
        np = np + 1;
        x(np) = i; y(np) = j;
        z(np) = 1 + 2*i + 3*j + 4*i*i + 5*i*j + 6*j*j;   % poly22 ground truth
    end
end
[sf, gof] = fit([x y], z, 'poly22');
fprintf('r2 = %.6f\n', gof.rsquare);
c = coeffvalues(sf);
fprintf('ncoef = %.0f\n', numel(c));
fprintf('sf(1,1) = %.4f\n', sf(1, 1));    % 1+2+3+4+5+6 = 21
fprintf('sf(2,3) = %.4f\n', sf(2, 3));    % = 114
