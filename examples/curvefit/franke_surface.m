% franke_surface.m — Curve Fitting Toolbox Tier-5 headline.
% ----------------------------------------------------------------------
% Fit the classic Franke test surface with a polynomial surface model
% (poly55), read the goodness-of-fit, evaluate the fitted sfit on a grid,
% and render it.  The Franke function is the UG "Surface Fitting" demo.
n = 11;                                  % 11x11 sample grid over [0,1]^2
npts = n * n;
xs = zeros(npts, 1); ys = zeros(npts, 1); zs = zeros(npts, 1);
p = 0;
for ix = 1:n
    for iy = 1:n
        p = p + 1;
        xv = (ix - 1) / (n - 1);
        yv = (iy - 1) / (n - 1);
        t1 = ((9*xv - 2)^2 + (9*yv - 2)^2) / 4;
        t2 = (9*xv + 1)^2 / 49 + (9*yv + 1) / 10;
        t3 = ((9*xv - 7)^2 + (9*yv - 3)^2) / 4;
        t4 = (9*xv - 4)^2 + (9*yv - 7)^2;
        f = 0.75*exp(-t1) + 0.75*exp(-t2) + 0.5*exp(-t3) - 0.2*exp(-t4);
        xs(p) = xv; ys(p) = yv; zs(p) = f;
    end
end

[sf, gof] = fit([xs ys], zs, 'poly55');
fprintf('poly55 surface fit:\n');
fprintf('  R-squared = %.4f\n', gof.rsquare);
fprintf('  RMSE      = %.4f\n', gof.rmse);

% evaluate the fitted surface on a mesh and render it
[X, Y] = meshgrid(0:0.05:1, 0:0.05:1);
Z = feval(sf, X, Y);
figure;
surf(X, Y, Z);
xlabel('x'); ylabel('y'); zlabel('z');
title('Franke data — poly55 surface fit');
saveas(gcf, '/tmp/franke_surface.png');
