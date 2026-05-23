% Curve Fitting Toolbox Tier-4 — smooth() branches + csaps + smoothingspline.
y = [1; 3; 2; 5; 4; 7; 6; 9; 8; 11];
m = smooth(y, 3);                       % moving average, span 3
fprintf('moving m(2)=%.4f m(5)=%.4f\n', m(2), m(5));
l = smooth(y, 5, 'lowess');            % local linear regression
fprintf('lowess l(5)=%.4f\n', l(5));

x = (0:0.5:5)';
yq = x.^2;
v = csaps(x, yq, 0.99, [1.0; 2.0; 3.0]);
fprintf('csaps at 2 = %.4f\n', v(2));

fsp = fit(x, yq, 'smoothingspline');
fprintf('smoothingspline at 2 = %.4f\n', fsp(2));
