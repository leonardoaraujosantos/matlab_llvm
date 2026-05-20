% System Identification Tier-1 — sim / predict / compare / goodnessOfFit.
% Noiseless ARX so the simulated output reproduces the data exactly
% (fit = 100 %, NRMSE = 0) — a clean determinism check for the
% validation surface.
N = 250;
u = zeros(N, 1);
for k = 1:N
    u(k) = sin(0.25 * k) + sin(0.04 * k);
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.6 * y(k-1) + 0.8 * u(k-1);
end
z = iddata(y, u, 1);
m = arx(z, [1 1 1]);

fit = compare(z, m);
fprintf('fit = %.2f\n', fit);          % 100.00

ys = sim(m, u);
fprintf('sim20 = %.4f\n', ys(20));
fprintf('y20   = %.4f\n', y(20));      % matches sim20

yh = predict(m, z, 1);
g = goodnessOfFit(yh, y);
fprintf('NRMSE = %.6f\n', g);   % 0.000000 (noiseless)
