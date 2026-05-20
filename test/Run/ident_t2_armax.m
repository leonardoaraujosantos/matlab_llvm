% System Identification Tier-2 — ARMAX prediction-error estimation.
% True system  A=[1 -0.5], B=[0 1.0], C=[1 0.3] driven by a measured
% input and a deterministic-LCG innovation; armax recovers A/B/C.
N = 800;
e = zeros(N, 1); u = zeros(N, 1); sd = 99173;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648); e(k) = (sd/2147483648 - 0.5) * 0.2;
    sd = mod(sd * 1103515245 + 12345, 2147483648); u(k) = sign(sd/2147483648 - 0.5);
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5*y(k-1) + 1.0*u(k-1) + e(k) + 0.3*e(k-1);
end
z = iddata(y, u, 1);
m = armax(z, [1 1 1 1]);
fprintf('A2 = %.2f\n', m.A(2));   % -0.50
fprintf('B2 = %.2f\n', m.B(2));   %  1.00
fprintf('C2 = %.2f\n', m.C(2));   %  0.27
fprintf('Np = %.0f\n', m.Np);     %  3
fprintf('fit = %.0f\n', compare(z, m));   % high
