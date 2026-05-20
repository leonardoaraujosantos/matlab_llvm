% System Identification Tier-6 — regularized arx via arxOptions.
% Verifies arxOptions plumbing, the ridge solve, and Lambda round-trip.
N = 200;
u = zeros(N, 1); e = zeros(N, 1); sd = 12345;
for k = 1:N
    sd = mod(sd*1103515245 + 12345, 2147483648); u(k) = sign(sd/2147483648 - 0.5);
    sd = mod(sd*1103515245 + 12345, 2147483648); e(k) = (sd/2147483648 - 0.5) * 0.5;
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5*y(k-1) + 1.0*u(k-1) + e(k);
end
z = iddata(y, u, 1);
m  = arx(z, [1 1 1]);
fprintf('plain a = %.2f\n', m.A(2));         % -0.50
fprintf('plain Lambda = %.2f\n', m.Lambda);   %  0.00
opt = arxOptions();
opt.Regularization = 0.5;
mr = arx(z, [1 1 1], opt);
fprintf('reg   a = %.2f\n', mr.A(2));         % ~-0.50 (slightly shrunk)
fprintf('reg Lambda = %.2f\n', mr.Lambda);    %  0.50
