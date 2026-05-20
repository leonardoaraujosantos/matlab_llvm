% System Identification Tier-3 — transfer-function estimation.
% Same 2nd-order plant; tfest(z, 2, 2) recovers the discrete TF in
% idpoly form (B = numerator, F = denominator).  The denominator
% F = [1 -1.5 0.7] carries the pole information.
N = 600;
u = zeros(N, 1); sd = 271828;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
end
y = zeros(N, 1);
for k = 3:N
    y(k) = 1.5*y(k-1) - 0.7*y(k-2) + 1.0*u(k-1) + 0.5*u(k-2);
end
z = iddata(y, u, 0.1);
g = tfest(z, 2, 2);
fprintf('F2 = %.2f\n', g.F(2));   % -1.50
fprintf('F3 = %.2f\n', g.F(3));   %  0.70
fprintf('fit = %.0f\n', compare(z, g));   % high
