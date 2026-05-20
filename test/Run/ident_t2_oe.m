% System Identification Tier-2 — Output-Error (OE) estimation.
% True system  y = B/F u + e,  F=[1 -0.7], B=[0 0.5].
N = 800;
e = zeros(N, 1); u = zeros(N, 1); sd = 99173;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648); e(k) = (sd/2147483648 - 0.5) * 0.2;
    sd = mod(sd * 1103515245 + 12345, 2147483648); u(k) = sign(sd/2147483648 - 0.5);
end
yf = zeros(N, 1); y = zeros(N, 1);
for k = 2:N
    yf(k) = 0.7*yf(k-1) + 0.5*u(k-1);
    y(k)  = yf(k) + e(k);
end
z = iddata(y, u, 1);
m = oe(z, [1 1 1]);
fprintf('B2 = %.2f\n', m.B(2));   %  0.50
fprintf('F2 = %.2f\n', m.F(2));   % -0.70
fprintf('Np = %.0f\n', m.Np);     %  2
