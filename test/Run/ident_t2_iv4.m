% System Identification Tier-2 — instrumental variables (iv4) + delayest.
% Under a coloured additive disturbance v = e + 0.9 e(t-1), plain ARX is
% biased; the IV method recovers A/B more accurately.  delayest finds
% the transport delay of a separate delay-3 record.
N = 1000;
e = zeros(N, 1); u = zeros(N, 1); sd = 5551212;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648); e(k) = (sd/2147483648 - 0.5) * 0.6;
    sd = mod(sd * 1103515245 + 12345, 2147483648); u(k) = sign(sd/2147483648 - 0.5);
end
v = zeros(N, 1); y = zeros(N, 1);
for k = 2:N
    v(k) = e(k) + 0.9 * e(k-1);
    y(k) = 0.5 * y(k-1) + 1.0 * u(k-1) + v(k);
end
z = iddata(y, u, 1);
mi = iv4(z, [1 1 1]);
fprintf('IV4 A2 = %.2f\n', mi.A(2));   % -0.50
fprintf('IV4 B2 = %.2f\n', mi.B(2));   %  1.01
fprintf('IV4 Np = %.0f\n', mi.Np);     %  2

y2 = zeros(N, 1);
for k = 4:N
    y2(k) = 0.5 * y2(k-1) + 1.0 * u(k-3);
end
z2 = iddata(y2, u, 1);
fprintf('delayest = %.0f\n', delayest(z2));   % 3
