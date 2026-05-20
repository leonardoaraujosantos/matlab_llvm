% System Identification Tier-5 — recursive (online) estimation.
% recursiveARX tracks an ARX coefficient that jumps mid-stream;
% recursiveLS fits a static linear regression online.
N = 600;
u = zeros(N, 1); sd = 7;
for k = 1:N
    sd = mod(sd*1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
end
y = zeros(N, 1);
for k = 2:N
    if k < 300, a = 0.5; else, a = 0.8; end
    y(k) = a*y(k-1) + 1.0*u(k-1);
end
r = recursiveARX([1 1 1]);
r.ForgettingFactor = 0.97;
th = [0; 0];
for k = 2:N
    th = step(r, y(k), u(k));
end
fprintf('rARX a = %.2f\n', th(1));   % -0.80 (A = [1 -a], tracked to 0.8)
fprintf('rARX b = %.2f\n', th(2));   %  1.00

rls = recursiveLS(2);
tl = [0; 0]; sd = 99;
for k = 1:200
    sd = mod(sd*1103515245 + 12345, 2147483648); x1 = sd/2147483648;
    sd = mod(sd*1103515245 + 12345, 2147483648); x2 = sd/2147483648;
    tl = step(rls, 2.0*x1 + 3.0*x2, [x1, x2]);
end
fprintf('rLS p1 = %.2f\n', tl(1));   % 2.00
fprintf('rLS p2 = %.2f\n', tl(2));   % 3.00
