% System Identification Tier-4 — impulseest + forecast.
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
z = iddata(y, u, 1);
% impulseest: B = impulse response; B(1)=g(0)=0, B(2)=g(1)=1.0.
mi = impulseest(z, 20);
fprintf('B1 = %.2f\n', abs(mi.B(1)));   % 0.00 (no feedthrough)
fprintf('B2 = %.2f\n', mi.B(2));        % 0.99
fprintf('len = %.0f\n', size(mi.B, 2)); % 21

% forecast: AR(2) one-step-ahead equals the recursion -a1 y(N) - a2 y(N-1).
ya = zeros(N, 1);
for k = 3:N
    ya(k) = 0.3*ya(k-1) - 0.2*ya(k-2) + 0.01*sin(k);
end
za = iddata(ya, [], 1);
ma = ar(za, 2);
f = forecast(ma, za, 5);
expect = -ma.A(2)*ya(N) - ma.A(3)*ya(N-1);
fprintf('fc_match = %.6f\n', f(1) - expect);   % 0.000000
fprintf('fc_len = %.0f\n', size(f, 1));         % 5
