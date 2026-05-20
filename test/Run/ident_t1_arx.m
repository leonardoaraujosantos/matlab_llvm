% System Identification Tier-1 — ARX least-squares estimation.
% Noiseless data from y(t) = 0.5 y(t-1) + 1.0 u(t-1) is recovered
% exactly: A = [1, -0.5], B = [0, 1.0], nk = 1.
N = 300;
u = zeros(N, 1);
for k = 1:N
    u(k) = sin(0.3 * k) + 0.5 * sin(0.07 * k);
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5 * y(k-1) + 1.0 * u(k-1);
end
z = iddata(y, u, 1);
m = arx(z, [1 1 1]);
fprintf('A2 = %.4f\n', m.A(2));      % -0.5000
fprintf('B1 = %.4f\n', m.B(1));      %  0.0000 (nk = 1 leading zero)
fprintf('B2 = %.4f\n', m.B(2));      %  1.0000
fprintf('nk = %.0f\n', m.nk(1));     %  1
fprintf('Ts = %.2f\n', m.Ts);        %  1.00
fprintf('Np = %.0f\n', m.Np);        %  2
fprintf('Ns = %.0f\n', m.Ns);        %  299
fprintf('V = %.6f\n', m.NoiseVariance);   % 0.000000 (noiseless)
