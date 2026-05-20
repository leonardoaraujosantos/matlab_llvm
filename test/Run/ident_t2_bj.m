% System Identification Tier-2 — Box-Jenkins (BJ) estimation.
% True system  y = B/F u + C/D e,  B=[0 0.5], F=[1 -0.7],
% C=[1 0.4], D=[1 -0.2].  bj recovers the dynamics B/F well; the
% noise model C/D is harder, so only the dynamics are gated tightly.
N = 1000;
e = zeros(N, 1); u = zeros(N, 1); sd = 314159;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648); e(k) = (sd/2147483648 - 0.5) * 0.2;
    sd = mod(sd * 1103515245 + 12345, 2147483648); u(k) = sign(sd/2147483648 - 0.5);
end
yf = zeros(N, 1); ne = zeros(N, 1); y = zeros(N, 1);
for k = 2:N
    yf(k) = 0.7*yf(k-1) + 0.5*u(k-1);          % B/F u
    ne(k) = 0.2*ne(k-1) + e(k) + 0.4*e(k-1);   % C/D e
    y(k)  = yf(k) + ne(k);
end
z = iddata(y, u, 1);
m = bj(z, [1 1 1 1 1]);   % [nb nc nd nf nk]
fprintf('B2 = %.2f\n', m.B(2));   %  0.50
fprintf('F2 = %.2f\n', m.F(2));   % -0.70
fprintf('Np = %.0f\n', m.Np);     %  4
fprintf('fit = %.0f\n', compare(z, m));   % high
