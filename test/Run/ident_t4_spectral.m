% System Identification Tier-4 — non-parametric frequency response.
% etfe / spa estimate G(e^{jω}) from data; the DC-bin magnitude equals
% the static gain sum(B)/sum(A) = 1.5/0.2 = 7.5.
N = 600;
u = zeros(N, 1); sd = 271828;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = 1.0 + sign(sd/2147483648 - 0.5);   % DC offset so the DC bin is excited
end
y = zeros(N, 1);
for k = 3:N
    y(k) = 1.5*y(k-1) - 0.7*y(k-2) + 1.0*u(k-1) + 0.5*u(k-2);
end
z = iddata(y, u, 1);
ge = etfe(z);
gs = spa(z);
fprintf('etfe_DC = %.1f\n', ge.ResponseMag(1));   % ~7.4
fprintf('spa_DC  = %.1f\n', gs.ResponseMag(1));    % ~7.4
fprintf('Nf = %.0f\n', size(ge.Frequency, 1));     % 301
fprintf('f0 = %.2f\n', ge.Frequency(1));           % 0.00 (DC)
