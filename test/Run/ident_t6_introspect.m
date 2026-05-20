% System Identification Tier-6 — getpvec / setpvec / getcov introspection.
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
m = arx(z, [1 1 1]);

% getpvec — reconstruct θ.
p = getpvec(m);
fprintf('pvec len = %.0f\n', size(p, 1));     % 2
fprintf('pvec(1) = %.2f\n', p(1));            % -0.50 (= m.A(2))
fprintf('pvec(2) = %.2f\n', p(2));            %  1.00 (= m.B(2))

% setpvec — write back and verify.
setpvec(m, [-0.7; 1.2]);
fprintf('A after = %.2f\n', m.A(2));          % -0.70
fprintf('B after = %.2f\n', m.B(2));          %  1.20

% getcov — np×np parameter covariance from cached Gram.
m2 = arx(z, [1 1 1]);
cv = getcov(m2);
fprintf('cov rows = %.0f\n', size(cv, 1));    % 2
fprintf('cov cols = %.0f\n', size(cv, 2));    % 2
fprintf('cov11 = %.5f\n', cv(1, 1));          % small positive
