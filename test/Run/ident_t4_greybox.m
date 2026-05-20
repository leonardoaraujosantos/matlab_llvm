% System Identification Tier-4 headline — linear grey-box estimation.
% Estimate the physical parameters [k/m, c/m] of a mass-spring-damper
% from data via a structure function par -> packed continuous [A B; C D].
% True: k/m = 4.0, c/m = 1.2.
Ts = 0.05; N = 800;
a1 = 4.0; a2 = 1.2;
u = zeros(N, 1); sd = 13579;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
end
% Fine sub-stepped integration = exact ZOH response (matches greyest c2d).
x1 = 0; x2 = 0; y = zeros(N, 1); h = Ts / 50;
for k = 1:N
    y(k) = x1;
    uk = u(k);
    for sub = 1:50
        dx1 = x2;
        dx2 = -a1*x1 - a2*x2 + uk;
        x1 = x1 + h*dx1;
        x2 = x2 + h*dx2;
    end
end
z = iddata(y, u, Ts);
structfn = @(p) [0, 1, 0; -p(1), -p(2), 1; 1, 0, 0];
m = greyest(z, [3.0; 1.0], structfn, 2);
fprintf('k/m = %.2f\n', m.Parameters(1));   % 4.00
fprintf('c/m = %.2f\n', m.Parameters(2));   % 1.20
fprintf('fit = %.1f\n', compare(z, m));     % ~99.9
