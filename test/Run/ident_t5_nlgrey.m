% System Identification Tier-5 — nonlinear grey-box (nlgreyest).
% Estimate the nonlinear-pendulum physical params from data.
%   dx1/dt = x2;  dx2/dt = -p1*sin(x1) - p2*x2 + u.   y = x1.
% Data generated with the SAME single-step Euler that nlgreyest uses
% internally, so the parameters are recovered exactly.
Ts = 0.02; N = 1000;
p1t = 5.0; p2t = 0.4;
u = zeros(N, 1); sd = 2024;
for k = 1:N
    sd = mod(sd*1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
end
x1 = 0.2; x2 = 0; y = zeros(N, 1);
for k = 1:N
    y(k) = x1;
    dx1 = x2;
    dx2 = -p1t*sin(x1) - p2t*x2 + u(k);
    x1 = x1 + Ts*dx1;
    x2 = x2 + Ts*dx2;
end
z = iddata(y, u, Ts);
statefn = @(zz) [zz(2); -zz(4)*sin(zz(1)) - zz(5)*zz(2) + zz(3)];
m = nlgreyest(z, [4.0; 0.2], statefn, 2);
fprintf('p1 = %.2f (true 5.0)\n', m.Parameters(1));
fprintf('p2 = %.2f (true 0.4)\n', m.Parameters(2));
