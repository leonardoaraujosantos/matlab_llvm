% System Identification Tier-3 headline — data-driven MPC.
% Identify a plant from I/O data (ssest), validate, convert to ss, and
% design an MPC controller on the identified model.  Exercises the
% System ID -> Control System -> MPC cross-toolbox chain (and confirms
% the ident + mpc class preludes coexist in one translation unit).
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
z = iddata(y, u, 0.1);
sys = ssest(z, 2);
fprintf('fit = %.1f\n', compare(z, sys));   % ~96.8
P = ss(sys);
ctrl = mpc(P, 10, 3);
yc = sim(ctrl, 30, 1.0);
fprintf('y(10) = %.2f\n', yc(10, 1));   % tracking 1.00
fprintf('y(30) = %.2f\n', yc(30, 1));   % tracking 1.00
