% greybox_msd.m — System Identification Toolbox Tier-4 headline.
%
% Linear grey-box estimation (User's Guide Ch.13 "ODE Parameter
% Estimation"): recover the *physical constants* of a system whose
% structure is known but whose parameters are not.  Here a
% mass-spring-damper
%
%     m·q'' + c·q' + k·q = F  ->  x1' = x2,  x2' = -(k/m)x1 - (c/m)x2 + (1/m)F
%     y = q = x1
%
% is identified from input/output data by fitting [k/m, c/m] with a
% structure function par -> packed continuous [A B; C D].  greyest
% discretizes (ZOH) at the data Ts and minimizes the prediction error
% with the shipped Optimization Toolbox `lsqnonlin`.
%
%   iddata -> greyest(@structfn) -> physical parameters + validated fit

% ----- 1.  Collect data from the real (unknown-parameter) rig ----------
% True physical ratios: k/m = 4.0 (omega_n = 2 rad/s), c/m = 1.2 (zeta=0.3).
Ts = 0.05;
N  = 800;
a1_true = 4.0;     % k/m
a2_true = 1.2;     % c/m
u  = zeros(N, 1);
sd = 13579;
for k = 1:N
    sd   = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd / 2147483648 - 0.5);          % force excitation
end
% Fine sub-stepped ODE integration (piecewise-constant u) = exact ZOH.
x1 = 0; x2 = 0; y = zeros(N, 1); h = Ts / 50;
for k = 1:N
    y(k) = x1;
    uk = u(k);
    for sub = 1:50
        dx1 = x2;
        dx2 = -a1_true*x1 - a2_true*x2 + uk;
        x1 = x1 + h*dx1;
        x2 = x2 + h*dx2;
    end
end
z = iddata(y, u, Ts);
fprintf('Collected %.0f samples at Ts = %.2f s\n', N, Ts);

% ----- 2.  Define the grey-box structure ------------------------------
% par = [k/m; c/m].  Packed continuous realization M = [A B; C D]:
%   A = [0 1; -par(1) -par(2)],  B = [0; 1],  C = [1 0],  D = 0.
structfn = @(p) [0, 1, 0; -p(1), -p(2), 1; 1, 0, 0];

% ----- 3.  Estimate the physical parameters ---------------------------
p0 = [3.0; 1.0];                       % initial guess (deliberately off)
m  = greyest(z, p0, structfn, 2);
fprintf('\nEstimated physical parameters:\n');
fprintf('  k/m = %.4f   (true 4.0000)\n', m.Parameters(1));
fprintf('  c/m = %.4f   (true 1.2000)\n', m.Parameters(2));
fprintf('  natural frequency  omega_n = %.4f rad/s\n', sqrt(m.Parameters(1)));

% ----- 4.  Validate ----------------------------------------------------
fprintf('\nValidation fit (NRMSE) = %.2f %%\n', compare(z, m));
fprintf('Innovations variance V = %.3e\n', m.NoiseVariance);
