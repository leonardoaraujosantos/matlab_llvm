% arx_regularization.m — System Identification Toolbox Tier-6 headline.
%
% Regularized ARX (User's Guide §1 "Regularized Estimates of Model
% Parameters"): on short / noisy data, plain least-squares is sensitive
% to noise; a small ridge λ on the Gram matrix shrinks the estimate
% toward zero and tightens the parameter covariance.  The toolbox
% exposes the option through the standard `arxOptions` carrier:
%
%   opt = arxOptions(); opt.Regularization = lambda;
%   m   = arx(data, [na nb nk], opt);
%
% The Tier-6 polish surface also adds getpvec / setpvec / getcov for
% parameter introspection.

% ----- 1.  A deliberately short, noisy ARX record ---------------------
% True system:  y(t) = 0.5 y(t-1) + 1.0 u(t-1) + e(t),  σ_e ≈ 0.5
N  = 60;
u  = zeros(N, 1);  e = zeros(N, 1);
sd = 17;
for k = 1:N
    sd   = mod(sd*1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
    sd   = mod(sd*1103515245 + 12345, 2147483648);
    e(k) = (sd/2147483648 - 0.5) * 1.0;            % large innovations
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5*y(k-1) + 1.0*u(k-1) + e(k);
end
z = iddata(y, u, 1);
fprintf('Short record: %.0f samples, large innovations\n', N);

% ----- 2.  Plain ARX (the baseline) -----------------------------------
m = arx(z, [1 1 1]);
fprintf('\nPlain ARX:    a = %+.3f, b = %+.3f (truth a=-0.500, b=1.000)\n', ...
        m.A(2), m.B(2));
c = getcov(m);
fprintf('Plain ARX:    var(a) = %.4f, var(b) = %.4f\n', c(1, 1), c(2, 2));

% ----- 3.  Regularized ARX via arxOptions -----------------------------
opt = arxOptions();
opt.Regularization = 1.0;
mr = arx(z, [1 1 1], opt);
fprintf('\nRidged ARX (λ = %.1f):\n', mr.Lambda);
fprintf('              a = %+.3f, b = %+.3f\n', mr.A(2), mr.B(2));
cr = getcov(mr);
fprintf('              var(a) = %.4f, var(b) = %.4f (shrunk)\n', cr(1, 1), cr(2, 2));

% ----- 4.  Parameter introspection (Tier-6) ---------------------------
% getpvec rebuilds θ from the polynomial tails; setpvec writes back.
theta = getpvec(mr);
fprintf('\ngetpvec(mr) = [%+.3f; %+.3f]\n', theta(1), theta(2));
setpvec(mr, [-0.45; 0.95]);
fprintf('after setpvec: A(2) = %+.3f, B(2) = %+.3f\n', mr.A(2), mr.B(2));
