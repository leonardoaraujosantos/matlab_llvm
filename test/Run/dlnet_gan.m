% Deep Learning T5.4 gating test — GAN with alternating SGD over the
% autodiff.  Uses the **least-squares GAN** (LSGAN) formulation:
%   D wants D(real) -> 1, D(fake) -> 0   --> 0.5*mse(D(real),1) + 0.5*mse(D(fake),0)
%   G wants D(fake) -> 1                 --> 0.5*mse(D(fake),1)
% — the shared-discriminator path (same `Wd*` used for both real and fake
% inputs) exercises the classdef-operator-overloading dispatch on the
% binary `+` of two `mse` calls — without that fix the loss would silently
% lower to `matlab_add_mm` and segfault when called on dlarray pointers.
%
% Real samples come from N(2, 0.5^2); generator G(z) must learn to produce
% samples whose mean approaches 2.  Gating signal:
%   (a) D's loss strictly drops once it has been updated,
%   (b) G's sample mean moves toward the target 2.0.

rng(0);
Nreal = 8;
real_data = 2.0 + 0.5 * randn(1, Nreal);

% Generator: z(1) -> 4 hidden -> 1
Wg1 = dlarray(0.2 * randn(4, 1)); bg1 = dlarray(zeros(4, 1));
Wg2 = dlarray(0.2 * randn(1, 4)); bg2 = dlarray(zeros(1, 1));

% Discriminator: x(1) -> 4 hidden -> 1 (linear output, no sigmoid)
Wd1 = dlarray(0.2 * randn(4, 1)); bd1 = dlarray(zeros(4, 1));
Wd2 = dlarray(0.2 * randn(1, 4)); bd2 = dlarray(zeros(1, 1));

real_dl = dlarray(real_data);
one_row = dlarray(ones(1, Nreal));
zero_row = dlarray(zeros(1, Nreal));

lr = 0.02;
nIter = 80;
initial_d_loss = 0;
final_d_loss = 0;
initial_fake_mean = 0;

for it = 1:nIter
    z = randn(1, Nreal);
    z_dl = dlarray(z);

    % --- Update D --------------------------------------------------------
    fake = Wg2 * relu(Wg1 * z_dl + bg1) + bg2;
    d_real = Wd2 * relu(Wd1 * real_dl + bd1) + bd2;
    d_fake = Wd2 * relu(Wd1 * fake + bd1) + bd2;
    d_loss = mse(d_real, one_row) + mse(d_fake, zero_row);
    Lv = extractdata(d_loss);
    if it == 1; initial_d_loss = Lv(1); end
    final_d_loss = Lv(1);

    gWd1 = dlgradient(d_loss, Wd1); gbd1 = dlgradient(d_loss, bd1);
    gWd2 = dlgradient(d_loss, Wd2); gbd2 = dlgradient(d_loss, bd2);
    Wd1 = dlarray(extractdata(Wd1) - lr * gWd1); bd1 = dlarray(extractdata(bd1) - lr * gbd1);
    Wd2 = dlarray(extractdata(Wd2) - lr * gWd2); bd2 = dlarray(extractdata(bd2) - lr * gbd2);

    % --- Update G --------------------------------------------------------
    fake = Wg2 * relu(Wg1 * z_dl + bg1) + bg2;
    d_fake = Wd2 * relu(Wd1 * fake + bd1) + bd2;
    g_loss = mse(d_fake, one_row);

    gWg1 = dlgradient(g_loss, Wg1); gbg1 = dlgradient(g_loss, bg1);
    gWg2 = dlgradient(g_loss, Wg2); gbg2 = dlgradient(g_loss, bg2);
    Wg1 = dlarray(extractdata(Wg1) - lr * gWg1); bg1 = dlarray(extractdata(bg1) - lr * gbg1);
    Wg2 = dlarray(extractdata(Wg2) - lr * gWg2); bg2 = dlarray(extractdata(bg2) - lr * gbg2);

    if it == 1
        f0 = extractdata(fake);
        s = 0;
        for n = 1:Nreal; s = s + f0(n); end
        initial_fake_mean = s / Nreal;
    end
end

% Final generator sample mean over 32 fresh noise draws.
zf = randn(1, 32);
fake_f = extractdata(Wg2 * relu(Wg1 * dlarray(zf) + bg1) + bg2);
sum_f = 0;
for n = 1:32; sum_f = sum_f + fake_f(n); end
final_fake_mean = sum_f / 32;

d_learned = 0;
if final_d_loss < initial_d_loss
    d_learned = 1;
end
g_toward_target = 0;
if abs(final_fake_mean - 2.0) < abs(initial_fake_mean - 2.0)
    g_toward_target = 1;
end

fprintf('discriminator loss strictly drops = %.0f\n', d_learned);
fprintf('generator mean moves toward target = %.0f\n', g_toward_target);
