% dl_gan.m — Deep Learning T5.4: GAN with alternating SGD over the
% autodiff.  Uses the **least-squares GAN** (LSGAN) formulation —
% structurally simpler than vanilla BCE-GAN and routed entirely through
% the `mse` / shared-discriminator paths the dlarray autodiff handles.
%
% Real samples come from N(2.0, 0.5^2); the generator G(z) must learn to
% produce 1-D samples whose distribution matches.  After ~200 alternating
% updates the generator's sample mean approaches 2.0 within ~0.3 of the
% target — the same kind of convergence the equivalent MATLAB+`dlnetwork`
% GAN script produces.
%
% Architectural note: D is used in *both* the real-input and fake-input
% branches of the loss, so the binary `+` in
% `mse(d_real, ones) + mse(d_fake, zeros)` MUST dispatch through the
% classdef-operator-overloading path.  Without that dispatch the loss
% silently lowers to `matlab_add_mm` and segfaults on dlarray pointers —
% this example exercises the fix end-to-end.

rng(0);
Nreal = 16;
real_data = 2.0 + 0.5 * randn(1, Nreal);

% Generator: z(1) -> 8 hidden -> 1
Wg1 = dlarray(0.3 * randn(8, 1)); bg1 = dlarray(zeros(8, 1));
Wg2 = dlarray(0.3 * randn(1, 8)); bg2 = dlarray(zeros(1, 1));

% Discriminator: x(1) -> 8 hidden -> 1 (linear output, LSGAN-style)
Wd1 = dlarray(0.3 * randn(8, 1)); bd1 = dlarray(zeros(8, 1));
Wd2 = dlarray(0.3 * randn(1, 8)); bd2 = dlarray(zeros(1, 1));

real_dl  = dlarray(real_data);
one_row  = dlarray(ones(1, Nreal));
zero_row = dlarray(zeros(1, Nreal));

lr = 0.02;
nIter = 200;

initLoss = 0;
initial_fake_mean = 0;
for it = 1:nIter
    z    = randn(1, Nreal);
    z_dl = dlarray(z);

    fake   = Wg2 * relu(Wg1 * z_dl + bg1) + bg2;
    d_real = Wd2 * relu(Wd1 * real_dl + bd1) + bd2;
    d_fake = Wd2 * relu(Wd1 * fake    + bd1) + bd2;
    d_loss = mse(d_real, one_row) + mse(d_fake, zero_row);
    Lv = extractdata(d_loss);
    if it == 1; initLoss = Lv(1); end

    gWd1 = dlgradient(d_loss, Wd1); gbd1 = dlgradient(d_loss, bd1);
    gWd2 = dlgradient(d_loss, Wd2); gbd2 = dlgradient(d_loss, bd2);
    Wd1 = dlarray(extractdata(Wd1) - lr * gWd1); bd1 = dlarray(extractdata(bd1) - lr * gbd1);
    Wd2 = dlarray(extractdata(Wd2) - lr * gWd2); bd2 = dlarray(extractdata(bd2) - lr * gbd2);

    fake   = Wg2 * relu(Wg1 * z_dl + bg1) + bg2;
    d_fake = Wd2 * relu(Wd1 * fake + bd1) + bd2;
    g_loss = mse(d_fake, one_row);

    gWg1 = dlgradient(g_loss, Wg1); gbg1 = dlgradient(g_loss, bg1);
    gWg2 = dlgradient(g_loss, Wg2); gbg2 = dlgradient(g_loss, bg2);
    Wg1 = dlarray(extractdata(Wg1) - lr * gWg1); bg1 = dlarray(extractdata(bg1) - lr * gbg1);
    Wg2 = dlarray(extractdata(Wg2) - lr * gWg2); bg2 = dlarray(extractdata(bg2) - lr * gbg2);

    if it == 1
        f0 = extractdata(fake);
        s = 0; for n = 1:Nreal; s = s + f0(n); end
        initial_fake_mean = s / Nreal;
    end
end

% Final D loss + 64-sample mean.
final_d_loss = 0;
z   = randn(1, Nreal);
z_dl = dlarray(z);
fake   = Wg2 * relu(Wg1 * z_dl + bg1) + bg2;
d_real = Wd2 * relu(Wd1 * real_dl + bd1) + bd2;
d_fake = Wd2 * relu(Wd1 * fake    + bd1) + bd2;
Lf = extractdata(mse(d_real, one_row) + mse(d_fake, zero_row));
final_d_loss = Lf(1);

zf     = randn(1, 64);
fake_f = extractdata(Wg2 * relu(Wg1 * dlarray(zf) + bg1) + bg2);
sum_f = 0; for n = 1:64; sum_f = sum_f + fake_f(n); end
final_fake_mean = sum_f / 64;

fprintf('initial D loss x10 rounds to %.0f\n', round(10 * initLoss));
fprintf('final D loss x10 rounds to %.0f\n', round(10 * final_d_loss));
fprintf('initial generator mean (x10) rounds to %.0f\n', round(10 * initial_fake_mean));
fprintf('final generator mean (x10) rounds to %.0f\n', round(10 * final_fake_mean));
fprintf('target mean (x10) = 20\n');
