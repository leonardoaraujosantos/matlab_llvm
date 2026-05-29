% Deep Learning T5.5 gating test — autoencoder over the autodiff.  Encoder
% compresses a 4-D input to a 2-D latent; decoder reconstructs.  All four
% parameter tensors (encoder W/b, decoder W/b) live in the autodiff and are
% trained jointly by `dlgradient` for each.
%
% Gating signal: (a) reconstruction loss strictly drops, (b) gradient is
% non-zero for both encoder and decoder weights.

rng(0);
D = 4; Z = 2;

% Trainable parameters.  Use `tanh` activation (always non-zero gradient
% — `relu` kills the gradient signal in the random-init regime here).
We = dlarray(0.5 * randn(Z, D)); be = dlarray(zeros(Z, 1));
Wd = dlarray(0.5 * randn(D, Z)); bd = dlarray(zeros(D, 1));

X = dlarray(randn(D, 1));

% One forward to establish initial loss.
z_lat = tanh(We * X + be);
Xhat  = Wd * z_lat + bd;
L0    = mse(Xhat, X);
L0v   = extractdata(L0); initLoss = L0v(1);

% Five SGD iters.
for it = 1:5
    z_lat = tanh(We * X + be);
    Xhat  = Wd * z_lat + bd;
    L     = mse(Xhat, X);
    gWe   = dlgradient(L, We); gbe = dlgradient(L, be);
    gWd   = dlgradient(L, Wd); gbd = dlgradient(L, bd);
    We = dlarray(extractdata(We) - 0.05 * gWe); be = dlarray(extractdata(be) - 0.05 * gbe);
    Wd = dlarray(extractdata(Wd) - 0.05 * gWd); bd = dlarray(extractdata(bd) - 0.05 * gbd);
end

% Final loss.
z_lat = tanh(We * X + be);
Xhat  = Wd * z_lat + bd;
Lf = extractdata(mse(Xhat, X)); finalLoss = Lf(1);

loss_drop = 0;
if finalLoss < initLoss
    loss_drop = 1;
end

% Gradient magnitudes (we re-derive on the FINAL network to ensure
% non-trivial values made it through both encoder and decoder).
z2  = tanh(We * X + be);
Xh2 = Wd * z2 + bd;
L2  = mse(Xh2, X);
gWe_f = dlgradient(L2, We);
gWd_f = dlgradient(L2, Wd);
m_e = sum(sum(gWe_f .* gWe_f));
m_d = sum(sum(gWd_f .* gWd_f));

% A non-zero gradient sum-of-squares is enough — even a small magnitude
% (post-training) counts as "the path is wired".
both_learn = 0;
if m_e > 1e-12
    if m_d > 1e-12
        both_learn = 1;
    end
end

fprintf('autoencoder loss drops = %.0f\n', loss_drop);
fprintf('encoder + decoder both receive gradient = %.0f\n', both_learn);
