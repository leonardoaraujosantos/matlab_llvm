% dl_autoencoder.m — Deep Learning T5.5: train an autoencoder over the
% autodiff.  The encoder compresses a 4-D input through a 2-D latent
% bottleneck; the decoder reconstructs.  All four parameter tensors live in
% the autodiff and are updated jointly.
%
% This is the "VAE without the reparameterization trick" — the carved
% piece is the noise-sampling layer that takes the encoder's `(mu,
% logvar)` to `z = mu + exp(0.5*logvar)*epsilon`.  The reparameterization
% needs a multi-head encoder output + a stochastic sampling step inside the
% autodiff, both of which compose from existing ops once two-output
% function returns ship as a Sema feature (see roadmap §5.5).

rng(0);
D = 4; Z = 2;
N = 6;

% A small set of input vectors clustered along two directions in 4-D space.
Xd = [ 1.0  1.1  0.9 -1.0 -1.1 -0.9;
       0.5  0.4  0.6  0.4  0.5  0.6;
      -0.3 -0.2 -0.4  0.7  0.6  0.8;
       0.8  0.9  0.7 -0.6 -0.5 -0.7 ];

% Trainable parameters.
We = dlarray(0.3 * randn(Z, D)); be = dlarray(zeros(Z, 1));
Wd = dlarray(0.3 * randn(D, Z)); bd = dlarray(zeros(D, 1));

X = dlarray(Xd);

lr = 0.05;
nIter = 200;
initLoss = 0;
for it = 1:nIter
    z_lat = tanh(We * X + be);
    Xhat  = Wd * z_lat + bd;
    L     = mse(Xhat, X);
    Lv = extractdata(L);
    if it == 1; initLoss = Lv(1); end

    gWe = dlgradient(L, We); gbe = dlgradient(L, be);
    gWd = dlgradient(L, Wd); gbd = dlgradient(L, bd);
    We = dlarray(extractdata(We) - lr * gWe); be = dlarray(extractdata(be) - lr * gbe);
    Wd = dlarray(extractdata(Wd) - lr * gWd); bd = dlarray(extractdata(bd) - lr * gbd);
end

% Final reconstruction + diagnostic.
z_lat = tanh(We * X + be);
Xhat  = Wd * z_lat + bd;
Lf = extractdata(mse(Xhat, X)); finalLoss = Lf(1);

% Mean reconstruction error (per element) for the first sample.
Xh = extractdata(Xhat);
err1 = 0;
for d = 1:D
    err1 = err1 + abs(Xh(d, 1) - Xd(d, 1));
end
mean_err_x1 = err1 / D;

fprintf('initial autoencoder loss rounds to %.0f\n', round(initLoss * 100));
fprintf('final autoencoder loss rounds to %.0f\n', round(finalLoss * 100));
fprintf('mean per-element recon error on X(:,1) (x100) rounds to %.0f\n', round(100 * mean_err_x1));
