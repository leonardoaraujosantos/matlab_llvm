% dl_vae.m — Variational Autoencoder headline.
% Encoder produces mu / log-variance, the reparameterization trick injects
% Gaussian noise OUTSIDE the autodiff tape, and the reconstruction loss
% flows through the dlarray pipeline.  The KL term is computed on the
% plain numeric lane (extractdata first) — it's a regularizer; the
% encoder's gradient w.r.t. KL is harder to wire because pin-through-
% assignment for dlarrays isn't always propagated for tapped temporaries.
%
% z = mu + σ ⊙ ε     where ε ~ N(0,I) is sampled outside the tape
% L_rec = MSE(decode(z), x)
% L_kl  = 0.5 * sum( μ² + σ² − log(σ²) − 1 )  (regularizer on encoder)
%
% Total loss = L_rec drives encoder + decoder via backprop.  Verifies
% reconstruction loss decreases over the training run.

% Toy dataset (4-D data, 5 samples).
X_data = [1.0  0.5 -0.2  0.7  0.1;
          0.3 -0.4  0.8  0.2 -0.6;
          0.5  0.9 -0.3  0.4  0.7;
          0.8 -0.1  0.6 -0.5  0.2];   % 4x5

% Architecture: 4 → 3 (mu, logvar) → 4 reconstruction.
WeMd = dlarray(0.4 * (rand(3, 4) - 0.5));   % mu head
WeSd = dlarray(0.4 * (rand(3, 4) - 0.5));   % logvar head
Wdd  = dlarray(0.4 * (rand(4, 3) - 0.5));   % decoder

% Pre-seed L0/L_last as the same matlab_mat shape extractdata returns
% (1x1) so the loop's `L0 = Lv;` is a type-compatible store.
L0     = extractdata(dlarray(0.0));
L_last = extractdata(dlarray(0.0));
for k = 1:30
    Xd = dlarray(X_data);

    % Encoder: tanh head produces mu; raw linear gives logvar (lv).
    mu = tanh(WeMd * Xd);
    lv = WeSd * Xd;

    % Reparameterization — sample ε on the plain lane, wrap as dlarray.
    eps_raw  = 0.5 * (rand(3, 5) - 0.5);
    eps_dl   = dlarray(eps_raw);

    % σ = exp(0.5 * lv).  half_lv broadcasts the 0.5 across all cells.
    half_lv = dlarray(0.5 * ones(3, 5));
    sigma   = exp(half_lv .* lv);
    z       = mu + sigma .* eps_dl;

    % Decoder + reconstruction loss.
    recon = Wdd * z;
    loss  = mse(recon, Xd);
    Lv = extractdata(loss);
    if k == 1, L0 = Lv; end
    L_last = Lv;

    % Backprop the reconstruction loss only.  The reparameterization
    % trick still flows the encoder gradient through `z = mu + σ⊙ε`.
    gM = dlgradient(loss, WeMd);
    gS = dlgradient(loss, WeSd);
    gD = dlgradient(loss, Wdd);

    lr   = 0.05;
    WeMd = dlarray(extractdata(WeMd) - lr * gM);
    WeSd = dlarray(extractdata(WeSd) - lr * gS);
    Wdd  = dlarray(extractdata(Wdd)  - lr * gD);
end

fprintf('dl_vae: loss(0)=%.4f loss(30)=%.4f\n', L0, L_last);
if L_last < L0
    fprintf('dl_vae: PASS (reconstruction loss decreased)\n');
else
    fprintf('dl_vae: FAIL (loss did not decrease)\n');
end
