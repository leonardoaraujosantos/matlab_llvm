% dl_groupnorm_bn_ema.m — Gating for GroupNorm + EMA-tracked BN.

% =====  GroupNorm  =================================================
% Input X: 4 x 4 x 4 x 1, G = 2 groups of 2 channels each.  Each group's
% (H, W, C/G) population gets its own (μ, σ).
X = zeros(4, 4, 4, 1);
for c = 1:4
    for h = 1:4
        for w = 1:4
            X(h, w, c, 1) = c * 10 + h + w * 0.1;
        end
    end
end
gamma = ones(1, 4);
beta  = zeros(1, 4);
Xdl = dlarray(X);
Y = groupnorm(Xdl, dlarray(gamma), dlarray(beta), 2);
Yv = extractdata(Y);

% Verify per-(group, sample) stats: each group's M = H*W*(C/G) = 32
% cells must have mean ≈ 0 and var ≈ 1.
for g = 1:2
    c_lo = (g - 1) * 2 + 1;  c_hi = c_lo + 1;
    s = 0; ss = 0; cnt = 0;
    for c = c_lo:c_hi
        for h = 1:4
            for w = 1:4
                v = Yv(h, w, c, 1);
                s = s + v; ss = ss + v*v; cnt = cnt + 1;
            end
        end
    end
    mu = s / cnt;  vr = ss / cnt - mu * mu;
    fprintf('dl_groupnorm_bn_ema: GN g%.0f mean=%.4f var=%.4f\n', g, mu, vr);
end

% Backward sanity — γ-grad finite.
Tg = dlarray(zeros(4, 4, 4, 1));
loss_gn = mse(Y, Tg);
gG = dlgradient(loss_gn, dlarray(gamma));
fprintf('dl_groupnorm_bn_ema: GN sum(gG)=%.4f\n', sum(gG));


% =====  EMA-tracked BN  ============================================
% Two-step "training" over the same input; verify running stats move
% from their initial (0, 1) values toward the batch stats (μ_c, σ²_c)
% via momentum = 0.5.
Xbn = zeros(2, 2, 2, 2);
for n = 1:2
    for c = 1:2
        for h = 1:2
            for w = 1:2
                Xbn(h, w, c, n) = (n - 1) * 10 + (c - 1) * 5 + h + w * 0.1;
            end
        end
    end
end
gamma_bn = ones(1, 2);
beta_bn  = zeros(1, 2);
run_mean = zeros(1, 2);
run_var  = ones(1, 2);
mom = 0.5;

Xbn_dl = dlarray(Xbn);
Gbn = dlarray(gamma_bn); Bbn = dlarray(beta_bn);
RM  = dlarray(run_mean); RV  = dlarray(run_var);

Y_bn = batchnorm_train(Xbn_dl, Gbn, Bbn, RM, RV, mom);

% Read updated running stats back out via extractdata.
rm_v = extractdata(RM);
rv_v = extractdata(RV);
fprintf('dl_groupnorm_bn_ema: BN train run_mean = %.3f %.3f\n', ...
        rm_v(1), rm_v(2));
fprintf('dl_groupnorm_bn_ema: BN train run_var  = %.3f %.3f\n', ...
        rv_v(1), rv_v(2));

% Inference-mode forward with the updated stats (no autodiff backward).
Y_ev = batchnorm_eval(Xbn_dl, Gbn, Bbn, RM, RV);
Yev = extractdata(Y_ev);
fprintf('dl_groupnorm_bn_ema: bn_eval (1,1,1,1)=%.3f (2,2,2,2)=%.3f\n', ...
        Yev(1, 1, 1, 1), Yev(2, 2, 2, 2));

fprintf('dl_groupnorm_bn_ema: PASS\n');
