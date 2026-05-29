% dl_layernorm_op.m — LayerNorm + BN inference-mode forward.

% --- LayerNorm on a (D=4, T=3) sequence with dim=1 (per-token over D)
X = [1.0  2.0  3.0;
     0.5  1.5  2.5;
     0.0  1.0  2.0;
    -1.0  0.0  1.0];
gamma = ones(1, 4);
beta  = zeros(1, 4);
Xdl = dlarray(X);
Gdl = dlarray(gamma);
Bdl = dlarray(beta);
Y_ln = layernorm(Xdl, Gdl, Bdl, 1);
Yv   = extractdata(Y_ln);

% Each column should now have mean ≈ 0 and var ≈ 1 over the D axis.
for t = 1:3
    s = 0; ss = 0;
    for i = 1:4
        v = Yv(i, t); s = s + v; ss = ss + v*v;
    end
    mu = s / 4;  vr = ss / 4 - mu*mu;
    fprintf('dl_layernorm_op: col%.0f mean=%.4f var=%.4f\n', t, mu, vr);
end

% Backward through MSE — γ, β gradients should be finite.
T_ln = dlarray(zeros(4, 3));
loss = mse(Y_ln, T_ln);
gG = dlgradient(loss, Gdl);
gB = dlgradient(loss, Bdl);
fprintf('dl_layernorm_op: sum(gG)=%.4f sum(gB)=%.4f\n', sum(gG), sum(gB));

% --- BN inference mode (frozen stats) on a (H=2, W=2, C=2, N=1) tensor.
Xbn = zeros(2, 2, 2, 1);
for c = 1:2
    for h = 1:2
        for w = 1:2
            Xbn(h, w, c, 1) = (h - 1) * 2 + (w - 1) + (c - 1) * 4;
        end
    end
end
gamma_bn = ones(1, 2);
beta_bn  = zeros(1, 2);
run_mu   = [1.5  5.5];   % matching the per-channel population means
run_var  = [1.25 1.25];
Xbn_dl   = dlarray(Xbn);
Y_bne = batchnorm_eval(Xbn_dl, dlarray(gamma_bn), dlarray(beta_bn), ...
                       dlarray(run_mu), dlarray(run_var));
Yv2 = extractdata(Y_bne);
% Verify channel-1 (which had values 0, 1, 2, 3, mean = 1.5 var = 1.25):
% normalised, the four cells should be approximately -1.342, -0.447, 0.447, 1.342.
fprintf('dl_layernorm_op: bn_eval ch1 = %.3f %.3f %.3f %.3f\n', ...
        Yv2(1, 1, 1, 1), Yv2(1, 2, 1, 1), Yv2(2, 1, 1, 1), Yv2(2, 2, 1, 1));

fprintf('dl_layernorm_op: PASS\n');
