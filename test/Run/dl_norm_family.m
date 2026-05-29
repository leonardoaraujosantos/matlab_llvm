% dl_norm_family.m — InstanceNorm + RMSNorm + multi-axis reductions.

% ===== InstanceNorm =====
% Input X: 4 x 4 x 3 x 2 (4x4 spatial, 3 channels, 2 samples).
X = zeros(4, 4, 3, 2);
for n = 1:2
    for c = 1:3
        for h = 1:4
            for w = 1:4
                X(h, w, c, n) = (n - 1) * 100 + (c - 1) * 20 + h + w * 0.1;
            end
        end
    end
end
gamma = ones(1, 3);
beta  = zeros(1, 3);
Y_in = instancenorm(dlarray(X), dlarray(gamma), dlarray(beta));
Yv = extractdata(Y_in);

% Each (c, n) slice should have mean ≈ 0, var ≈ 1.
for n = 1:2
    for c = 1:3
        s = 0; ss = 0;
        for h = 1:4
            for w = 1:4
                v = Yv(h, w, c, n);
                s = s + v; ss = ss + v*v;
            end
        end
        mu = s / 16;  vr = ss / 16 - mu * mu;
        fprintf('dl_norm_family: IN n=%.0f c=%.0f mean=%.4f var=%.4f\n', n, c, mu, vr);
    end
end

% ===== RMSNorm =====
% 4x3 input, normalize along dim=1.
R = [3.0 4.0 5.0;
     1.5 2.5 3.5;
     0.5 1.5 2.5;
    -1.0  0.0  1.0];
gamma_r = ones(1, 4);
Y_r = rmsnorm(dlarray(R), dlarray(gamma_r), 1);
Yrv = extractdata(Y_r);
% Each column should have RMS ≈ 1 (no mean subtraction).
for t = 1:3
    ss = 0;
    for i = 1:4
        ss = ss + Yrv(i, t)^2;
    end
    fprintf('dl_norm_family: RMS col%.0f rms2=%.4f\n', t, ss / 4);
end

% ===== Multi-axis sum =====
% sum over [1 2 3] of a 2x2x2x3: collapse first three axes -> 1x1x1x3
% (then trailing-drop to a 1x3 row).
M = zeros(2, 2, 2, 3);
for n = 1:3
    for h = 1:2, for w = 1:2, for c = 1:2
        M(h, w, c, n) = n;  % each sample has constant value n
    end, end, end
end
S = sum(M, [1 2 3]);
fprintf('dl_norm_family: sum(M,[1 2 3]) ndims=%.0f size=%.0f %.0f\n', ...
        ndims(S), size(S, 1), size(S, 2));
% Each sample's total: 8 cells × n.  Expect [8 16 24].
fprintf('dl_norm_family: sum totals = %.0f %.0f %.0f\n', S(1), S(2), S(3));

% mean over [3 4] of a 4x4x2x2: collapse last two axes -> 4x4.
M2 = zeros(4, 4, 2, 2);
for h = 1:4, for w = 1:4
    M2(h, w, 1, 1) = h + w * 0.1;
    M2(h, w, 2, 1) = (h + w * 0.1) * 2;
    M2(h, w, 1, 2) = (h + w * 0.1) * 3;
    M2(h, w, 2, 2) = (h + w * 0.1) * 4;
end, end
Mm = mean(M2, [3 4]);
% Each (h, w) cell is the mean of (1, 2, 3, 4) * (h+w*0.1) = 2.5*(h+w*0.1).
fprintf('dl_norm_family: mean(M2,[3 4]) ndims=%.0f size=%.0f %.0f\n', ...
        ndims(Mm), size(Mm, 1), size(Mm, 2));
% Sample cell (1, 1): 2.5 * 1.1 = 2.75.
fprintf('dl_norm_family: Mm(1,1)=%.3f Mm(4,4)=%.3f\n', Mm(1, 1), Mm(4, 4));

fprintf('dl_norm_family: PASS\n');
