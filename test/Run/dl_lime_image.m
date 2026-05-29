% dl_lime_image — image-domain LIME (Local Interpretable Model-agnostic
% Explanations, Ribeiro et al. 2016) on a tiny CNN.
%
% LIME assigns importance to "superpixels" (regions of the input) by:
%   1. partitioning the input into S superpixels (here: 4 quadrants of
%      a 4x4 image),
%   2. sampling N random masks (binary per-superpixel on/off),
%   3. running the network on each masked input, collecting (mask, y),
%   4. fitting a linear surrogate y ≈ w·mask, where w_s is the
%      importance of superpixel s.
%
% This is the smallest tractable demo: superpixels = 2x2 blocks of a
% 4x4 input, network = the vertical-bar detector from dl_gradcam_image.
% The vertical bar lives in (1:4, 2), so it crosses the top-right
% (rows 1-2, col 2) and bottom-right (rows 3-4, col 2) superpixels.
% Those two should dominate the LIME weights.

% Network: same vertical-bar detector.
W = zeros(3, 3, 1, 1);
for i = 1:3, W(i, 2, 1, 1) = 1.0; end

% Input + superpixel definition.
X = zeros(4, 4, 1, 1);
for i = 1:4, X(i, 2, 1, 1) = 1.0; end

% 4 superpixels (2x2 quadrants of the 4x4 input).  superpixel-id at
% each (h, w) cell: top-left = 1, top-right = 2, bottom-left = 3,
% bottom-right = 4.
sp_id = zeros(4, 4);
for h = 1:4
    for w = 1:4
        sh = 1; if h > 2, sh = 2; end
        sw = 1; if w > 2, sw = 2; end
        sp_id(h, w) = (sh - 1) * 2 + sw;
    end
end

% Generate 32 deterministic mask samples + collect (mask, y) pairs.
N = 32;
S = 4;
mask_data = zeros(N, S);
y_data = zeros(N, 1);
rng(0);
for n = 1:N
    % Use rand() per superpixel — gives diverse mask patterns so the
    % least-squares system is well-conditioned.
    for s = 1:S
        u = rand(1, 1);
        if u(1, 1) > 0.5
            mask_data(n, s) = 1.0;
        else
            mask_data(n, s) = 0.0;
        end
    end
    % Build masked input.
    Xm = zeros(4, 4, 1, 1);
    for h = 1:4
        for w = 1:4
            if mask_data(n, sp_id(h, w)) > 0.5
                Xm(h, w, 1, 1) = X(h, w, 1, 1);
            end
        end
    end
    Ym = conv2d_batch(Xm, W);
    s = 0.0;
    for h = 1:2
        for w = 1:2
            s = s + Ym(h, w, 1, 1);
        end
    end
    y_data(n, 1) = s;
end

% Fit linear surrogate y = mask * w via least squares: w = (M'M)^-1 M' y.
% Build a tiny 4x4 normal equation system; solve by closed-form 4x4
% Gauss-Jordan since the dataset is small + deterministic.
MtM = zeros(S, S);
Mty = zeros(S, 1);
for i = 1:S
    for j = 1:S
        v = 0.0;
        for n = 1:N
            v = v + mask_data(n, i) * mask_data(n, j);
        end
        MtM(i, j) = v;
    end
    v = 0.0;
    for n = 1:N
        v = v + mask_data(n, i) * y_data(n, 1);
    end
    Mty(i, 1) = v;
end

% Solve via mldivide.
w_lime = MtM \ Mty;

fprintf('dl_lime_image: w_lime = [%.3f %.3f %.3f %.3f]\n', ...
        w_lime(1), w_lime(2), w_lime(3), w_lime(4));

% Importance check: superpixels covering the bar (top-left = cols 1-2,
% bottom-left = cols 1-2) carry signal; top-right + bottom-right (cols
% 3-4) cover empty input and should have ~0 weight.
% Superpixel ids: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right
bar_weight  = w_lime(1) + w_lime(3);   % left quadrants — bar lives here
empty_weight = w_lime(2) + w_lime(4);  % right quadrants — empty input
fprintf('dl_lime_image: bar-covering weight sum   = %.3f\n', bar_weight);
fprintf('dl_lime_image: empty-quad  weight sum   = %.3f\n', empty_weight);

if bar_weight > 2.5 && abs(empty_weight) < 0.5
    fprintf('dl_lime_image: PASS\n');
else
    fprintf('dl_lime_image: FAIL\n');
end
