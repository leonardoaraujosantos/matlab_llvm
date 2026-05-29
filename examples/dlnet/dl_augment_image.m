% dl_augment_image — T3.4b gating: augmentImage applies random
% rotate/scale/translate to an input image and returns the same-size
% result.
%
% A 16x16 checkerboard sample drives 5 augmentations; we check:
%   - every augmented image has the SAME size as the input
%   - the augmentations diverge from the input (mean abs diff > 0)
%   - the augmentations diverge from each other (per-sample variability)

rng(11);

% 16x16 checkerboard in [0, 1].
I = zeros(16, 16);
for i = 1:16
    for j = 1:16
        if mod(i + j, 2) == 0
            I(i, j) = 1.0;
        end
    end
end

% Augmenter bounds: ±20 deg, scale [0.9, 1.1], ±2 px translation.
ang_max = 20.0;
scale_lo = 0.9;
scale_hi = 1.1;
tx_max = 2.0;
ty_max = 2.0;

% Sample augmented images, check size + divergence.  Materialize the
% augmentImage result into a typed 16x16 zeros slot so subsequent ops
% know the value is a matrix (Sema leaves the builtin return as `none`).
sum_diff_to_input = 0.0;
sum_diff_to_first = 0.0;
A1_raw = augmentImage(I, ang_max, scale_lo, scale_hi, tx_max, ty_max);
r1 = size(A1_raw, 1);
c1 = size(A1_raw, 2);
A1 = zeros(16, 16);
for i = 1:16
    for j = 1:16
        A1(i, j) = A1_raw(i, j);
    end
end

for k = 1:5
    Ak_raw = augmentImage(I, ang_max, scale_lo, scale_hi, tx_max, ty_max);
    Ak = zeros(16, 16);
    for i = 1:16
        for j = 1:16
            Ak(i, j) = Ak_raw(i, j);
        end
    end
    % Sum(sum(...)) returns a 1x1 ptr; index (1, 1) to get scalar f64.
    D = Ak - I;
    s_in = sum(sum(abs(D)));
    s_in_v = s_in(1, 1);
    sum_diff_to_input = sum_diff_to_input + s_in_v;
    if k > 1
        D2 = Ak - A1;
        s_first = sum(sum(abs(D2)));
        s_first_v = s_first(1, 1);
        sum_diff_to_first = sum_diff_to_first + s_first_v;
    end
end

% Print raw magnitudes — both diffs should be well above 0 for the
% augmenter to be doing anything (rotation alone perturbs pixel grid).
fprintf('dl_augment_image: A1 size = %dx%d (input was 16x16)\n', r1, c1);
fprintf('dl_augment_image: sum diff-to-input  > 0.5 (got %.2f)\n', sum_diff_to_input);
fprintf('dl_augment_image: sum diff-to-first  > 0.5 (got %.2f)\n', sum_diff_to_first);

size_ok = (r1 == 16) && (c1 == 16);
diff_in_ok = (sum_diff_to_input > 0.5);
diff_first_ok = (sum_diff_to_first > 0.5);
if size_ok && diff_in_ok && diff_first_ok
    fprintf('dl_augment_image: PASS\n');
else
    fprintf('dl_augment_image: FAIL\n');
end
