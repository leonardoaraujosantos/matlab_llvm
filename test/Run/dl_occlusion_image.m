% dl_occlusion_image — image-domain occlusionSensitivity via the
% Tier-C rank-4 conv kernel.
%
% Sweep a small occluding patch (zero block) over each input location;
% measure how much the network's predicted score for a target class
% drops vs the unoccluded baseline.  Pixels where occlusion causes a
% large score drop are the "important" pixels for that class.
%
% Setup: same 4x4 vertical-bar input + vertical-detector kernel as
% dl_gradcam_image.m.  Baseline score = sum(conv(X, W)) for filter 1.
% Occlude each (i, j) by zeroing input pixel (i, j) and re-running the
% forward pass; track the maximum score drop.  Pixels where the bar
% lives (col 2) should produce the biggest drop.

X = zeros(4, 4, 1, 1);
for i = 1:4
    X(i, 2, 1, 1) = 1.0;     % vertical bar
end

W = zeros(3, 3, 1, 1);
for i = 1:3
    W(i, 2, 1, 1) = 1.0;     % vertical detector
end

% Baseline forward score.
Y0 = conv2d_batch(X, W);
% Y0 is 2x2x1x1.  Score = sum of all entries.
baseline = 0.0;
for h = 1:2
    for w = 1:2
        baseline = baseline + Y0(h, w, 1, 1);
    end
end
fprintf('dl_occlusion_image: baseline score = %.2f\n', baseline);

% Sweep occlusion across the 4x4 input.  Track per-pixel score drop.
drops = zeros(4, 4);
max_drop = 0.0;
max_i = 0; max_j = 0;
for i = 1:4
    for j = 1:4
        % Build occluded input.
        Xo = zeros(4, 4, 1, 1);
        for h = 1:4
            for w = 1:4
                Xo(h, w, 1, 1) = X(h, w, 1, 1);
            end
        end
        Xo(i, j, 1, 1) = 0.0;
        Yo = conv2d_batch(Xo, W);
        s = 0.0;
        for h = 1:2
            for w = 1:2
                s = s + Yo(h, w, 1, 1);
            end
        end
        d = baseline - s;
        drops(i, j) = d;
        if d > max_drop
            max_drop = d;
            max_i = i;
            max_j = j;
        end
    end
end

fprintf('dl_occlusion_image: peak drop = %.2f at (%d, %d)\n', max_drop, max_i, max_j);
fprintf('dl_occlusion_image: drops(1, :) = %.0f %.0f %.0f %.0f\n', ...
        drops(1, 1), drops(1, 2), drops(1, 3), drops(1, 4));
fprintf('dl_occlusion_image: drops(2, :) = %.0f %.0f %.0f %.0f\n', ...
        drops(2, 1), drops(2, 2), drops(2, 3), drops(2, 4));

% PASS criteria: peak drop is in column 2 (where the bar is); pixels
% in cols 1/3/4 produce zero drop (no signal there).
if max_j == 2 && drops(1, 1) == 0 && drops(1, 3) == 0 && drops(1, 4) == 0
    fprintf('dl_occlusion_image: PASS\n');
else
    fprintf('dl_occlusion_image: FAIL\n');
end
