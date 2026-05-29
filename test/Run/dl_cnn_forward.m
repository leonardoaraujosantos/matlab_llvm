% dl_cnn_forward.m — CNN forward-pass headline.  First headline that
% exercises the Tier C rank-4 (H × W × C × N) tensor descriptor end
% to end: batched 2-D convolution + ReLU + global-average-pool +
% softmax classifier.
%
% Input batch X (5 × 5 × 1 × 2):  two 5×5 single-channel "images".
%   sample 1: a plus-sign pattern  (centre row + centre col bright)
%   sample 2: an X-pattern         (both diagonals bright)
%
% Filter set W (3 × 3 × 1 × 2):  two 3×3 single-channel detectors
%   filter 1: vertical bar (centre column)   -> fires on plus, mild on X
%   filter 2: diagonal      (NW-SE)          -> fires on X, mild on plus
%
% After conv → ReLU → mean across spatial axes → softmax we get the
% probability the input matches each detector.  Expectation:
%   sample 1 -> filter 1 (vertical bar) wins
%   sample 2 -> filter 2 (diagonal)     wins

% --- build the input batch (5x5x1x2) using rank-4 zeros + per-cell store
X = zeros(5, 5, 1, 2);
% sample 1: pure vertical bar (centre column only).
for i = 1:5
    X(i, 3, 1, 1) = 1.0;
end
% sample 2: pure NW-SE diagonal.
for i = 1:5
    X(i, i, 1, 2) = 1.0;
end

% --- build the filter bank (3x3x1x2) ---
% Discriminative filters: each peaks on its target pattern and suppresses
% the other.  Centre-column-pos / off-column-neg (vertical bar detector);
% diagonal-pos / off-diagonal-neg (NW-SE diagonal detector).
W = zeros(3, 3, 1, 2);
for i = 1:3
    for j = 1:3
        % vertical bar filter: +1 on centre column, -1 elsewhere
        if j == 2
            W(i, j, 1, 1) = 1.0;
        else
            W(i, j, 1, 1) = -1.0;
        end
        % diagonal filter: +1 on diagonal, -1 elsewhere
        if i == j
            W(i, j, 1, 2) = 1.0;
        else
            W(i, j, 1, 2) = -1.0;
        end
    end
end

% --- conv forward via Tier C batched convolution ---
Y = conv2d_batch(X, W);   % expected output shape: 3 x 3 x 2 x 2
fprintf('dl_cnn_forward: Y ndims=%.0f size=%.0f %.0f %.0f %.0f\n', ...
        ndims(Y), size(Y, 1), size(Y, 2), size(Y, 3), size(Y, 4));

% --- ReLU + global-average pool per-(filter, sample) ---
% Manual nested loop to keep the demo small.
score = zeros(2, 2);
for n = 1:2
    for k = 1:2
        acc = 0.0;
        for h = 1:3
            for w = 1:3
                v = Y(h, w, k, n);
                if v > 0
                    acc = acc + v;
                end
            end
        end
        score(k, n) = acc / 9;
    end
end
fprintf('dl_cnn_forward: score(bar.vert) =%.3f score(bar.diag) =%.3f\n', ...
        score(1, 1), score(2, 1));
fprintf('dl_cnn_forward: score(diag.vert)=%.3f score(diag.diag)=%.3f\n', ...
        score(1, 2), score(2, 2));

% --- softmax over the 2 classes for each sample ---
prob1 = exp(score(:, 1));  prob1 = prob1 / sum(prob1);
prob2 = exp(score(:, 2));  prob2 = prob2 / sum(prob2);
fprintf('dl_cnn_forward: bar  -> P(vert)=%.3f P(diag)=%.3f\n', prob1(1), prob1(2));
fprintf('dl_cnn_forward: diag -> P(vert)=%.3f P(diag)=%.3f\n', prob2(1), prob2(2));

if prob1(1) > prob1(2) && prob2(2) > prob2(1)
    fprintf('dl_cnn_forward: PASS (each input picks its matching filter)\n');
else
    fprintf('dl_cnn_forward: FAIL\n');
end
