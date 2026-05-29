% stats_tsne.m — t-SNE non-linear embedding headline.
% Stats T6.2 carve-down close.  Embeds 3 well-separated 4-D clusters
% (10 points each, σ=0.05) into 2-D and verifies the resulting
% embedding preserves the cluster structure: within-cluster mean pair-
% wise distance is small relative to between-cluster mean distance.

% Three blobs in 4-D space, centred at distinct corners of a 4-D box.
rng_seed = 7;
X = zeros(30, 4);
for i = 1:10
    X(i,        :) = [ 1.0  1.0  0.0  0.0] + 0.05 * (rand(1, 4) - 0.5);
end
for i = 1:10
    X(10 + i,   :) = [-1.0 -1.0  0.0  0.0] + 0.05 * (rand(1, 4) - 0.5);
end
for i = 1:10
    X(20 + i,   :) = [ 0.0  0.0  1.0  1.0] + 0.05 * (rand(1, 4) - 0.5);
end

Y = tsne(X);                       % 30x2 embedding

% Inspect cluster cohesion in the embedding via per-cell access.
% Each true cluster's centroid in Y; mean within-cluster distance
% should be < mean between-cluster centroid distance.
c1x = 0; c1y = 0; c2x = 0; c2y = 0; c3x = 0; c3y = 0;
for i = 1:10
    c1x = c1x + Y(i,        1); c1y = c1y + Y(i,        2);
    c2x = c2x + Y(10 + i,   1); c2y = c2y + Y(10 + i,   2);
    c3x = c3x + Y(20 + i,   1); c3y = c3y + Y(20 + i,   2);
end
c1x = c1x / 10; c1y = c1y / 10;
c2x = c2x / 10; c2y = c2y / 10;
c3x = c3x / 10; c3y = c3y / 10;

d_within = 0.0;
for i = 1:10
    dx = Y(i, 1) - c1x; dy = Y(i, 2) - c1y;
    d_within = d_within + sqrt(dx*dx + dy*dy);
    dx = Y(10 + i, 1) - c2x; dy = Y(10 + i, 2) - c2y;
    d_within = d_within + sqrt(dx*dx + dy*dy);
    dx = Y(20 + i, 1) - c3x; dy = Y(20 + i, 2) - c3y;
    d_within = d_within + sqrt(dx*dx + dy*dy);
end
d_within = d_within / 30;

d12 = sqrt((c1x - c2x)^2 + (c1y - c2y)^2);
d13 = sqrt((c1x - c3x)^2 + (c1y - c3y)^2);
d23 = sqrt((c2x - c3x)^2 + (c2y - c3y)^2);
d_between = (d12 + d13 + d23) / 3;

fprintf('stats_tsne: within=%.3f between=%.3f ratio=%.2f\n', ...
        d_within, d_between, d_between / d_within);
if d_between > 2.0 * d_within
    fprintf('stats_tsne: PASS (clusters separated)\n');
else
    fprintf('stats_tsne: FAIL (cluster ratio too small)\n');
end
