% Statistics Toolbox Tier-4 — PCA + clustering.
% PCA on rank-2 data (col3 = col1+col2): 3rd component carries ~0 variance.
Xb = [1 4; 2 1; 3 5; 4 2; 5 6; 6 3; 7 2; 8 5];
c3 = Xb(:,1) + Xb(:,2);
Xa = [Xb c3];
[coeff, score, latent, ts, explained] = pca(Xa);
fprintf('lat3   %.4f\n', latent(3));            % ~0
fprintf('exp12  %.2f\n', explained(1)+explained(2));   % 100.00
% kmeans on two separated blobs (deterministic data)
P = [0 0; 0.2 0.1; -0.1 0.2; 5 5; 5.1 4.9; 4.9 5.2];
rng(7);
[idx, C] = kmeans(P, 2);
fprintf('same12 %.0f\n', abs(idx(1) - idx(2)));   % 0 (first 3 share a cluster)
fprintf('diff14 %.0f\n', abs(idx(1) - idx(4)));   % 1 (blob1 vs blob2 differ)
s = silhouette(P, idx);
fprintf('sil    %.3f\n', mean(s));               % high (well separated)
% distances
D = pdist2([0 0; 3 4], [0 0]);
fprintf('d2     %.1f\n', D(2));                   % 5.0
dv = pdist([0 0; 3 4]);
fprintf('pd     %.1f\n', dv(1));                  % 5.0
