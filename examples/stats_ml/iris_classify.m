% iris_classify.m — Statistics and Machine Learning Toolbox HEADLINE.
% ----------------------------------------------------------------------
% The canonical Fisher-iris pipeline, exercising the descriptive →
% unsupervised → supervised arc end to end:
%   summarise  ->  pca (reduce)  ->  kmeans (unsupervised check)
%              ->  fitcecoc (one-vs-one linear SVM)  ->  predict
%              ->  confusionmat + accuracy.
%
% The 150x4 dataset is generated from the real Fisher-iris class means and
% spreads (sepal length/width, petal length/width for setosa / versicolor /
% virginica) over the shipped rng-reproducible PRNG — a faithful stand-in
% that keeps the example self-contained (no data file).  No external
% dependency: PCA via a hand-coded symmetric eigensolver, k-means via
% Lloyd + k-means++, the SVM by squared-hinge minimization.
rng(1);
n = 50;
setosa     = [normrnd(5.0,0.35,n,1) normrnd(3.4,0.38,n,1) normrnd(1.5,0.17,n,1) normrnd(0.25,0.10,n,1)];
versicolor = [normrnd(5.9,0.52,n,1) normrnd(2.8,0.31,n,1) normrnd(4.3,0.47,n,1) normrnd(1.30,0.20,n,1)];
virginica  = [normrnd(6.6,0.64,n,1) normrnd(3.0,0.32,n,1) normrnd(5.6,0.55,n,1) normrnd(2.00,0.27,n,1)];
X = [setosa; versicolor; virginica];
y = [ones(n,1); 2*ones(n,1); 3*ones(n,1)];

% ----- descriptive summary --------------------------------------------
fprintf('feature means : %.2f %.2f %.2f %.2f\n', ...
        mean(X(:,1)), mean(X(:,2)), mean(X(:,3)), mean(X(:,4)));

% ----- PCA: how much variance lives in the first two components -------
[coeff, score, latent, ts, explained] = pca(X);
fprintf('PCA explained : PC1 %.1f%%  PC2 %.1f%%\n', explained(1), explained(2));

% ----- unsupervised check: k-means into 3 clusters --------------------
rng(2);
idx = kmeans(X, 3);
fprintf('kmeans silhouette = %.3f\n', mean(silhouette(X, idx)));

% ----- supervised: multiclass SVM (ECOC), score on the data ----------
mdl = fitcecoc(X, y);
yp  = predict(mdl, X);
Cm  = confusionmat(y, yp);
fprintf('confusion diag : %.0f %.0f %.0f  (of %.0f per class)\n', ...
        Cm(1,1), Cm(2,2), Cm(3,3), n);
acc = (Cm(1,1) + Cm(2,2) + Cm(3,3)) / (3*n);
fprintf('SVM accuracy   = %.1f%%\n', 100*acc);
