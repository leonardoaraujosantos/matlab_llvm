% ensemble_classify.m — Statistics Toolbox Tier-6: ensemble learning.
% ----------------------------------------------------------------------
% Bagged trees (`fitcensemble`) and a random forest (`TreeBagger`,
% bootstrap + random feature subset per split) on Fisher-iris-like data,
% compared with a single CART tree.  Ensembles average out the variance of
% individual trees, so they recover the versicolor/virginica boundary more
% reliably.
rng(1);
m = 50;
setosa     = [normrnd(5.0,0.35,m,1) normrnd(3.4,0.38,m,1) normrnd(1.5,0.17,m,1) normrnd(0.25,0.10,m,1)];
versicolor = [normrnd(5.9,0.52,m,1) normrnd(2.8,0.31,m,1) normrnd(4.3,0.47,m,1) normrnd(1.30,0.20,m,1)];
virginica  = [normrnd(6.6,0.64,m,1) normrnd(3.0,0.32,m,1) normrnd(5.6,0.55,m,1) normrnd(2.00,0.27,m,1)];
X = [setosa; versicolor; virginica];
y = [ones(m,1); 2*ones(m,1); 3*ones(m,1)];

% single CART tree
t  = fitctree(X, y);
Ct = confusionmat(y, predict(t, X));
at = 100 * (Ct(1,1) + Ct(2,2) + Ct(3,3)) / (3*m);
fprintf('single tree    accuracy = %.1f%%\n', at);

% bagged ensemble of trees
rng(2);
e  = fitcensemble(X, y);
Ce = confusionmat(y, predict(e, X));
ae = 100 * (Ce(1,1) + Ce(2,2) + Ce(3,3)) / (3*m);
fprintf('bagged trees   accuracy = %.1f%%\n', ae);

% random forest (random feature subset per split)
rng(3);
rf = TreeBagger(50, X, y);
Cf = confusionmat(y, predict(rf, X));
af = 100 * (Cf(1,1) + Cf(2,2) + Cf(3,3)) / (3*m);
fprintf('random forest  accuracy = %.1f%%\n', af);
