% Statistics Toolbox Tier-6 — ensembles (fitcensemble bagging / TreeBagger).
X = [1 1; 1.2 0.9; 0.8 1.1; 1.1 1.0; ...
     5 5; 5.2 4.8; 4.9 5.1; 5.1 5.0; ...
     1 9; 1.1 8.9; 0.9 9.1; 1.0 9.0];
y = [1;1;1;1; 2;2;2;2; 3;3;3;3];
Xt = [1.0 1.0; 5.0 5.0; 1.0 9.0];
rng(1); mb = fitcensemble(X, y); pb = predict(mb, Xt);
fprintf('bag    %.0f %.0f %.0f\n', pb(1), pb(2), pb(3));
rng(2); mf = TreeBagger(30, X, y); pf = predict(mf, Xt);
fprintf('forest %.0f %.0f %.0f\n', pf(1), pf(2), pf(3));
Cm = confusionmat(y, predict(mb, X));
fprintf('diag   %.0f %.0f %.0f\n', Cm(1,1), Cm(2,2), Cm(3,3));
