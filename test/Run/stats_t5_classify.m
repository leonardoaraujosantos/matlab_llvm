% Statistics Toolbox Tier-5 — supervised classification.
% Deterministic 3-class, 2-feature, well-separated data.
X = [1 1; 1.2 0.9; 0.8 1.1; 1.1 1.0; ...
     5 5; 5.2 4.8; 4.9 5.1; 5.1 5.0; ...
     1 9; 1.1 8.9; 0.9 9.1; 1.0 9.0];
y = [1;1;1;1; 2;2;2;2; 3;3;3;3];
Xtest = [1.0 1.0; 5.0 5.0; 1.0 9.0];
mk = fitcknn(X, y);   pk = predict(mk, Xtest);
fprintf('knn  %.0f %.0f %.0f\n', pk(1), pk(2), pk(3));
mn = fitcnb(X, y);    pn = predict(mn, Xtest);
fprintf('nb   %.0f %.0f %.0f\n', pn(1), pn(2), pn(3));
md = fitcdiscr(X, y); pd = predict(md, Xtest);
fprintf('lda  %.0f %.0f %.0f\n', pd(1), pd(2), pd(3));
mt = fitctree(X, y);  pt = predict(mt, Xtest);
fprintf('tree %.0f %.0f %.0f\n', pt(1), pt(2), pt(3));
ms = fitcsvm([1 1;1.2 0.9;0.8 1.1; 5 5;5.2 4.8;4.9 5.1], [1;1;1;2;2;2]);
ps = predict(ms, [1 1; 5 5]);
fprintf('svm  %.0f %.0f\n', ps(1), ps(2));
me = fitcecoc(X, y);  pe = predict(me, Xtest);
fprintf('ecoc %.0f %.0f %.0f\n', pe(1), pe(2), pe(3));
ptr = predict(md, X);
Cm = confusionmat(y, ptr);
fprintf('conf %.0f %.0f %.0f\n', Cm(1,1), Cm(2,2), Cm(3,3));
