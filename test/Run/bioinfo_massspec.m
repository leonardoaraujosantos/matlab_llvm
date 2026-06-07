% Bioinformatics Toolbox Tier-6 — mass-spec preprocessing + learning helpers.
mz = (1:20);
y = 0.5 * ones(1, 20);
y(5) = 10; y(12) = 8;            % two peaks on a flat baseline

yb = msbackadj(mz, y);          % baseline removal
pk = mspeaks(mz, yb);           % peak detection
fprintf('peaks: %.0f at mz %.0f and %.0f\n', size(pk,1), pk(1,1), pk(2,1));

yn = msnorm(mz, y);
fprintf('msnorm max: %.2f\n', max(yn));

rs = msresample(mz, y, 10);
fprintf('resampled to %.0f points, mz1=%.1f\n', size(rs,1), rs(1,1));

% Feature ranking: row 1 separates the two groups, row 2 is constant.
Xf = [1 1 1 9 9 9; 5 5 5 5 5 5];
g  = [1 1 1 2 2 2];
r  = rankfeatures(Xf, g);
fprintf('top feature: %.0f\n', r(1));

% Deterministic k-fold indices.
cv = crossvalind('Kfold', 6, 3);
fprintf('folds: %.0f %.0f %.0f %.0f %.0f %.0f\n', ...
        cv(1), cv(2), cv(3), cv(4), cv(5), cv(6));

% KNN imputation of a missing value.
Xi = [1 2 3; 1 2 0/0; 9 9 9];
Yi = knnimpute(Xi);
fprintf('imputed(2,3)=%.1f\n', Yi(2,3));
