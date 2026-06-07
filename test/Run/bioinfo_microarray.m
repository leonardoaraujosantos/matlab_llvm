% Bioinformatics Toolbox Tier-6 — microarray normalization / filtering /
% clustering + the DataMatrix container.
X = [10 20 30; 40 50 60; 7 8 9; 100 100 100; 11 21 31];

dm = DataMatrix(X);
fprintf('DataMatrix: %.0f x %.0f\n', dm.NRows, dm.NCols);

Q = quantilenorm(dm.Data);
fprintf('qnorm row1: %.1f %.1f %.1f\n', Q(1,1), Q(1,2), Q(1,3));

m = mean(manorm(X));
fprintf('manorm col means: %.2f %.2f %.2f\n', m(1), m(2), m(3));

F = genevarfilter(X);
fprintf('varfilter rows: %.0f -> %.0f\n', size(X,1), size(F,1));

ord = clustergram(X);
fprintf('cluster leaf order: %.0f %.0f %.0f %.0f %.0f\n', ...
        ord(1), ord(2), ord(3), ord(4), ord(5));
