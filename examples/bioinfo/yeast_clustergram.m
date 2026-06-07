% yeast_clustergram.m — Bioinformatics Toolbox Phase-C headline (Tier-6).
% ----------------------------------------------------------------------
% The canonical microarray workflow: wrap a gene-expression matrix in a
% DataMatrix, quantile-normalize across samples, drop the low-variance genes,
% then hierarchically cluster the surviving genes and report the dendrogram
% leaf order.  All numeric work reuses the shipped reductions + the Tier-4
% UPGMA tree builder; no external dependency.
%
% Rows = genes, columns = samples (time points).  The synthetic data has two
% clear co-expression groups (rising vs falling) plus a flat housekeeping gene.

expr = [ ...
    1.0 2.0 4.0 8.0; ...     % gene 1 — rising
    1.2 2.1 3.9 7.8; ...     % gene 2 — rising (like gene 1)
    8.0 4.0 2.0 1.0; ...     % gene 3 — falling
    7.9 4.1 2.1 1.1; ...     % gene 4 — falling (like gene 3)
    5.0 5.0 5.0 5.0; ...     % gene 5 — flat (housekeeping)
    1.1 2.2 4.1 8.2];        % gene 6 — rising

dm = DataMatrix(expr);
fprintf('expression: %.0f genes x %.0f samples\n', dm.NRows, dm.NCols);

% Quantile-normalize the samples (columns).
Q = quantilenorm(dm.Data);
fprintf('quantile-normalized sample 1 mean = %.3f\n', mean(Q(:,1)));

% Remove the lowest-variance genes (the flat housekeeping gene goes first).
F = genevarfilter(Q);
fprintf('genes after variance filter: %.0f / %.0f\n', size(F,1), size(expr,1));

% Hierarchically cluster the surviving genes; the rising group and the
% falling group should each end up contiguous in the leaf order.
order = clustergram(F);
fprintf('dendrogram leaf order: ');
for i = 1:length(order)
    fprintf('%.0f ', order(i));
end
fprintf('\n');
