% Bioinformatics Toolbox Tier-4 — distances + phylogenetic trees.
%   seqpdist (p-distance + Jukes-Cantor), seqlinkage (UPGMA) -> phytree,
%   getnewickstr, phytree property reads, pdist (patristic), seqneighjoin.
seqs = {'MKTAYIAKQR', 'MKTAYIAKNR', 'MKTAYIPKNR'};
d = seqpdist(seqs);
fprintf('p-dist: %.3f %.3f %.3f\n', d(1), d(2), d(3));    % 0.1 0.2 0.1
dj = seqpdist(seqs, 'Jukes-Cantor');
fprintf('jc d12: %.4f\n', dj(1));

tr = seqlinkage(d, 'average', {'a', 'b', 'c'});
fprintf('leaves=%.0f branches=%.0f\n', tr.NumLeaves, tr.NumLeaves - 1);
disp('--- UPGMA newick ---');
disp(getnewickstr(tr));

pd = pdist(tr);                                            % patristic distances
fprintf('patristic: %.3f %.3f %.3f\n', pd(1), pd(2), pd(3));

trc = seqlinkage(d, 'complete', {'a', 'b', 'c'});
disp('--- complete-linkage newick ---');
disp(getnewickstr(trc));

tr2 = seqneighjoin(d, 'equivar', {'x', 'y', 'z'});
disp('--- neighbor-joining newick ---');
disp(getnewickstr(tr2));
