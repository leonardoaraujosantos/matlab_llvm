% primate_phylotree.m — Bioinformatics Toolbox Phase-B headline (Tier-4).
% ----------------------------------------------------------------------
% Build a phylogenetic tree of four primates from a conserved mitochondrial
% sequence fragment: compute the all-pairs Jukes-Cantor distance with
% seqpdist, build a UPGMA tree with seqlinkage, and report the tree as a
% Newick string plus the patristic (tree-path) distances.  This is the
% canonical seqpdist -> seqlinkage -> phytree workflow; the tree is a
% class-pinned phytree object whose getnewickstr / pdist methods read its
% pointer + branch-length fields.  No external dependency, no network.

names = {'human', 'chimp', 'gorilla', 'orangutan'};
seqs  = { ...
    'ACGTTACGGATCGATTACGGCATTACGAGTCGATCGATCGATTAGCAT', ...  % human
    'ACGTTACGGATCGATTACGGCATTACGAGTCGATCGATCGATTAGCTT', ...  % chimp (1 diff)
    'ACGTTACGGATCGATAACGGCATTACGAGTCGATCGATCGATTAGGTT', ...  % gorilla
    'ACGTTACGAATCGATAACGCCATTACCAGTCGATCGTTCGATTAGGTA'};      % orangutan (outgroup)

% Pairwise evolutionary distances (Jukes-Cantor corrected).
D = seqpdist(seqs, 'Jukes-Cantor');
fprintf('pairwise distances (%.0f pairs):\n', length(D));
fprintf('  human-chimp     = %.4f\n', D(1));
fprintf('  human-orangutan = %.4f\n', D(3));

% UPGMA tree.
tr = seqlinkage(D, 'average', names);
fprintf('tree leaves: %.0f\n', tr.NumLeaves);
fprintf('Newick:\n');
disp(getnewickstr(tr));

% Patristic distances along the tree (should track the input distances).
P = pdist(tr);
fprintf('patristic human-chimp = %.4f\n', P(1));

% Neighbor-joining tree of the same data.
trnj = seqneighjoin(D, 'equivar', names);
fprintf('NJ Newick:\n');
disp(getnewickstr(trnj));
