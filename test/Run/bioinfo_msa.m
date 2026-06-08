% Bioinformatics Toolbox Tier-3 — multiple sequence alignment + profiles.
%   multialign (progressive), seqconsensus, seqprofile, profalign.
seqs = {'MKTAYIAKQR', 'MKTAYIAKNR', 'MKTAYIPKNR'};
aln = multialign(seqs);
disp('--- alignment ---');
disp(aln);
disp('--- consensus ---');
disp(seqconsensus(aln));

P = seqprofile(aln);
fprintf('profile %.0f x %.0f\n', size(P, 1), size(P, 2));
fprintf('col1 Met freq: %.2f\n', P(13, 1));   % column 1 is all M (AA index 13)

% Unequal-length set exercises gap insertion across rows.
ns = multialign({'ACGTACGT', 'ACGACGT', 'ACGTAGT'});
disp('--- nt MSA ---');
disp(ns);

% Profile/pairwise alignment of two sequences.
disp('--- profalign ---');
disp(profalign('ACDEF', 'ACEF'));
