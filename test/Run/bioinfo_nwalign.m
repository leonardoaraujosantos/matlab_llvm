% Bioinformatics Toolbox Tier-2 — pairwise alignment + scoring matrices.
%   nwalign (Needleman-Wunsch global), swalign (Smith-Waterman local),
%   the BLOSUM62 scoring-matrix catalogue, and seqdotplot.
[sc, al] = nwalign('MKTAYIAKQR', 'MKTAYIAKNR', 'BLOSUM62');
fprintf('global score: %.1f\n', sc);   % 44 (sum of BLOSUM62 diagonal, Q/N=0)
disp(al);

[ls, la] = swalign('HEAGAWGHEE', 'PAWHEAE', 'BLOSUM62');
fprintf('local score: %.1f\n', ls);
disp(la);

% Nucleotide alignment uses NUC44 by default (match +5 / mismatch -4).
ns = nwalign('ACGTACGT', 'ACGTTCGT');
fprintf('nt score: %.1f\n', ns);       % 7 identical * 5 + 1 mismatch * -4

B = blosum(62);
fprintf('blosum size: %.0f x %.0f\n', size(B, 1), size(B, 2));  % 24 x 24
fprintf('B(A,A)=%.0f B(W,W)=%.0f B(C,C)=%.0f\n', B(1,1), B(18,18), B(5,5));

d = seqdotplot('ACGTACGT', 'ACGTACGT');
fprintf('dotplot diag sum: %.0f\n', sum(diag(d)));   % 8 (self-match diagonal)
