% align_globin_pair.m — Bioinformatics Toolbox Phase-A headline.
% ----------------------------------------------------------------------
% Globally align the N-terminal fragments of the human and mouse
% hemoglobin beta chains with the Needleman-Wunsch algorithm under the
% BLOSUM62 substitution matrix, report the alignment score and percent
% identity, and summarise the seqdotplot self-similarity map.  This is the
% canonical sequence-alignment workflow: a scoring-matrix lookup + dynamic
% programming fill + traceback, all hand-coded over the char/string lane.
%
% nwalign / swalign / blosum / seqdotplot are baked-in — no external
% dependency (no BioPerl, no NCBI).

% Human HBB and mouse Hbb-b1, residues 1-50 (the conserved globin fold).
human = 'MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLS';
mouse = 'MVHLTDAEKAAVSCLWGKVNSDEVGGEALGRLLVVYPWTQRYFDSFGDLS';

[score, alignment] = nwalign(human, mouse, 'BLOSUM62');
fprintf('Needleman-Wunsch (BLOSUM62) score: %.1f\n', score);
fprintf('Alignment:\n');
disp(alignment);

% Percent identity from the seqdotplot diagonal (these equal-length, gap-free
% globin fragments align position-for-position, so the diagonal counts the
% identical columns).
D = seqdotplot(human, mouse);
nident = sum(diag(D));
fprintf('identical positions: %.0f / %.0f\n', nident, length(human));
fprintf('percent identity: %.1f%%\n', 100 * nident / length(human));

% Amino-acid composition of the human fragment.
aa = aacount(human);
fprintf('human Leu(L)=%.0f Val(V)=%.0f Glu(E)=%.0f\n', aa.L, aa.V, aa.E);
