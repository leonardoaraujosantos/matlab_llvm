% fasta_align.m — Bioinformatics Toolbox Tier-1/2.
% ----------------------------------------------------------------------
% Write a small multi-record FASTA file, read it back into a struct array,
% and locally align two of its sequences with Smith-Waterman.  Demonstrates
% the fastawrite -> fastaread round-trip and computing directly on the
% Sequence fields of the returned struct array.

fn = '/tmp/bioinfo_demo.fa';
fastawrite(fn, 'chainA', 'HEAGAWGHEE');
fastawrite(fn, 'chainB', 'PAWHEAE');
fastawrite(fn, 'chainC', 'GAWGHEEKL');

s = fastaread(fn);
fprintf('records read: %.0f\n', numel(s));
disp(s(1).Header);
disp(s(1).Sequence);

% Local alignment of the first two chains (the classic Smith-Waterman pair).
[score, alignment] = swalign(s(1).Sequence, s(2).Sequence, 'BLOSUM62');
fprintf('Smith-Waterman (BLOSUM62) local score: %.1f\n', score);
disp(alignment);

% Global alignment of chains A and C.
gscore = nwalign(s(1).Sequence, s(3).Sequence, 'BLOSUM62');
fprintf('A-vs-C global score: %.1f\n', gscore);
