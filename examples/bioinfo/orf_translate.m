% orf_translate.m — Bioinformatics Toolbox Tier-1.
% ----------------------------------------------------------------------
% Take a DNA coding sequence, inspect its base composition, transcribe it
% to mRNA, translate the open reading frame to protein (standard genetic
% code, frame 1), and confirm the reverse-complement round-trips.  Every
% step is a char-lane transform over a baked-in codon/complement table.

dna = 'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG';

b = basecount(dna);
fprintf('length %.0f: A=%.0f C=%.0f G=%.0f T=%.0f\n', ...
        length(dna), b.A, b.C, b.G, b.T);

mrna = dna2rna(dna);
fprintf('mRNA: ');
disp(mrna);

protein = nt2aa(dna);
fprintf('protein (frame 1): ');
disp(protein);                       % M A I V M G R * K G A R *

% Reverse complement and verify the double round-trip is the identity.
rc  = seqrcomplement(dna);
rcc = seqrcomplement(rc);
fprintf('revcomp: ');
disp(rc);
fprintf('double-revcomp restores original: %.0f\n', strcmp(rcc, dna));
