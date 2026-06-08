% Bioinformatics Toolbox Tier-1 — sequence conversion + composition.
%   complement / reverse-complement / reverse, DNA<->RNA, codon translation,
%   numeric-lane conversions, base / amino-acid composition counts.
disp(seqcomplement('ATGC'));        % TACG
disp(seqrcomplement('ATGC'));       % GCAT
disp(seqreverse('ATGCAA'));         % AACGTA
disp(dna2rna('ATGC'));              % AUGC
disp(rna2dna('AUGC'));              % ATGC
disp(nt2aa('ATGGCCTGTTAA'));        % MAC*  (ATG=M GCC=A TGT=C TAA=*)
disp(nt2int('ACGT'));               % 1 2 3 4
disp(int2nt([1 2 3 4]));            % ACGT
disp(aa2int('ARNDC'));              % 1 2 3 4 5
b = basecount('AACGTACGT');
fprintf('base A=%.0f C=%.0f G=%.0f T=%.0f\n', b.A, b.C, b.G, b.T);
aa = aacount('ARNDARN');
fprintf('aa A=%.0f R=%.0f N=%.0f D=%.0f\n', aa.A, aa.R, aa.N, aa.D);
fprintf('randseq length: %.0f\n', length(randseq(30)));
