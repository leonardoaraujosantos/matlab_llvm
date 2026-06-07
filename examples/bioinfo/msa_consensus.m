% msa_consensus.m — Bioinformatics Toolbox Phase-B (Tier-3).
% ----------------------------------------------------------------------
% Multiple-sequence-align a small protein family with multialign, then
% summarise the alignment with a consensus sequence (seqconsensus) and a
% position-specific residue-frequency profile (seqprofile).  Progressive
% alignment over the shipped Needleman-Wunsch core; no external dependency.

family = { ...
    'MKTAYIAKQRQISFVK', ...
    'MKTAYIAKNRQISFVK', ...
    'MKTAYIPKNRQLSFVK', ...
    'MKSAYIAKNRQISFAK'};

aln = multialign(family);
fprintf('aligned family (%.0f sequences):\n', length(family));
disp(aln);

cons = seqconsensus(aln);
fprintf('consensus: ');
disp(cons);

P = seqprofile(aln);
fprintf('profile is %.0f residues x %.0f columns\n', size(P, 1), size(P, 2));

% Fully conserved columns: a profile column with a single residue at
% frequency 1.0.  Count them.
nconserved = 0;
for j = 1:size(P, 2)
    if max(P(:, j)) >= 1.0
        nconserved = nconserved + 1;
    end
end
fprintf('fully conserved columns: %.0f / %.0f\n', nconserved, size(P, 2));
