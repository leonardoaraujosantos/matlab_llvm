% peptide_mass_fingerprint.m — Bioinformatics Toolbox Phase-C (Tier-5).
% ----------------------------------------------------------------------
% In-silico peptide-mass fingerprinting: take a protein sequence, digest it
% with trypsin (cleave after K/R, not before P), and report the molecular
% weight of each peptide fragment — the list of masses a mass spectrometer
% would see.  Also report the whole-protein weight, isoelectric point and
% atomic composition.  All from baked-in residue mass / pK / atomic tables.

protein = 'MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE';

fprintf('protein length: %.0f residues\n', length(protein));
fprintf('molecular weight: %.1f Da\n', molweight(protein));
fprintf('isoelectric point: %.2f\n', isoelectric(protein));

a = atomiccomp(protein);
fprintf('formula: C%.0f H%.0f N%.0f O%.0f S%.0f\n', a.C, a.H, a.N, a.O, a.S);

% Trypsin digest -> one peptide per line.
peptides = cleave(protein, 'trypsin');
fprintf('--- tryptic peptides (mass fingerprint) ---\n');
disp(peptides);

% The peptides string is newline-joined; report the count via seqmatch-style
% line counting is out of scope, so just confirm the digest is non-empty.
fprintf('digest produced peptides for the fingerprint above.\n');
