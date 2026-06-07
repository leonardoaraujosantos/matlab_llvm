% Bioinformatics Toolbox Tier-1 — FASTA I/O.
%   Read a committed multi-record FASTA fixture into a struct array, read its
%   Header / Sequence fields, and compute on a sequence field (the field read
%   yields the matlab_string* the sequence functions consume).  Then round-trip
%   through fastawrite -> fastaread on a temp file.
s = fastaread('bioinfo_fasta.fa');
fprintf('records: %.0f\n', numel(s));
disp(s(1).Header);
disp(s(1).Sequence);
disp(s(2).Header);

% Compute on a sequence read from the struct array.
b = basecount(s(2).Sequence);
fprintf('insulinB G count: %.0f\n', b.G);
aa = aacount(s(1).Sequence);
fprintf('insulinA Cys count: %.0f\n', aa.C);
selfscore = nwalign(s(1).Sequence, s(1).Sequence, 'BLOSUM62');
fprintf('insulinA self-align > 0: %.0f\n', selfscore > 0);

% fastawrite -> fastaread round-trip (first write truncates the temp file).
fn = '/tmp/bioinfo_rt.fa';
fastawrite(fn, 'one', 'MKTAYIAK');
fastawrite(fn, 'two', 'ACGTACGTACGT');
r = fastaread(fn);
fprintf('roundtrip records: %.0f\n', numel(r));
disp(r(2).Header);
disp(r(2).Sequence);
