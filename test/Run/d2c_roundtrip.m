% d2c — ZOH discrete->continuous, the inverse of c2d (explicit-matrix form
% [A, B] = d2c(Ad, Bd, Ts)).  A round-trip on a decoupled system (real,
% distinct eigenvalues) recovers the continuous A, B.
% (Full matrices whose real Schur form keeps a 2x2 block hit a matlab_logm
% limitation — see docs/examples_status_report.md.)
A = [-1 0; 0 -2]; B = [1; 1]; Ts = 0.1;
[Ad, Bd] = c2d(A, B, Ts);
[Ar, Br] = d2c(Ad, Bd, Ts);
fprintf('A %.4f %.4f\n', Ar(1,1), Ar(2,2));
fprintf('B %.4f %.4f\n', Br(1), Br(2));
