% icare / idare — numerically-robust Riccati aliases (Tier-1.5
% follow-on of CST roadmap §2.5). v1 routes through the existing
% matlab_sign-Newton care / Newton-Kleinman dare; the structure-
% preserving Mehrmann-Voss QZ path is the proper follow-on.

% --- 1. Continuous: icare on the double-integrator should match the
% canonical care answer X = [√3 1; 1 √3].
A = [0 1; 0 0];
B = [0; 1];
Q = eye(2);
R = 1;
Xc = icare(A, B, Q, R);
Xc_ref = care(A, B, Q, R);
disp(round((Xc(1,1) - Xc_ref(1,1))*1e10)/1e10);   % 0
disp(round((Xc(1,2) - Xc_ref(1,2))*1e10)/1e10);   % 0
disp(round((Xc(2,2) - Xc_ref(2,2))*1e10)/1e10);   % 0
% Closed-form spot-check.
disp(round((Xc(2,2) - sqrt(3))*1e10)/1e10);       % 0

% --- 2. Discrete: idare on a Schur-stable diagonal plant — same X
% as dare.
Ad = [0.5 0; 0 0.7];
Bd = eye(2);
Qd = eye(2);
Rd = eye(2);
Xd = idare(Ad, Bd, Qd, Rd);
Xd_ref = dare(Ad, Bd, Qd, Rd);
disp(round((Xd(1,1) - Xd_ref(1,1))*1e10)/1e10);   % 0
disp(round((Xd(2,2) - Xd_ref(2,2))*1e10)/1e10);   % 0
% Print the diagonals — both must be positive (Schur-stable plant
% has a unique stabilising X with X = X' ≻ 0).
disp(Xd(1,1));
disp(Xd(2,2));
