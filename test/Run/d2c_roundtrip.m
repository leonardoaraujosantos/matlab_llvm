% d2c — ZOH discrete->continuous, the inverse of c2d ([A,B]=d2c(Ad,Bd,Ts)).
% Round-trips both a decoupled and a coupled (full, real-eigenvalue) system;
% the full case exercises the matlab_logm 2x2-Schur-block standardization.
A = [-1 0; 0 -2]; B = [1; 1]; Ts = 0.1;
[Ad, Bd] = c2d(A, B, Ts);
[Ar, Br] = d2c(Ad, Bd, Ts);
fprintf('diag A %.4f %.4f\n', Ar(1,1), Ar(2,2));
fprintf('diag B %.4f %.4f\n', Br(1), Br(2));

A2 = [0 1; -2 -3]; B2 = [0; 1];
[Ad2, Bd2] = c2d(A2, B2, Ts);
[Ar2, Br2] = d2c(Ad2, Bd2, Ts);
fprintf('full A %.4f %.4f %.4f %.4f\n', Ar2(1,1), Ar2(1,2), Ar2(2,1), Ar2(2,2));
fprintf('full B %.4f %.4f\n', Br2(1), Br2(2));
