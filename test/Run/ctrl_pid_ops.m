% pid operator overloads — §3.1 sibling math.
%
% v1 algebra: pid + pid / pid - pid / -pid combine coefficient-wise.
% Tf is averaged (it acts as a denominator pole; a sum of two distinct
% Tfs would split into two parallel D-paths and is not representable
% in a single pid struct). True pid-as-tf composition (Laplace
% expansion + tf algebra) is a follow-on.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_model_objects.

C1 = pid(2.0, 0.5, 0.1, 0.01);
C2 = pid(1.0, 0.2, 0.05, 0.01);

% --- plus
S = C1 + C2;
disp(S.Kp);
disp(S.Ki);
disp(S.Kd);
disp(S.Tf);

% --- minus
D = C1 - C2;
disp(D.Kp);
disp(D.Ki);
disp(D.Kd);
disp(D.Tf);

% --- uminus
M = -C1;
disp(M.Kp);
disp(M.Ki);
disp(M.Kd);
disp(M.Tf);
