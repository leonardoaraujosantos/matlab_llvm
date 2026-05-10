% §3.1 sibling classdefs — `ss` / `zpk` / `pid` / `frd` minimal
% surface (constructor + property storage + property reads).
%
% These four classdefs ship without operator overloads in v1; the
% real CST math (`ss + ss` block-diagonal A; `zpk * zpk` root
% concatenation + gain product; `pid` Laplace expansion + tf
% conversion; `frd` interpolation) is a follow-on. The current
% scope: construct an instance, read its properties back, and
% confirm the storage round-trips. matlabc's per-class prelude
% inclusion (cst_class_<name>.m, conditional on the user code
% mentioning the class name as a call target or assignment LHS)
% means tf-only programs don't pay the unused-classdef cost.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_tf_basic
% (class struct passed by value vs. void* runtime ABI).

% --- ss(A, B, C, D)
A = [-1 0; 0 -2];
B = [1; 1];
C = [1 0];
D = 0;
sys = ss(A, B, C, D);
disp(sys.A);
disp(sys.B);
disp(sys.C);
disp(sys.D);

% --- zpk(z, p, k)
G = zpk([-1; -3], [-2; -4], 5);
disp(G.Z);
disp(G.P);
disp(G.K);

% --- pid(Kp, Ki, Kd, Tf)
Cz = pid(2.5, 0.5, 0.1, 0.01);
disp(Cz.Kp);
disp(Cz.Ki);
disp(Cz.Kd);
disp(Cz.Tf);

% --- frd(response, freqs)
H = frd([1; 0.5; 0.25], [1; 10; 100]);
disp(H.ResponseData);
disp(H.Frequency);
