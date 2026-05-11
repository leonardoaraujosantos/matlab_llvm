% Tier-4 close — class-returning model-object short forms.
%
% Sema's `pinnedOfRhs` now propagates the class pin from the first
% arg through known model-object short-form calls (c2d, feedback,
% series, parallel, append, blkdiag), so the LHS slot of
% `sys_d = c2d(sys, Ts)` lands on class `ss` and downstream
% property reads (`sys_d.A`, etc.) route through the class path.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as the existing
% model-object tests.

% --- §3.2 c2d(sys, Ts) — ZOH discretisation. For A = diag(-1, -2),
% Ts = 0.1: Ad = diag(e^{-0.1}, e^{-0.2}); Bd integrates each
% mode independently.
A = [-1 0; 0 -2];
B = [1; 1];
C = [1 0];
D = [0];
sys = ss(A, B, C, D);
sys_d = c2d(sys, 0.1);
disp(sys_d.A);
disp(sys_d.B);
disp(sys_d.C);
disp(sys_d.D);

% --- §3.6 feedback / series / parallel — strictly-proper
% closed-loop assembly returning a fresh ss.
sys1 = ss([-1], [1], [1], 0);
sys2 = ss([-2], [1], [1], 0);

cl = feedback(sys1, sys2);
disp(cl.A);
disp(cl.B);
disp(cl.C);

ser = series(sys1, sys2);
disp(ser.A);
disp(ser.C);

par = parallel(sys1, sys2);
disp(par.A);
disp(par.C);

% --- §5.2 append(sys1, sys2) / blkdiag(sys1, sys2) — block-
% diagonal MIMO assembly. The two-state composite has B and C as
% block-diagonal stacks of the operands' B and C.
ap = append(sys1, sys2);
disp(ap.A);
disp(ap.B);
disp(ap.C);
