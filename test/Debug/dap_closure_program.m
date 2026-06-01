% Regression fixture for issue #77 — a `-dap` launch of a program that
% uses anonymous-function closures.
%
% Before the "anon closures use the local-slot lane in JIT/-dap" fix, a
% handle/anon variable was routed through the workspace in ReplMode,
% which severed the make_anon -> addressof -> call_indirect chain, so a
% captured closure `@(x) x + k` (and the matrix-capturing `@(s) M*s`)
% couldn't reconstruct its capture and `matlabc -dap` answered `launch`
% with "failed to compile program". Reaching `terminated` is the
% assertion — every closure shape below must lower for the program to
% compile + run.
%
% NB: kept to closure shapes that the real `examples/` solver objectives
% exercise (captured scalar/matrix closures + a capture-free anon, all
% direct calls). The higher-order "anon passed to a USER function that
% calls it through a parameter" shape is deliberately NOT here — it trips
% a separate, narrow JIT materialization crash (a function-handle
% parameter indirect call) tracked apart from the closure-lowering fix.
k = 3;
f = @(x) x + k;          % captured scalar closure
disp(f(10));             % -> 13
g = @(x) x * x;          % capture-free anon
disp(g(6));              % -> 36
M = [1 2; 3 4];
scaled = @(s) M * s;     % captured matrix closure (#4)
disp(scaled(2));         % -> [2 4; 6 8]
