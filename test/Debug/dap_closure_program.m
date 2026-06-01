% Regression fixture for issue #77 — a `-dap` launch of a program that
% uses anonymous-function closures the way solver objectives do.
%
% Before the "anon closures use the local-slot lane in JIT/-dap" fix, a
% handle/anon variable was routed through the workspace in ReplMode,
% which severed the make_anon -> addressof -> call_indirect chain, so:
%   - a captured closure `@(x) x + k` couldn't reconstruct its capture, and
%   - an anon passed to another function and called indirectly there
%     (`apply(@(x) x*x, 5)`) resolved as an "indirect call to an undefined
%     name",
% and `matlabc -dap` answered `launch` with "failed to compile program".
%
% Reaching `terminated` is the assertion — every closure shape below must
% lower for the program to compile + run.
k = 3;
f = @(x) x + k;            % captured scalar closure
disp(f(10));               % direct call -> 13
g = @(x) x * x;            % capture-free anon
disp(apply(g, 5));         % anon passed to a fn + called indirectly -> 25
disp(apply(@(x) x + 1, 7)); % anon literal passed inline -> 8

function y = apply(h, v)
    y = h(v);
end
