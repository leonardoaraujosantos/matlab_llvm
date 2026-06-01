% Regression fixture for issue #77 — a `-dap` launch of a program that
% uses a toolbox classdef (here `dlarray` + its operator/method overloads
% from dlnet_classdefs.m).
%
% Before the prelude-parse + dead-strip fix, the JIT/-dap path:
%   1. parsed the merged toolbox classdef prelude as a standalone buffer,
%      which errored "stray tokens after classdef" at the 2nd classdef and
%      silently dropped the ENTIRE prelude, then
%   2. left every (now-undispatched) classdef method as an unlowered
%      builtin / kept uncalled methods whose internal runtime calls never
%      lowered,
% so `matlabc -dap` answered `launch` with "failed to compile program".
%
% This program exercises method dispatch (`relu`), operator overloads
% (`*`, `+` on dlarray), and a value round-trip (`extractdata`). If the
% classdef prelude is dropped, none of these dispatch and the launch
% fails to compile — so simply reaching `terminated` is the assertion.
W = dlarray([1 0; 0 1]);
x = dlarray([2; 3]);
b = dlarray([1; 1]);
y = relu(W * x + b);
v = extractdata(y);
fprintf('y = %g %g\n', v(1), v(2));
