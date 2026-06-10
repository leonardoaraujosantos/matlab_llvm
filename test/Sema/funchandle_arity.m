% funchandle_arity.m — #191 P4.1. A function handle now carries its arity:
% an anonymous fn from its params (always 1 output), `@userfn` from the
% function's declared inputs/outputs. `@builtin` stays opaque (unknown arity).
% Printed as @handle(in->out); ? marks an unknown side.

f = @(x, y) x + y;   % @handle(2->1)
g = @() 42;          % @handle(0->1)
h = @sin;            % @handle      (builtin — unknown arity)
k = @myfn;           % @handle(3->2)

function [a, b] = myfn(p, q, r)
  a = p;
  b = q + r;
end
