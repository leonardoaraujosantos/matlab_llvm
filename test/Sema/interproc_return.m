% interproc_return.m — #191 P2.1. A call to a user function defined LATER in
% the file (the usual MATLAB layout) now recovers the callee's return type:
% function bodies are inferred before the script body. Param-independent
% outputs propagate concretely; a param-dependent output (depends on an
% untyped parameter) still degrades to Any.
%
% Also pins the companion fix: bare `true` / `false` are logical scalars (were
% mistyped as @handle), so a function returning one propagates `logical`.

d = makeval();    % double  (return type propagated from a later definition)
b = isready();    % logical (function returns `true`)
a = addone(5);    % any     (output depends on untyped param `x`)
t = true;         % logical (bare constant, not a handle)

function r = makeval()
  r = 3.14;
end
function ok = isready()
  ok = true;
end
function y = addone(x)
  y = x + 1;
end
