% Function handles with `@name` — take a pointer to a scalar math
% builtin OR to a user-defined function and call it through the
% variable.
%
% Supported builtin handles today (signature f64 -> f64):
%   @sin  @cos  @tan  @exp  @log  @sqrt  @abs
%
% User-function handles (`@mySq; f(3)`) are resolved at compile time
% into direct calls; the handle-through-slot chain is folded away
% before the LowerUserCalls fixpoint refines the callee's signature.

f = @sin;
disp('sin(0) via handle =');
disp(f(0));
disp('sin(pi/2) via handle =');
disp(f(3.14159265358979 / 2));

g = @sqrt;
disp('sqrt(16) =');
disp(g(16));
disp('sqrt(2) =');
disp(g(2));

h = @abs;
disp('abs(-42) =');
disp(h(-42));

k = @exp;
disp('exp(1) =');
disp(k(1));

% User-function handles.
p = @mySq;
disp('mySq(6) via user-handle =');
disp(p(6));

q = @myCube;
disp('myCube(4) =');
disp(q(4));

function y = mySq(x)
    y = x * x;
end

function y = myCube(x)
    y = x * x * x;
end
