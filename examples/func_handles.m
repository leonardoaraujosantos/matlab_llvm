% Function handles with `@name` — take a pointer to a runtime scalar
% builtin and call it through the variable. Supported callees today
% are the scalar math functions whose signatures are (f64) -> f64:
%   @sin  @cos  @tan  @exp  @log  @sqrt  @abs
%
% Anonymous functions `@(x) expr` also work as long as they don't
% capture outer variables; see `math_anon_call.m` in test/Run/ for a
% pure-body example.

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
