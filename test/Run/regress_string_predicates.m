% String builtins called with CHAR-LITERAL args ('a' rather than a string var)
% used to hit "unsupported call shape": the literals lower to matlab.const_char
% (tensor), not a string ptr, so the LowerTensorOps arms (which required PtrTy
% operands) never matched. They now materialise via matlab_string_from_literal.
u = upper('abc');
l = lower('ABC');
t = strtrim('  hi  ');
cc = strcat('foo', 'bar');
rr = strrep('aXbXc', 'X', '-');
fprintf('%s %s %s %s %s\n', u, l, t, cc, rr);
a = contains('hello world', 'world');
b = startsWith('hello', 'he');
d = endsWith('hello', 'lo');
e = strcmp('abc', 'abc');
f = contains('abc', 'xyz');
fprintf('%.0f %.0f %.0f %.0f %.0f\n', a, b, d, e, f);
g = str2double('3.14');
h = str2double('-42');
fprintf('%.2f %.0f\n', g, h);
