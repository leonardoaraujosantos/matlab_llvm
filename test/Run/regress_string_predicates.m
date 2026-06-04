% contains/startsWith/endsWith/strcmp with CHAR-LITERAL args used to hit
% "unsupported call shape" — the args arrive as matlab.const_char (tensor),
% not a string ptr, so the lowering arm never matched. They now materialise.
a = contains('hello world', 'world');
b = startsWith('hello', 'he');
c = endsWith('hello', 'lo');
d = contains('abc', 'xyz');
e = strcmp('abc', 'abc');
fprintf('%.0f %.0f %.0f %.0f %.0f\n', a, b, c, d, e);
