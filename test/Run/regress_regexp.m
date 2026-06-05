% Regression for #235: regexp (default start-index form) + regexprep over a
% minimal backtracking regex engine. Covers literals, \d, +, [0-9] classes
% with ranges, (group|alt), greedy quantifiers, {n}, escaped metacharacters,
% anchors, and the $0 (whole-match) replacement token. Numeric results print
% via fprintf %.0f and string results via disp, so output is byte-identical
% across all four execute backends. (The cell-returning 'match'/'tokens'
% forms are out of scope — they need the #233 cell-result work.)
a = regexp('abc123def456', '\d+');
fprintf('%.0f %.0f %.0f\n', numel(a), a(1), a(2));
b = regexp('a1b2c3', '\d');
fprintf('%.0f %.0f\n', numel(b), sum(b));
c = regexp('a.b.c', '\.');
fprintf('%.0f %.0f %.0f\n', numel(c), c(1), c(2));
n = regexp('abc', 'xyz');
fprintf('%.0f\n', numel(n));
disp(regexprep('hello world', 'o', '0'));
disp(regexprep('abc123', '\d+', '#'));
disp(regexprep('a1b2c3', '[0-9]', 'X'));
disp(regexprep('foo bar', '(foo|bar)', 'Z'));
disp(regexprep('aaa', 'a+', 'b'));
disp(regexprep('2024-01-15', '-', '/'));
disp(regexprep('x12y34', '\d{2}', 'N'));
disp(regexprep('cat', '(.)at', '$0!'));
