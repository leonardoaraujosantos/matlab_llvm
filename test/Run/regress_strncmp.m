% strncmp(a, b, n): true iff the first n characters match; false if either
% string is shorter than n. Was unrecognised ("undefined name 'strncmp'").
fprintf('%.0f %.0f %.0f %.0f\n', ...
  strncmp('abcde','abcxx',3), strncmp('abcde','abcxx',4), ...
  strncmp('foo','foobar',3), strncmp('ab','abc',3));
s = 'hello';
fprintf('var %.0f\n', strncmp(s, 'help', 3));
