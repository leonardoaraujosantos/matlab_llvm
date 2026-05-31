% #79.2: char-string assignment to a struct field + read via disp & fprintf %s.
s.name = 'hello';
s.count = 42;
y = s.name;
disp(y);
fprintf('%s %g\n', s.name, s.count);
