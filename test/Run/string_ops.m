% String / formatting builtins.

s = "Hello, World";
disp(upper(s));              % HELLO, WORLD
disp(lower(s));              % hello, world

disp(startsWith(s, "Hello"));% 1
disp(startsWith(s, "hello"));% 0
disp(endsWith(s, "World"));  % 1
disp(endsWith(s, "!"));      % 0
disp(contains(s, "lo, W"));  % 1
disp(contains(s, "xyz"));    % 0

disp(strtrim("   padded   "));  % padded

disp(strrep("foo bar foo", "foo", "baz"));  % baz bar baz

disp(strcat("foo", "bar"));  % foobar

fmt = "x = %.3f\n";
disp(sprintf(fmt, 3.14159));% "x = 3.142\n" (trailing newline shown)

disp(num2str(42.5));         % 42.5

disp(str2double("3.25"));    % 3.25
