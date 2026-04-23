% Double-quoted real string literals (distinct from 'char arrays')
s = "hello";
t = "world";
u = s + " " + t;
disp(s);
disp(t);
disp(u);
disp(strlen(u));
disp(isstring(s));
n = 42;
disp(isstring(n));

% Nested concat
greeting = "hi, " + "there" + "!";
disp(greeting);
disp(strlen(greeting));

% Char-array side still behaves exactly as before (single quotes).
c = 'char array';
disp(c);
disp(isstring(c));
