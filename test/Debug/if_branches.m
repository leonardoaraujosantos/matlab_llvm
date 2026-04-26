% if/else fixture: the IfStmt itself emits a hook on the keyword line,
% and each branch's body statements emit their own hooks. There is no
% separate hook on the `else` keyword or the trailing `end`.
x = 5;

if x > 0
    a = 1;
else
    a = 2;
end

b = a * 2;
