% mod() is a builtin LowerIO doesn't lower; the remaining
% matlab.call_builtin op is unsupported in the C emitter and must
% trigger the fail-fast path.
y = mod(5, 2);
disp(y);
