% A user function calling `disp` is unsynthesizable — I/O has no
% RTL form. The `@script` driver may still call `disp` on the result.
T = numerictype(1, 16, 0);
y = bad(fi(1, T));
disp(y);

function y = bad(x)
    disp(x);
    y = x + x;
end
