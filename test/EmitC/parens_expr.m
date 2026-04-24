% Exercises paren-simplification candidates: nested arithmetic, loads,
% returns, comparisons. Locks the expected paren density in output.
disp(compute(2, 3));
disp(compute(4, 5));

function y = compute(a, b)
    y = (a + b) * (a - b);
end
