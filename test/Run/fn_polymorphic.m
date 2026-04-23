disp(sq(5));
disp(sq([1 2 3]));
disp(twice([1 2; 3 4]));
disp(twice(7));

function y = sq(x)
    y = x .* x;
end

function y = twice(x)
    y = x + x;
end
