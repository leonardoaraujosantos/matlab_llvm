total = 0;
parfor i = 1:5
    total = total + sq(i);
end
disp(total);

function y = sq(x)
    y = x * x;
end
