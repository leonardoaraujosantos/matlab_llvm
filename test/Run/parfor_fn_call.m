parfor i = 1:3
    work(i);
end

function work(x)
    fprintf('iteration %g\n', x);
end
