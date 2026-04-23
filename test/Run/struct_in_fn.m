disp(compute(3));

function y = compute(n)
    p.base = n;
    p.sq = n * n;
    p.cube = n * n * n;
    y = p.base + p.sq + p.cube;
end
