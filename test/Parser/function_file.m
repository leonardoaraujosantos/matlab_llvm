function [y, z] = outer(x)
    y = inner(x) + 1;
    z = x * 2;
end

function r = inner(v)
    r = v * v;
end
