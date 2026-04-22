function r = demo(x, y)
    a = x > y;              % logical
    b = x == y;             % logical
    c = ~a;                 % logical
    d = a && b;             % logical
    e = a & b;              % logical (vector if a,b are arrays)
    r = a;
end
