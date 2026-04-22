function y = f(x, n)
    if x > 0
        y = x;
    elseif x < 0
        y = -x;
    else
        y = 0;
    end
    for i = 1:n
        y = y + i;
    end
    while y > 100
        y = y / 2;
    end
    switch mod(y, 3)
        case 0
            y = y + 1;
        case {1, 2}
            y = y - 1;
        otherwise
            y = 0;
    end
    try
        y = sqrt(y);
    catch err
        y = -1;
    end
end
