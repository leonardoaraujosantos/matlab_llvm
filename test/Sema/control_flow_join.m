function y = demo(flag)
    if flag
        y = 1.0;
    else
        y = [1 2 3];
    end
    % y's type here is the join: double with unknown shape
end
