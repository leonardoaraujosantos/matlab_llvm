% Recursion (direct or indirect call-graph cycle) is not synthesizable.
T = numerictype(1, 16, 0);
y = fact(fi(5, T));
disp(y);

function y = fact(n)
    if n <= 1
        y = n;
    else
        y = n * fact(n);
    end
end
