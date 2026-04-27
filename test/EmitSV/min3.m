% Phase 1 SV — three-way minimum via nested if/elseif/else cascade
% over signed comparisons.
T = numerictype(1, 16, 0);
y = min3(fi(7, T), fi(2, T), fi(5, T));
disp(y);

function y = min3(a, b, c)
    y = a;
    if b < y
        y = b;
    end
    if c < y
        y = c;
    end
end
