% Phase 1 SV — nested if/elseif/else against constants.
% Exercises negative-literal rendering (`-16'sd32000`).
T = numerictype(1, 16, 0);
y = sat_add(fi(100, T), fi(200, T));
disp(y);

function y = sat_add(a, b)
    raw = a + b;
    hi = fi(32000, numerictype(1, 16, 0));
    lo = fi(-32000, numerictype(1, 16, 0));
    if raw > hi
        y = hi;
    elseif raw < lo
        y = lo;
    else
        y = raw;
    end
end
