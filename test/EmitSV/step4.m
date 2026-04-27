% Phase 2 SV — bounded for-loop with constant body.
% `for i = 1:4` with the iv unused; pure scalar accumulator.
T = numerictype(1, 16, 0);
y = step4(fi(0, T));
disp(y);

function y = step4(seed)
    y = seed;
    inc = fi(1, numerictype(1, 16, 0));
    for i = 1:4
        y = y + inc;
    end
end
