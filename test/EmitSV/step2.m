% Phase 2 SV — bounded for-loop with non-unit step.
% `for i = 1:2:9` exercises the `i = i + step` rendering.
T = numerictype(1, 16, 0);
y = step2(fi(0, T));
disp(y);

function y = step2(seed)
    y = seed;
    inc = fi(2, numerictype(1, 16, 0));
    for i = 1:2:9
        y = y + inc;
    end
end
