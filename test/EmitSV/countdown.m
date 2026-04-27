% Phase 2 SV — descending bounded for-loop.
% Exercises the `for i = init:-step:end` shape and the `i = i - |step|`
% rendering of the SV for-head.
T = numerictype(1, 16, 0);
y = countdown(fi(0, T));
disp(y);

function y = countdown(seed)
    y = seed;
    inc = fi(1, numerictype(1, 16, 0));
    for i = 5:-1:1
        y = y + inc;
    end
end
