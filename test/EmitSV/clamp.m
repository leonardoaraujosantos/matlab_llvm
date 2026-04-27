% Phase 1 SV — multi-store scalar slot in `always_comb`.
% Two sequential `if` branches both write `y`. Verilator must not
% infer a latch (every path through the function assigns y).
T = numerictype(1, 16, 0);
y = clamp(fi(20, T), fi(-5, T), fi(10, T));
disp(y);

function y = clamp(x, lo, hi)
    y = x;
    if x < lo
        y = lo;
    end
    if x > hi
        y = hi;
    end
end
