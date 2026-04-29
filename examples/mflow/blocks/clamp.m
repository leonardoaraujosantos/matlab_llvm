% clamp(x, lo, hi) — saturate `x` into the closed interval [lo, hi].
% Used as a custom block from examples/mflow/custom_clamp.mflow.
function y = clamp(x, lo, hi)
    if x < lo
        y = lo;
    elseif x > hi
        y = hi;
    else
        y = x;
    end
end
