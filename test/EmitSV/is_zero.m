% Phase 4.5.3 SV — `true` / `false` lower to `arith.constant 1|0 : i1`
% (instead of `matlab.make_handle{callee="true"|"false"}`), so a
% function that writes a bool output port emits cleanly with an i1
% port type and `1'b1` / `1'b0` literals.
T = numerictype(0, 8, 0);
y = is_zero(fi(0, T));
disp(y);

function r = is_zero(v)
    r = false;
    z = fi(0, numerictype(0, 8, 0));
    if v == z
        r = true;
    end
end
