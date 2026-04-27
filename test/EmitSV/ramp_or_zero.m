% Phase 2 SV — bounded for-loop nested inside a scf.if branch.
% Exercises the `if (cond) begin ... for(...) ... end` shape and
% confirms the latch-guard prelude still covers signals written
% conditionally inside the loop body.
T = numerictype(1, 16, 0);
y = ramp_or_zero(fi(0, T), fi(1, numerictype(0, 1, 0)));
disp(y);

function y = ramp_or_zero(seed, en)
    y = seed;
    if en > fi(0, numerictype(0, 1, 0))
        inc = fi(1, numerictype(1, 16, 0));
        for i = 1:3
            y = y + inc;
        end
    end
end
