% Phase 2 — induction variable used as a datapath value.
% The loop body consumes `i` (via fi(i, ...)). Phase 2 requires the
% induction variable to be unused inside the body; lowering integer-
% typed iv uses is a Phase 4 enhancement.
T = numerictype(1, 16, 0);
y = bad(fi(0, T));
disp(y);

function y = bad(seed)
    y = seed;
    for i = 1:4
        y = y + fi(i, numerictype(1, 16, 0));
    end
end
