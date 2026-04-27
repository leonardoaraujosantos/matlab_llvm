% Phase 2 — for-loop bounds must be compile-time constants.
% Here `n` is a runtime parameter, so the loop trip count is
% data-dependent. The synthesizability gate rejects with a clear
% diagnostic rather than emitting an unbounded RTL construct.
T = numerictype(1, 16, 0);
y = bad(fi(0, T), fi(5, T));
disp(y);

function y = bad(seed, n)
    y = seed;
    inc = fi(1, numerictype(1, 16, 0));
    for i = 1:n
        y = y + inc;
    end
end
