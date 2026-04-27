% Phase 3 — persistent variable without an `if isempty(...) ... end`
% initializer. The synthesizability gate rejects: a register has no
% well-defined reset value to drive into the always_ff's reset branch.
T = numerictype(0, 8, 0);
y = bad(fi(1, T));
disp(y);

function out = bad(d)
    persistent r;
    z = fi(0, numerictype(0, 8, 0));
    if d > z
        r = d;
    end
    out = r;
end
