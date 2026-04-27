% Phase 3 SV — synchronous up-counter with sync clear + enable.
% This is the canonical HDL Coder counter idiom: a single persistent
% variable initialized via `if isempty(...)`, conditionally updated
% by a runtime-input enable / reset pair. The SV emitter renders the
% persistent as an `always_ff`-driven register with the
% async-low (default ASIC) reset convention.
T = numerictype(0, 8, 0);
en = fi(1, T);
rst = fi(0, T);
y = counter(en, rst);
disp(y);

function count = counter(en, rst)
    persistent c;
    if isempty(c)
        c = fi(0, numerictype(0, 8, 0));
    end
    z = fi(0, numerictype(0, 8, 0));
    if rst > z
        c = fi(0, numerictype(0, 8, 0));
    elseif en > z
        c = c + fi(1, numerictype(0, 8, 0));
    end
    count = c;
end
