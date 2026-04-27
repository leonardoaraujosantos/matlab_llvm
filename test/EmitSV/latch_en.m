% Phase 3 SV — simple enable-gated register (data latch with clock).
% No counter logic — just `r := d` when `en` is asserted, `r` holds
% otherwise. Confirms the "hold by default" pattern (`r_next = r;` at
% the top of always_comb) protects against latch inference even when
% the user's body has only one conditional update.
T = numerictype(0, 8, 0);
y = latch_en(fi(42, T), fi(1, T));
disp(y);

function out = latch_en(d, en)
    persistent r;
    if isempty(r)
        r = fi(0, numerictype(0, 8, 0));
    end
    z = fi(0, numerictype(0, 8, 0));
    if en > z
        r = d;
    end
    out = r;
end
