% Workstream C SV — saturation result consumed only by an
% anonymous comparison and a return slot. Verifies the C1
% naming pass falls back gracefully (structural prefix like
% `sat_in_<N>` / `sat_out_<N>`) when there is no named slot or
% return port to derive a base from.
function ok = sat_unnamed_consumer(a, b)
    %#codegen
    % hdl: port(a, fi, signed, 16, 12)
    % hdl: port(b, fi, signed, 16, 12)
    s = fi(a, 1, 16, 12) * fi(b, 1, 16, 12);
    s_clip = fi(s, 1, 16, 12, 'OverflowAction', 'Saturate');
    % Compare the saturated result against zero — the comparison
    % itself is anonymous (its result is the unnamed boolean
    % we return), so the saturation output has no obvious
    % destination name to inherit.
    ok = s_clip > fi(0, 1, 16, 12);
end
