% Workstream C SV — saturation chain whose final value is stored
% into a named MATLAB local. Post-C1 the unsaturated and clipped
% intermediates carry context-derived names (`acc_pre`, `acc_sat`)
% derived from the destination slot, replacing the anonymous
% `vN_1` placeholders. Lint-clean under verilator -Wall.
function r = sat_named_chain(a, b)
    %#codegen
    % hdl: port(a, fi, signed, 16, 12)
    % hdl: port(b, fi, signed, 16, 12)
    acc = fi(a, 1, 16, 12) * fi(b, 1, 16, 12);
    % First clamp to a wide intermediate, then to the narrow output —
    % a two-stage saturation cascade like fir_asic_pipelined's MAC.
    acc1 = fi(acc, 1, 24, 18, 'OverflowAction', 'Saturate');
    r = fi(acc1, 1, 16, 12, 'OverflowAction', 'Saturate');
end
