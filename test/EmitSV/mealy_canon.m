% Phase 4.5 SV — canonical Mealy FSM exercising several pipeline-
% hardening items together:
%
%   - `uint8(N)` constant casts (this commit's bonus item) fold to
%     `arith.constant N : i8` instead of routing through the
%     `matlab_uint8_s` runtime call. Lets the user write the
%     idiomatic state-constant declarations `S0 = uint8(0); S1 =
%     uint8(1)`.
%   - `out_signal = false` (Phase 4.5.3) emits as `output logic y`
%     with `1'b0` reset.
%   - `if reset` (Phase 4.5.2) lowers to `reset != 8'sd0` via the
%     RefineIfConds fixup.
%   - The `switch state` cascade routes to nested `if (state ==
%     <const>)` checks that the SV emitter renders inside
%     always_comb driving `state_next`.
%
% The HDL Coder canonical idiom keeps the `if isempty(...)`
% initializer as a separate statement from the explicit `if reset`
% — joining them with `||` makes the isempty result have multiple
% uses, which the HWStateInfer matcher rejects (an isempty's only
% legal consumer is a single cmpf feeding an scf.if guard).
T = numerictype(0, 8, 0);
y = mealy_canon(fi(1, T), fi(0, T));
disp(y);

function out_signal = mealy_canon(input_bit, reset)
    %#codegen
    S0 = uint8(0);
    S1 = uint8(1);
    persistent current_state;
    if isempty(current_state)
        current_state = S0;
    end
    if reset
        current_state = S0;
    end
    out_signal = false;
    switch current_state
        case S0
            if input_bit == uint8(1)
                current_state = S1;
            end
        case S1
            if input_bit == uint8(1)
                out_signal = true;
            else
                current_state = S0;
            end
        otherwise
            current_state = S0;
    end
end
