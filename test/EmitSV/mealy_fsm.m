% Phase 4.5 — literal `examples/hdl/mealy_fsm.m` emits clean SV.
%
% This is the actual user-written file from examples/hdl/, with the
% only difference being the typed driver prepended (the existing
% pipeline can't refine func args from a separate caller file
% without multi-file compilation, which works today via the SW1
% multi-file CLI). Exercises:
%
%   - `S0 = uint8(0); S1 = uint8(1)` constant casts (4.5 bonus)
%   - `out_signal = false` (4.5.3)
%   - `if isempty(current_state) || reset` joined-OR initializer
%     split into two `if`s by SplitIsEmptyOr (4.5 bonus2)
%   - `if input_bit == 1` (Phase 4 cmpf state-equality)
%   - persistent state → always_ff register (Phase 3)
T = numerictype(0, 8, 0);
y = mealy_fsm(fi(1, T), fi(0, T));
disp(y);

function out_signal = mealy_fsm(input_bit, reset)
    %#codegen
    S0 = uint8(0);
    S1 = uint8(1);

    persistent current_state;

    if isempty(current_state) || reset
        current_state = S0;
    end

    out_signal = false;

    switch current_state
        case S0
            if input_bit == 1
                current_state = S1;
                out_signal = false;
            else
                current_state = S0;
                out_signal = false;
            end
        case S1
            if input_bit == 1
                current_state = S1;
                out_signal = true;
            else
                current_state = S0;
                out_signal = false;
            end
        otherwise
            current_state = S0;
            out_signal = false;
    end
end
