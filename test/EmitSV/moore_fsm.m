% Phase 4.5 — literal `examples/hdl/moore_fsm.m` emits clean SV.
%
% Same structure as mealy_fsm.m but Moore-style: the output
% (`out_signal`) is decoded purely from the state register, after
% the `switch state` has settled to its next value. Two outputs:
% the boolean detector and the integer state-display.
T = numerictype(0, 8, 0);
[d, s] = moore_fsm(fi(1, T), fi(0, T));
disp(d);

function [out_signal, state_display] = moore_fsm(input_bit, reset)
    %#codegen
    S0 = uint8(0);
    S1 = uint8(1);
    S2 = uint8(2);

    persistent current_state;

    if isempty(current_state) || reset
        current_state = S0;
    end

    switch current_state
        case S0
            if input_bit == 1
                current_state = S1;
            end
        case S1
            if input_bit == 0
                current_state = S2;
            else
                current_state = S0;
            end
        case S2
            if input_bit == 1
                current_state = S1;
            else
                current_state = S0;
            end
        otherwise
            current_state = S0;
    end

    if current_state == S2
        out_signal = true;
    else
        out_signal = false;
    end
    state_display = current_state;
end
