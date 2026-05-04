function [data, valid] = uart_rx(rx, reset)
    %#codegen
    % hdl: port(rx, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Wider FSM stress test — 11 states for a UART receiver. Real UART
    % designs use baud-tick gating; this stripped-down version assumes
    % `rx` is already sampled at the bit clock so the state machine
    % advances one bit per cycle. Exercises:
    %   - non-sequential transitions (BIT7 -> STOP -> IDLE)
    %   - >2 state-bit width (11 states need 4 bits for binary
    %     encoding) — tests the FSM enum-width math
    %   - state-conditioned output writes (data shift-in only on BIT0..7,
    %     valid pulse only on STOP)

    S_IDLE  = uint8(0);
    S_START = uint8(1);
    S_BIT0  = uint8(2);
    S_BIT1  = uint8(3);
    S_BIT2  = uint8(4);
    S_BIT3  = uint8(5);
    S_BIT4  = uint8(6);
    S_BIT5  = uint8(7);
    S_BIT6  = uint8(8);
    S_BIT7  = uint8(9);
    S_STOP  = uint8(10);

    persistent state;
    persistent shift_reg;
    persistent data_reg;
    persistent valid_reg;

    if isempty(state) || reset
        state = S_IDLE;
    end
    if isempty(shift_reg) || reset
        shift_reg = uint8(0);
    end
    if isempty(data_reg) || reset
        data_reg = uint8(0);
    end
    if isempty(valid_reg) || reset
        valid_reg = false;
    end

    % Default: valid pulses high only on STOP.
    valid_reg = false;

    switch state
        case S_IDLE
            if rx == 0  % start bit detected (rx low)
                state = S_START;
            end
        case S_START
            shift_reg = uint8(0);
            state = S_BIT0;
        case S_BIT0
            if rx
                shift_reg = shift_reg + uint8(1);
            end
            state = S_BIT1;
        case S_BIT1
            if rx
                shift_reg = shift_reg + uint8(2);
            end
            state = S_BIT2;
        case S_BIT2
            if rx
                shift_reg = shift_reg + uint8(4);
            end
            state = S_BIT3;
        case S_BIT3
            if rx
                shift_reg = shift_reg + uint8(8);
            end
            state = S_BIT4;
        case S_BIT4
            if rx
                shift_reg = shift_reg + uint8(16);
            end
            state = S_BIT5;
        case S_BIT5
            if rx
                shift_reg = shift_reg + uint8(32);
            end
            state = S_BIT6;
        case S_BIT6
            if rx
                shift_reg = shift_reg + uint8(64);
            end
            state = S_BIT7;
        case S_BIT7
            if rx
                shift_reg = shift_reg + uint8(128);
            end
            state = S_STOP;
        case S_STOP
            data_reg = shift_reg;
            valid_reg = true;
            state = S_IDLE;
        otherwise
            state = S_IDLE;
    end

    data = data_reg;
    valid = valid_reg;
end
