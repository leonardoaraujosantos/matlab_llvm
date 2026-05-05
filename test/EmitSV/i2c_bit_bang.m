function [scl, sda, busy] = i2c_bit_bang(tx_byte, start, reset)
    %#codegen
    % hdl: port(tx_byte, fi, unsigned, 8, 0)
    % hdl: port(start, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Minimal I2C master bit-banger. On `start`, drives a START
    % condition (SDA high→low while SCL high), shifts out 8 data
    % bits MSB-first (one full SCL period per bit), then issues a
    % STOP condition (SDA low→high while SCL high) and returns to
    % IDLE. `busy` is high while the FSM is shifting.
    %
    % This is a quad-phase per-bit bit-banger — each data bit
    % advances through SETUP_LOW → CLK_HIGH → HOLD_HIGH → CLK_LOW
    % (so SCL stays high for half the bit time, low for half).
    % With one phase increment per call, a full byte takes 32
    % cycles plus the START/STOP framing.
    %
    % Tests:
    %   - 6-state FSM (IDLE, START, BIT, ACK, STOP, DONE) plus
    %     2-bit phase sub-counter inside BIT
    %   - shift register MSB-extract over 8 iterations
    %   - mux of (SCL, SDA) driven from FSM state + phase
    %   - canonical I2C-style multi-output controller pattern

    S_IDLE  = uint8(0);
    S_START = uint8(1);
    S_BIT   = uint8(2);
    S_ACK   = uint8(3);
    S_STOP  = uint8(4);
    S_DONE  = uint8(5);

    persistent state;
    persistent phase;     % 0..3 within a BIT slot
    persistent bit_idx;   % 0..7 across the byte
    persistent shift_reg;

    if isempty(state) || reset
        state = S_IDLE;
    end
    if isempty(phase) || reset
        phase = uint8(0);
    end
    if isempty(bit_idx) || reset
        bit_idx = uint8(0);
    end
    if isempty(shift_reg) || reset
        shift_reg = uint8(0);
    end

    sr = shift_reg + uint8(0);
    ph = phase + uint8(0);
    bi = bit_idx + uint8(0);

    % Defaults: bus released (open-drain idle), not busy.
    scl = true;
    sda = true;
    busy = false;

    switch state
        case S_IDLE
            scl = true;
            sda = true;
            if start
                shift_reg = tx_byte;
                bit_idx = uint8(0);
                phase = uint8(0);
                state = S_START;
            end
        case S_START
            % START: SDA falls while SCL is high.
            scl = true;
            sda = false;
            busy = true;
            state = S_BIT;
            phase = uint8(0);
        case S_BIT
            busy = true;
            % Current MSB on SDA throughout the bit slot.
            sda = bitand(bitshift(sr, -7), uint8(1)) ~= 0;
            if ph == 0
                scl = false;       % setup: SCL low
                phase = uint8(1);
            elseif ph == 1
                scl = true;        % rising edge: slave samples
                phase = uint8(2);
            elseif ph == 2
                scl = true;        % hold high
                phase = uint8(3);
            else
                scl = false;       % falling edge: shift on next cycle
                shift_reg = bitand(bitshift(sr, 1), uint8(254));
                if bi == 7
                    state = S_ACK;
                    phase = uint8(0);
                else
                    bit_idx = bi + uint8(1);
                    phase = uint8(0);
                end
            end
        case S_ACK
            % Release SDA, pulse SCL to read ACK (we ignore the
            % returned bit here — a real master would latch it).
            sda = true;
            busy = true;
            if ph == 0
                scl = true;
                phase = uint8(1);
            else
                scl = false;
                state = S_STOP;
                phase = uint8(0);
            end
        case S_STOP
            % STOP: SDA rises while SCL is high.
            busy = true;
            if ph == 0
                scl = true;
                sda = false;
                phase = uint8(1);
            else
                scl = true;
                sda = true;
                state = S_DONE;
            end
        case S_DONE
            scl = true;
            sda = true;
            state = S_IDLE;
        otherwise
            state = S_IDLE;
    end
end
