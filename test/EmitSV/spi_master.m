function [mosi, sclk_out, cs_n, done] = spi_master(tx_byte, start, miso, reset)
    %#codegen
    % hdl: port(tx_byte, fi, unsigned, 8, 0)
    % hdl: port(start, bool)
    % hdl: port(miso, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % SPI master mode 0 (CPOL=0, CPHA=0). Sends one byte MSB-first
    % when `start` pulses; pulses `done` for one cycle on
    % completion. Internally drives:
    %   cs_n   — chip select (active-low; held high in IDLE)
    %   sclk_out — serial clock (toggles in COMPUTE)
    %   mosi   — current MSB of the shift register
    %
    % Tests:
    %   - 4-state FSM (IDLE, SHIFT_HIGH, SHIFT_LOW, DONE)
    %   - shift register update gated by FSM state
    %   - bit-counter advancing per pair of cycles
    %   - multi-output controller (4 output ports + status bit)

    S_IDLE       = uint8(0);
    S_SHIFT_HIGH = uint8(1);   % SCLK rising edge — sample MISO
    S_SHIFT_LOW  = uint8(2);   % SCLK falling edge — shift MOSI
    S_DONE       = uint8(3);

    persistent state;
    persistent shift_reg;
    persistent bit_count;

    if isempty(state) || reset
        state = S_IDLE;
    end
    if isempty(shift_reg) || reset
        shift_reg = uint8(0);
    end
    if isempty(bit_count) || reset
        bit_count = uint8(0);
    end

    sr = shift_reg + uint8(0);
    bc = bit_count + uint8(0);

    % Defaults — overridden in case arms.
    cs_n = true;
    sclk_out = false;
    mosi = false;
    done = false;

    switch state
        case S_IDLE
            cs_n = true;
            sclk_out = false;
            if start
                shift_reg = tx_byte;
                bit_count = uint8(0);
                state = S_SHIFT_HIGH;
            end
        case S_SHIFT_HIGH
            cs_n = false;
            sclk_out = true;
            mosi = bitand(bitshift(sr, -7), uint8(1)) ~= 0;
            % Sample MISO into LSB; shift up the rest on the next
            % falling edge.
            if miso
                shift_reg = bitor(bitand(sr, uint8(254)), uint8(1));
            else
                shift_reg = bitand(sr, uint8(254));
            end
            state = S_SHIFT_LOW;
        case S_SHIFT_LOW
            cs_n = false;
            sclk_out = false;
            mosi = bitand(bitshift(sr, -7), uint8(1)) ~= 0;
            shift_reg = bitand(bitshift(sr, 1), uint8(254));
            if bc == 7
                state = S_DONE;
            else
                bit_count = bc + uint8(1);
                state = S_SHIFT_HIGH;
            end
        case S_DONE
            cs_n = true;
            done = true;
            state = S_IDLE;
        otherwise
            state = S_IDLE;
    end
end
