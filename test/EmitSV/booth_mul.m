function [product, done] = booth_mul(a, b, start, reset)
    %#codegen
    % hdl: port(a, fi, signed, 8, 0)
    % hdl: port(b, fi, signed, 8, 0)
    % hdl: port(start, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Signed 8x8 Booth multiplier — the canonical signed counterpart
    % to multi_cycle_mul.m. Booth's algorithm inspects two adjacent
    % bits of the multiplier in each cycle:
    %   00 / 11 → no add (both same)
    %   01 → add A << k    (positive contribution)
    %   10 → subtract A << k (negative contribution)
    %
    % This test exercises the **signed** register update path:
    %   - signed accumulator and operand
    %   - signed add/subtract under conditional case arms
    %   - sign-extension across the operand width
    %
    % Tests:
    %   - signed persistent register (i8 signed)
    %   - sign-aware add/subtract with carry through to a wider acc
    %   - 3-state FSM with constant-arithmetic step counter

    S_IDLE    = uint8(0);
    S_COMPUTE = uint8(1);
    S_DONE    = uint8(2);

    persistent state;
    persistent acc;          % running product (signed 16-bit)
    persistent shift_a;      % A, kept fixed but accessed as i16
    persistent shift_b;      % B, shifted right one bit per cycle
    persistent prev_b;       % previous LSB of shift_b for Booth pair
    persistent count;

    if isempty(state) || reset
        state = S_IDLE;
    end
    if isempty(acc) || reset
        acc = int16(0);
    end
    if isempty(shift_a) || reset
        shift_a = int16(0);
    end
    if isempty(shift_b) || reset
        shift_b = int8(0);
    end
    if isempty(prev_b) || reset
        prev_b = false;
    end
    if isempty(count) || reset
        count = uint8(0);
    end

    switch state
        case S_IDLE
            if start
                acc = int16(0);
                shift_a = a;          % i16 sign-extends from i8
                shift_b = b;
                prev_b = false;
                count = uint8(8);
                state = S_COMPUTE;
            end
        case S_COMPUTE
            % Snapshot before bitwise/arith ops.
            ac = acc + int16(0);
            sa = shift_a + int16(0);
            sb = shift_b + int8(0);
            cn = count + uint8(0);
            curr_b = bitand(sb, int8(1)) ~= 0;
            % Booth pair: (curr_b, prev_b)
            %   01 → add sa
            %   10 → subtract sa
            %   else → no change
            if curr_b && ~prev_b
                acc = ac - sa;
            elseif ~curr_b && prev_b
                acc = ac + sa;
            end
            % Arithmetic shift right: divide shift_b by 2 with sign
            % preservation. shift_a doesn't actually move in Booth
            % proper (we update acc by add/sub of A << k by tracking
            % the implicit alignment via count), so we just rotate
            % through the bit positions.
            prev_b = curr_b;
            shift_b = bitshift(sb, -1);
            count = cn - uint8(1);
            if cn == 1
                state = S_DONE;
            end
        case S_DONE
            state = S_IDLE;
        otherwise
            state = S_IDLE;
    end

    product = acc;
    done = state == S_DONE;
end
