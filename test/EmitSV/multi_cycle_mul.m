function [product, done] = multi_cycle_mul(a, b, start, reset)
    %#codegen
    % hdl: port(a, fi, unsigned, 16, 0)
    % hdl: port(b, fi, unsigned, 8, 0)
    % hdl: port(start, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 16x8 unsigned shift-and-add multiplier as a state machine.
    % `a` is the 16-bit operand (already wide enough to shift left
    % without overflow); `b` is the 8-bit operand. `start` latches
    % the operands and kicks off an 8-cycle computation; `done`
    % pulses for one cycle when the result is in `product`.
    %
    % Sequential execution — area-efficient alternative to a
    % single-cycle 16x8 multiplier when the result rate is below
    % the clock rate.
    %
    % Tests:
    %   - 3-state FSM (IDLE, COMPUTE, DONE) with branching
    %     transitions
    %   - chained persistent updates under conditional case arms
    %   - integer accumulator across multiple cycles

    S_IDLE    = uint8(0);
    S_COMPUTE = uint8(1);
    S_DONE    = uint8(2);

    persistent state;
    persistent acc;        % running product (16-bit)
    persistent shift_a;    % running shift of A (16-bit, grows left)
    persistent shift_b;    % running shift of B (8-bit, shifted right)
    persistent count;      % cycles remaining

    if isempty(state) || reset
        state = S_IDLE;
    end
    if isempty(acc) || reset
        acc = uint16(0);
    end
    if isempty(shift_a) || reset
        shift_a = uint16(0);
    end
    if isempty(shift_b) || reset
        shift_b = uint8(0);
    end
    if isempty(count) || reset
        count = uint8(0);
    end

    switch state
        case S_IDLE
            if start
                acc = uint16(0);
                shift_a = a;
                shift_b = b;
                count = uint8(8);
                state = S_COMPUTE;
            end
        case S_COMPUTE
            % Snapshot persistents into typed locals first so the
            % downstream ops see concrete integer types instead of
            % the runtime f64 ABI from matlab_global_get_f64.
            sb = shift_b + uint8(0);
            sa = shift_a + uint16(0);
            ac = acc + uint16(0);
            cn = count + uint8(0);
            % If LSB of sb is 1, accumulate sa into acc.
            if bitand(sb, uint8(1)) ~= 0
                acc = ac + sa;
            end
            shift_a = bitshift(sa, 1);
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
