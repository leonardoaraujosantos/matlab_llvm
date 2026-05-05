function manch_out = manchester_enc(data_in, reset)
    %#codegen
    % hdl: port(data_in, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Manchester encoder (IEEE 802.3 convention):
    %   data 0 → output transitions from low to high (01)
    %   data 1 → output transitions from high to low (10)
    %
    % This implementation runs at 2x the data rate; one data bit
    % takes two clock cycles. A persistent half-bit phase
    % counter alternates between "first half" and "second half"
    % of each data bit.
    %
    % Tests:
    %   - 2-state phase FSM driving a combinational output
    %   - bool XOR (`a ^ b`) — exercises matlab.bxor / arith.xori
    %     i1 path
    %   - canonical encoder pattern (state register + input → output)

    persistent phase;   % 0 = first half, 1 = second half
    if isempty(phase) || reset
        phase = false;
    end

    % Manchester output: first half = data_in XOR'd with 1 (so 0→1
    % and 1→0), second half = data_in directly. Equivalently:
    % first half = NOT data_in; second half = data_in.
    if phase
        manch_out = data_in;
    else
        manch_out = ~data_in;
    end

    phase = ~phase;
end
