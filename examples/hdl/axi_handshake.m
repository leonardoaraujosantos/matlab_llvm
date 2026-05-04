function [tdata, tvalid, sready] = axi_handshake(in_data, in_valid, m_ready, reset)
    %#codegen
    % hdl: port(in_data, fi, signed, 16, 0)
    % hdl: port(in_valid, bool)
    % hdl: port(m_ready, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Single-stage AXI-Stream-style register slice: latches
    % (in_data, in_valid) when the master interface is ready
    % (m_ready) and there's a valid input. Drives slave-side
    % `sready` based on whether the buffer is empty or being
    % drained this cycle.
    %
    % This is the canonical AXI register-slice pattern used as
    % a pipeline buffer in nearly every IP block. Tests:
    %   - dual persistent (data + valid) updated under nested
    %     conditional handshake logic
    %   - bool output derived from comparison + persistent

    persistent buf_data;
    persistent buf_valid;

    if isempty(buf_data) || reset
        buf_data = fi(0, 1, 16, 0);
    end
    if isempty(buf_valid) || reset
        buf_valid = false;
    end

    % Slave-side ready: we can accept new data when the buffer is
    % empty or about to drain.
    sready = ~buf_valid || m_ready;

    % Buffer update: shift in if we're accepting; clear if the
    % master accepted and no new sample is coming in.
    if sready && in_valid
        buf_data = in_data;
        buf_valid = true;
    elseif buf_valid && m_ready
        buf_valid = false;
    end

    tdata = buf_data;
    tvalid = buf_valid;
end
