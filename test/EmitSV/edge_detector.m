function [rise, fall] = edge_detector(sig, reset)
    %#codegen
    % hdl: port(sig, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Synchronous rising/falling edge detector with a 1-cycle delay
    % register. `sig_d` holds the previous-cycle sample; `rise` is
    % a one-cycle pulse on a 0→1 transition, `fall` is the pulse
    % on a 1→0 transition.
    %
    % Tests:
    %   - persistent register read AND combinational comparison
    %     against the just-read value
    %   - one persistent updated, two outputs both derived from
    %     the register's current vs new state

    persistent sig_d;
    if isempty(sig_d) || reset
        sig_d = false;
    end

    rise = sig && ~sig_d;
    fall = ~sig && sig_d;

    sig_d = sig;
end
