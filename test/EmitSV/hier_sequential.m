function [c1, c2] = hier_sequential(en, reset)
    %#codegen
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Hierarchical multi-module with a sequential helper. The top
    % instantiates two counter modules; clk + rst_n are auto-added
    % to the top because at least one callee (counter1) has
    % persistent state, and the instantiation wires both through.

    c1 = counter1(en, reset);
    c2 = counter1(en, reset);
end

function n = counter1(en, reset)
    %#codegen
    persistent n_reg;
    if isempty(n_reg) || reset
        n_reg = uint8(0);
    end
    if en
        n_reg = n_reg + uint8(1);
    end
    n = n_reg;
end
