function rdata = regfile_dyn(waddr, wdata, we, raddr, reset)
    %#codegen
    % hdl: port(waddr, fi, unsigned, 8, 0)
    % hdl: port(wdata, fi, signed, 16, 0)
    % hdl: port(we, bool)
    % hdl: port(raddr, fi, unsigned, 8, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 4-entry register file using runtime-indexed persistent fi-array.
    % Demonstrates the auto-decode path: `regs(addr+1) = wdata` and
    % `rdata = regs(addr+1)` with non-constant addr — Stage F expands
    % each into N decoded write enables (write) or an N-input mux
    % (read), producing the canonical SV regfile pattern with no
    % manual switch/case needed in source.

    persistent regs;
    if isempty(regs) || reset
        regs = fi(zeros(1, 4), 1, 16, 0);
    end

    if we
        regs(waddr + 1) = wdata;
    end

    rdata = regs(raddr + 1);
end
