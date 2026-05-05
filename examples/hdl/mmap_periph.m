function [rdata, ack] = mmap_periph(addr, wdata, we, re, reset)
    %#codegen
    % hdl: port(addr, fi, unsigned, 8, 0)
    % hdl: port(wdata, fi, unsigned, 16, 0)
    % hdl: port(we, bool)
    % hdl: port(re, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Memory-mapped peripheral with 4 16-bit registers. Write port
    % decodes `addr` and stores `wdata` into the matching register
    % when `we` is asserted. Read port returns the addressed
    % register's value (or r0 on miss). `ack` pulses for one cycle
    % on any successful read or write.
    %
    % Tests:
    %   - case-decoded write port across 4 persistent registers
    %   - read mux pattern (post B-workstream slot type-unification)
    %   - 1-cycle ack pulse derived from request inputs
    %
    % This is the canonical "register file accessed as memory" shape
    % used in many small peripheral designs (UART control regs, GPIO
    % direction/output, simple status registers).

    persistent r0; persistent r1; persistent r2; persistent r3;

    if isempty(r0) || reset; r0 = uint16(0); end
    if isempty(r1) || reset; r1 = uint16(0); end
    if isempty(r2) || reset; r2 = uint16(0); end
    if isempty(r3) || reset; r3 = uint16(0); end

    if we
        switch addr
            case 0; r0 = wdata;
            case 1; r1 = wdata;
            case 2; r2 = wdata;
            case 3; r3 = wdata;
            otherwise; r0 = r0;
        end
    end

    % Read mux. Default arm reads r0 so every read-port store has
    % a matching type — the multi-source slot type-unification path
    % requires all stores from the same width.
    switch addr
        case 0; rdata = r0;
        case 1; rdata = r1;
        case 2; rdata = r2;
        case 3; rdata = r3;
        otherwise; rdata = r0;
    end

    ack = we || re;
end
