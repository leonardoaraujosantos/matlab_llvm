function rdata = regfile(waddr, wdata, we, raddr, reset)
    %#codegen
    % hdl: port(waddr, fi, unsigned, 8, 0)
    % hdl: port(wdata, fi, signed, 16, 0)
    % hdl: port(we, bool)
    % hdl: port(raddr, fi, unsigned, 8, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)

    persistent r0; persistent r1; persistent r2; persistent r3;

    if isempty(r0) || reset; r0 = fi(0, 1, 16, 0); end
    if isempty(r1) || reset; r1 = fi(0, 1, 16, 0); end
    if isempty(r2) || reset; r2 = fi(0, 1, 16, 0); end
    if isempty(r3) || reset; r3 = fi(0, 1, 16, 0); end

    if we
        switch waddr
            case 0; r0 = wdata;
            case 1; r1 = wdata;
            case 2; r2 = wdata;
            case 3; r3 = wdata;
            otherwise; r0 = r0;
        end
    end

    if raddr == 0
        rdata = r0;
    elseif raddr == 1
        rdata = r1;
    elseif raddr == 2
        rdata = r2;
    else
        rdata = r3;
    end
end
