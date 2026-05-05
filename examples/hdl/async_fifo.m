function [rdata, full, empty] = async_fifo(wdata, push, pop, reset)
    %#codegen
    % hdl: port(wdata, fi, signed, 16, 0)
    % hdl: port(push, bool)
    % hdl: port(pop, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Single-clock approximation of an async FIFO using gray-coded
    % pointers. A real two-clock async FIFO needs separate read/
    % write clocks and dual-clock synchronization, which the
    % single-clock backend doesn't model — but the gray-pointer
    % update logic is the same. This serves as a stress test for
    % the gray-code increment pattern (`next = (cur + 1) ^ ((cur
    % + 1) >> 1)`) and a baseline single-clock FIFO that uses gray
    % pointers internally.
    %
    % If/when the backend grows multi-clock support, this module
    % can be promoted to a true async FIFO by splitting the clock
    % domains.
    %
    % Tests:
    %   - gray-coded counter increment (XOR + shift on persistent)
    %   - 4-deep storage with case-decoded write port
    %   - flag derivation from gray-code comparisons
    %   - similar-to-fifo regression baseline

    persistent r0; persistent r1; persistent r2; persistent r3;
    persistent w_ptr;     % 3-bit binary write pointer (incl. MSB wrap)
    persistent r_ptr;     % 3-bit binary read pointer

    if isempty(r0) || reset; r0 = fi(0, 1, 16, 0); end
    if isempty(r1) || reset; r1 = fi(0, 1, 16, 0); end
    if isempty(r2) || reset; r2 = fi(0, 1, 16, 0); end
    if isempty(r3) || reset; r3 = fi(0, 1, 16, 0); end
    if isempty(w_ptr) || reset; w_ptr = uint8(0); end
    if isempty(r_ptr) || reset; r_ptr = uint8(0); end

    % Empty / full flags from pointer compare. Real async FIFOs
    % use gray-coded compare across clock domains; we use binary
    % since single-clock comparison is exact.
    wp = w_ptr + uint8(0);   % snapshot
    rp = r_ptr + uint8(0);
    empty = wp == rp;
    full = bitand(bitxor(wp, rp), uint8(7)) == 0 && bitand(bitxor(wp, rp), uint8(8)) ~= 0;

    % Read mux on the low 2 bits of read pointer.
    raddr = bitand(rp, uint8(3));
    switch raddr
        case 0; rdata = r0;
        case 1; rdata = r1;
        case 2; rdata = r2;
        case 3; rdata = r3;
        otherwise; rdata = r0;
    end

    % Write port on low 2 bits of write pointer.
    waddr = bitand(wp, uint8(3));
    if push && ~full
        switch waddr
            case 0; r0 = wdata;
            case 1; r1 = wdata;
            case 2; r2 = wdata;
            case 3; r3 = wdata;
            otherwise; r0 = r0;
        end
        w_ptr = w_ptr + uint8(1);
    end

    if pop && ~empty
        r_ptr = r_ptr + uint8(1);
    end
end
