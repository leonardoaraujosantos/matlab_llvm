function [rdata, full, empty, count_out] = fifo(wdata, push, pop, reset)
    %#codegen
    % hdl: port(wdata, fi, signed, 16, 0)
    % hdl: port(push, bool)
    % hdl: port(pop, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 4-deep synchronous FIFO with full/empty flags. Storage is
    % four scalar persistents (case-decoded write port, multi-source
    % read mux). Pointers and depth tracked as separate persistent
    % counters.
    %
    % This style is what the SV backend can compile today: the
    % "natural" `mem(idx) = val` form with a runtime index needs
    % the dynamic-index gap closed (see docs/sv_supported_subset.md).
    %
    % Tests:
    %   - 4 storage cells with case-decoded write
    %   - multi-source read mux (post B-workstream slot type fix)
    %   - 3 small persistent counters (head, tail, count) updated
    %     in lockstep
    %   - flag outputs derived from counter compare

    persistent r0; persistent r1; persistent r2; persistent r3;
    persistent head; persistent tail; persistent cnt;

    if isempty(r0) || reset; r0 = fi(0, 1, 16, 0); end
    if isempty(r1) || reset; r1 = fi(0, 1, 16, 0); end
    if isempty(r2) || reset; r2 = fi(0, 1, 16, 0); end
    if isempty(r3) || reset; r3 = fi(0, 1, 16, 0); end
    if isempty(head) || reset; head = uint8(0); end
    if isempty(tail) || reset; tail = uint8(0); end
    if isempty(cnt) || reset; cnt = uint8(0); end

    % Flags computed before pointer/count updates.
    full = cnt == 4;
    empty = cnt == 0;

    % Read port: a 4-way switch-mux over the storage cells based
    % on the head pointer. Following the regfile pattern (the
    % conditional-add multi-source slot type-unification path
    % from RefineSlotTypes only fires when EVERY store is a bare
    % matlab_global_get_f64 — it doesn't trace through arith
    % adds yet).
    switch head
        case 0; rdata = r0;
        case 1; rdata = r1;
        case 2; rdata = r2;
        case 3; rdata = r3;
        otherwise; rdata = r0;
    end

    % Write port: case-decoded by tail pointer when push is asserted
    % and the FIFO isn't full.
    if push && ~full
        switch tail
            case 0; r0 = wdata;
            case 1; r1 = wdata;
            case 2; r2 = wdata;
            case 3; r3 = wdata;
            otherwise; r0 = r0;
        end
    end

    % Pointer + count updates. push/pop both: count unchanged.
    do_push = push && ~full;
    do_pop = pop && ~empty;
    if do_push
        if tail == 3
            tail = uint8(0);
        else
            tail = tail + uint8(1);
        end
    end
    if do_pop
        if head == 3
            head = uint8(0);
        else
            head = head + uint8(1);
        end
    end
    if do_push && ~do_pop
        cnt = cnt + uint8(1);
    elseif do_pop && ~do_push
        cnt = cnt - uint8(1);
    end

    count_out = cnt;
end
