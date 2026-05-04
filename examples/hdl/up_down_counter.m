function count = up_down_counter(dir, en, reset)
    %#codegen
    % hdl: port(dir, bool)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 8-bit up/down counter with enable. `dir==1` increments,
    % `dir==0` decrements. Exercises:
    %   - conditional persistent set with mutually-exclusive
    %     branches (the canonical "load enable + direction"
    %     register pattern in real RTL designs)
    %   - signed wraparound semantics on i8
    %   - persistent register write under nested if/else

    persistent count_reg;
    if isempty(count_reg) || reset
        count_reg = uint8(0);
    end

    if en
        if dir
            count_reg = count_reg + uint8(1);
        else
            count_reg = count_reg - uint8(1);
        end
    end

    count = count_reg;
end
