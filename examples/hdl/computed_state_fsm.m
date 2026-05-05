function [active, step_out] = computed_state_fsm(advance, jump, reset)
    %#codegen
    % hdl: port(advance, bool)
    % hdl: port(jump, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % FSM-counter hybrid: walks through 8 states sequentially when
    % `advance` is asserted; jumps back to step 0 when `jump` is
    % asserted. `step_out` exposes the current step number;
    % `active` is high when not at step 0.
    %
    % Tests the boundary between FSM cascades and counters that
    % the recent gatherFSMs fix manages: the register has both
    % computed (next = step + 1) and constant (next = 0) updates.
    % After the fix, the matcher treats it as a counter (skipping
    % cascade classification) and the result is a clean
    % `step_next = step + 1` style update.

    persistent step;
    if isempty(step) || reset
        step = uint8(0);
    end

    if jump
        step = uint8(0);
    elseif advance
        if step == 7
            step = uint8(0);
        else
            step = step + uint8(1);
        end
    end

    step_out = step;
    active = step ~= 0;
end
