% Phase 4 SV — 3-state Moore-style FSM. Output depends only on the
% state (decoded combinationally from the state register). Tests the
% full switch-case → nested-if-else lowering on a 3-state machine
% with an `otherwise` arm and an explicit reset transition.
T = numerictype(0, 8, 0);
y = fsm_moore3(fi(1, T));
disp(y);

function out = fsm_moore3(x)
    persistent state;
    if isempty(state)
        state = fi(0, numerictype(0, 8, 0));
    end
    zero = fi(0, numerictype(0, 8, 0));
    one  = fi(1, numerictype(0, 8, 0));
    two  = fi(2, numerictype(0, 8, 0));
    out = zero;
    switch state
        case 0
            if x > zero
                state = one;
            end
        case 1
            if x > zero
                state = two;
            else
                state = zero;
            end
        case 2
            state = zero;
        otherwise
            state = zero;
    end
    if state == one
        out = one;
    elseif state == two
        out = two;
    end
end
