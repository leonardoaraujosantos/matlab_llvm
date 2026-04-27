% Phase 4 v2.3 — duplicate case label in an FSM cascade.
% Two `case 1` arms in the same switch is unambiguously a bug:
% the second arm is unreachable. The synthesizability gate flags
% this rather than silently emitting an `unique case` with two
% S1: arms (which would itself be illegal SV).
T = numerictype(0, 8, 0);
y = bad_fsm(fi(1, T));
disp(y);

function out = bad_fsm(x)
    persistent state;
    if isempty(state)
        state = fi(0, numerictype(0, 8, 0));
    end
    z = fi(0, numerictype(0, 8, 0));
    one = fi(1, numerictype(0, 8, 0));
    out = z;
    switch state
        case 0
            if x > z
                state = one;
            end
            out = z;
        case 1
            state = z;
            out = one;
        case 1               % duplicate — unreachable
            state = one;
            out = z;
        otherwise
            state = z;
    end
end
