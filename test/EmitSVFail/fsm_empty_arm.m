% Phase 4 v2.3 — empty case arm in an FSM cascade. The user
% wrote `case 1` with no body; this is almost always an oversight
% (state stuck without explicit transitions). Reject so the
% emitter doesn't silently render an empty `S1: begin end` arm.
T = numerictype(0, 8, 0);
y = empty_arm(fi(1, T));
disp(y);

function out = empty_arm(x)
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
        case 1
            % intentionally empty
        otherwise
            state = z;
    end
end
