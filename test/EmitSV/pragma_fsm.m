% Phase 4 v2.6 SV — `% hdl: fsm_encoding('one_hot')` pragma
% inside a function body overrides the CLI-wide `-sv-fsm-encoding`
% flag for that function's FSMs. Useful when a single design has
% multiple FSMs that want different encodings (e.g. a control FSM
% kept binary for area and a fast-path FSM marked one-hot for
% decode latency). The pragma scanner runs after the user-call
% iteration loop and before HWLegalize.
T = numerictype(0, 8, 0);
y = pragma_fsm(fi(1, T));
disp(y);

function out = pragma_fsm(x)
    %#codegen
    % hdl: fsm_encoding('one_hot')
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
            out = one;
        otherwise
            state = z;
    end
end
