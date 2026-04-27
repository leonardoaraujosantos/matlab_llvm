% Phase 4 SV — 2-state Mealy-style FSM detecting two consecutive 1s.
% State variable is fi-typed to keep the value path on the integer
% pipeline (the existing pipeline lowers `uint8(N)` through a runtime
% call — Phase 4.5 work — so we use `fi(N, ...)` here). The output
% depends on both the current state and the input, which is the
% Mealy distinction. The SV emitter renders the persistent `state`
% as an i8 register driven from `always_ff @(posedge clk or negedge
% rst_n)` with the next-state logic computed combinationally inside
% `always_comb` from a chain of `state == <const>` comparisons.
T = numerictype(0, 8, 0);
y = fsm_2state(fi(1, T));
disp(y);

function out = fsm_2state(x)
    persistent state;
    if isempty(state)
        state = fi(0, numerictype(0, 8, 0));
    end
    zero = fi(0, numerictype(0, 8, 0));
    one  = fi(1, numerictype(0, 8, 0));
    out = zero;
    switch state
        case 0
            if x > zero
                state = one;
            end
            out = zero;
        case 1
            if x > zero
                out = one;
            else
                state = zero;
            end
        otherwise
            state = zero;
    end
end
