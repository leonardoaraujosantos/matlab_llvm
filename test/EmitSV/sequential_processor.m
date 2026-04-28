% Phase 5.6 closure SV — full sequential FIR processor.
%
% Verbatim copy of `examples/hdl/sequential_processor.m` with the
% `% hdl: port(...)` pragmas at the top so the function compiles
% standalone (no driver). Composes every Phase 5.6 stage in one
% module:
%
%   - Stage A.1: `fi(x, 1, 16, 14)` and `fi(gain, 1, 16, 12)` re-
%                cast on fi-typed function args (clamp form).
%   - Stage B:   `% hdl: port(...)` pragmas declare the port
%                types so the function emits without a typed
%                driver caller.
%   - Stage C:   `h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15)` —
%                literal coefficient table.
%   - Stage D:   `for i = 1:4; ... delay_line(i) ... h(i) ...
%                end` — loop-iv subscripts on static arrays.
%   - Stage E:   `[fi(x, ...), delay_line(1:3)]` — concat of
%                scalar + slice with static shapes.
%   - Stage F:   persistent fi-array shift register +
%                F.2 IR-level for-loop unroll so the loop-iv
%                subscripts on the persistent array become
%                per-iteration constants.
%   - `acc(:) = acc + prod` — scalar colon-assign as a regular
%                store (the `(:)` is a type-preserving rebind on
%                a scalar fi local; it lowers to a normal slot
%                store, no special handling).
%   - `if isempty(delay_line) || reset` — multi-guard init for
%                an array persistent. Each synthetic per-element
%                scalar persistent inherits the OR'd condition,
%                producing a clean `if (rst_n_low || reset) ...`
%                reset chain in the always_ff.
%   - `(full_res > 32767) || (full_res < -32768)` — overflow
%                check via short-circuit OR. Lowers via the new
%                `matlab.short_or` SV op handler.
function [y, ovfl] = sequential_processor(x, gain, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % hdl: port(reset, bool)
    h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15);
    persistent delay_line;
    if isempty(delay_line) || reset
        delay_line = fi(zeros(1, 4), 1, 16, 14);
    end
    delay_line = [fi(x, 1, 16, 14), delay_line(1:3)];
    acc = fi(0, 1, 36, 29);
    for i = 1:4
        prod = delay_line(i) * h(i);
        acc(:) = acc + prod;
    end
    full_res = acc * fi(gain, 1, 16, 12);
    y = fi(full_res, 1, 16, 12, 'OverflowAction', 'Saturate');
    ovfl = (full_res > 32767) || (full_res < -32768);
end
