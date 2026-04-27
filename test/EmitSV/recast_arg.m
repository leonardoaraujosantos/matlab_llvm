% Phase 5.6 Stage A.1 SV — fi-on-fi re-cast on a function argument.
%
% `fi(x, S, W, F)` where `x` is itself a fi-typed function arg
% lowers as a *clamp* cast (fi → fi rebind) rather than a real-
% valued constructor cast. The clamp path takes a concrete source
% spec from `fi_lhs_*` attrs that Lowering attaches when the AST's
% inferred type carries fi metadata.
%
% This pattern is the FIR / sequential_processor opener:
%   delay_line = [fi(x, 1, 16, 14), delay_line(1:3)];
%
% Sema's pre-pass scans the body for `fi(param, S, W, F)` re-cast
% sites and seeds the param's inferred type as `fixedScalar(spec)`
% (or `fixedArray(spec, Vector(N))` when also subscript-indexed).
% Lowering then emits the cast with `fi_clamp = 1` and the source
% spec as `fi_lhs_*`, and LowerFixedPoint's clamp branch takes
% over.
%
% Limitation: Sema infers the source spec from the BODY's re-cast
% TARGET (no cross-function inference). For HDL-Coder-style code
% where the call site passes a value with the same spec the body
% expects, this is correct. Mismatch silently yields a no-op
% shift — the same trade-off that all heuristic Sema inference
% has. See docs/emit_systemverilog.md Stage A.1.
T = numerictype(1, 16, 14);
y = recast_arg(fi(0.5, T));
disp(y);

function r = recast_arg(x)
    %#codegen
    % Re-cast `x` to the same fi spec — Sema infers x as
    % fi(1, 16, 14), the cast is identity, and the SV emit collapses
    % to a direct `r = x` assignment.
    r = fi(x, 1, 16, 14);
end
