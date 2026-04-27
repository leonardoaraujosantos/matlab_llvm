% Phase 5.6 closure SV — canonical FIR filter combining all of
% Stages B + C + D + E + F + A.1.
%
%   - Coefficient table: `h = fi([0.1, 0.2, 0.3, 0.4], ...)`
%     (Stage C: literal-init array → static `llvm.alloca [4 x
%     i16]` with constant-init stores).
%   - Persistent shift register: `delay_line = [fi(x, ...),
%     delay_line(1:3)]`. Stage F lowers the persistent fi-array
%     to N parallel scalar persistents; Stage A.1 turns the
%     `fi(x, ...)` re-cast on the fi-typed arg `x` into a no-op
%     clamp; Stage E rewrites the concat to a static-shape
%     zeros + per-element stores chain that Stage F's regular-
%     set path consumes.
%   - Sum-of-products: `delay_line(k) * h(k)` for k in [1..4].
%     Stage D handles the constant-index reads on the persistent
%     register; Stage C handles the constant-index reads on the
%     coefficient table.
%
% Expected SV shape:
%   - 4 parallel `always_ff` registers `delay_line0_0..3` for
%     the shift register.
%   - Static `logic signed [15:0] h_0_1 [4]` for the coefficient
%     table (Stage C's local).
%   - Combinational sum-of-products tree feeding the output
%     port `r`.
%
% Lint-clean under `verilator --lint-only -Wall`.
T = numerictype(1, 16, 0);
y = fir_filter(fi(7, T));
disp(y);

function r = fir_filter(x)
    %#codegen
    h = fi([1, 2, 3, 4], 1, 16, 0);
    persistent delay_line;
    if isempty(delay_line)
        delay_line = fi(zeros(1, 4), 1, 16, 0);
    end
    delay_line = [fi(x, 1, 16, 0), delay_line(1:3)];
    p1 = delay_line(1) * h(1);
    p2 = delay_line(2) * h(2);
    p3 = delay_line(3) * h(3);
    p4 = delay_line(4) * h(4);
    r = p1 + p2 + p3 + p4;
end
