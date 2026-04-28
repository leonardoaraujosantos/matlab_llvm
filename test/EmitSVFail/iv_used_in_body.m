% Phase 5.6 closure — runtime fi-quantize on a (post-unroll
% constant) iv survives to HWLegalize as an unhandled runtime
% call. After the Stage F.2 unroller, the iv `i` becomes a
% per-iteration `arith.constant : f64` and the body's `fi(i,
% numerictype(...))` lowers to `matlab_fi_quantize_s(i_const)` —
% a runtime call that has no synthesizable form. The proper
% closure is to constant-fold this at lower-time (similar to the
% `fi(literal, ...)` constant-fold path), but until that lands
% the gate correctly rejects the program.
T = numerictype(1, 16, 0);
y = bad(fi(0, T));
disp(y);

function y = bad(seed)
    y = seed;
    for i = 1:4
        y = y + fi(i, numerictype(1, 16, 0));
    end
end
