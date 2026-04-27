% Phase 1 SV — bitwise builtins on integer / fi-typed operands.
% Exercises bitand / bitor / bitxor lowered to SV `&` / `|` / `^`.
% bitcmp (one-operand bitwise NOT) and bitshift are also recognized
% by the emitter; this fixture covers the binary trio. The user-call
% refinement chain plus the operand-type-driven lowering pattern in
% LowerScalarsToArith collapses the matlab.call_builtin sites to
% `arith.andi/ori/xori` once the function args refine to i16.
T = numerictype(0, 16, 0);
y = bit_combo(fi(60, T), fi(13, T));
disp(y);

function y = bit_combo(a, b)
    y = bitand(a, b) + bitor(a, b) + bitxor(a, b);
end
