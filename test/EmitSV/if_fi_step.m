% Phase 4.5.2 SV — natural MATLAB `if <fi>` truthy form lowers to
% `<fi> != 0` automatically, without requiring the user to write
% `> fi(0, ...)` by hand. The MIR-to-MLIR lowering inserts an
% `unrealized_conversion_cast` placeholder when the cond is `none`-
% typed (function arg whose type lands later); the RefineIfConds
% pass replaces the placeholder with a real `arith.cmpi ne` once
% type-flow refines.
T = numerictype(0, 1, 0);
y = if_fi_step(fi(1, T), fi(0, T));
disp(y);

function out = if_fi_step(en, rst)
    persistent c;
    if isempty(c)
        c = fi(0, numerictype(0, 8, 0));
    end
    if rst
        c = fi(0, numerictype(0, 8, 0));
    elseif en
        c = c + fi(1, numerictype(0, 8, 0));
    end
    out = c;
end
