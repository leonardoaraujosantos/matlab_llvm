% regress_logical_ops.m — regression test for scalar logical operator
% lowering on the LLVM lane.  Before the fix, matlab.and / matlab.or /
% matlab.not / matlab.short_and / matlab.short_or survived to LLVM
% translation unlowered (only the Emit* backends handled them).  They
% now lower to arith.andi / arith.ori / arith.xori in
% LowerScalarsToArith, with operands truth-coerced to i1.
%
% Note: the frontend emits both operands of && / || eagerly, so the
% lowering is eager (no runtime short-circuit) — correct for every
% case except an RHS guarded against an error by the LHS.

a = 3;
b = 0;

% --- short-circuit && / || in an if condition --------------------
if a > 1 && b < 1; disp(1); else; disp(0); end
if a > 5 || b < 1; disp(1); else; disp(0); end
if a > 5 && b < 1; disp(0); else; disp(1); end
if a > 5 || b > 5; disp(0); else; disp(1); end

% --- element-wise & / | in an if condition -----------------------
if a > 1 & b < 1; disp(1); else; disp(0); end
if a > 5 | b < 1; disp(1); else; disp(0); end

% --- unary not ---------------------------------------------------
if ~(a > 5); disp(1); else; disp(0); end
if ~(a > 1); disp(0); else; disp(1); end

% --- logical op as a stored value, then used as a condition ------
c = (a > 1) && (b < 1);
if c; disp(1); else; disp(0); end
d = (a > 5) || (b > 5);
if d; disp(0); else; disp(1); end

% --- truth-coercion of a plain numeric operand (nonzero = true) --
if a && b < 1; disp(1); else; disp(0); end
if b || a > 1; disp(1); else; disp(0); end

% --- chained && --------------------------------------------------
if a > 1 && b < 1 && a < 10; disp(1); else; disp(0); end
