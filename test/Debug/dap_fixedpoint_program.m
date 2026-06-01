% Regression fixture for issue #77 — a `-dap` launch of a program that
% uses Fixed-Point Designer (`fi`) script vars.
%
% A `fi` value is integer-encoded (Q-format) and its arithmetic lowers to
% integer shifts/muls (LowerFixedPoint). Before the "fi script vars use the
% local-slot lane in JIT/-dap" fix, a fi binding was routed through the
% ReplMode workspace, which stored/loaded it as a matrix ptr
% (matlab_ws_get_mat); a later fi op then got a `!llvm.ptr` operand where it
% needed an integer (`arith.shrsi(!llvm.ptr, i32)` -> verifier failure), so
% `matlabc -dap` answered `launch` with "failed to compile program".
%
% The binding's InferredType is usually NOT fi-typed even for `x = fi(...)`
% (Sema leaves it `any`/scalar) — the spec is only reliably visible on the
% assignment LHS type at the store, so the fix records the binding there and
% consults that on the read side. Reaching `terminated` is the assertion:
% every fi op below must lower for the program to compile + run.
x = fi(0.75, 1, 16, 8);          % stored = 192
gain = fi(1.5, 1, 16, 8);        % stored = 384
y = fi(0, 1, 16, 8);
y(:) = x * gain;                 % real-world 1.125 — read x, gain as ints
disp(y);
