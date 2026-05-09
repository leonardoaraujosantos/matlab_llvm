% Regression fixture for the "Unhandled parameter attribute
% 'matlab.name'" DAP launch error. The bug fired when matlabc -dap
% compiled a program that contained a recursive user function with
% parameters: the lowering stamped `matlab.name` arg attrs on
% func.func args (so EmitC could render readable signatures), the
% LLVM-conversion pipeline propagated them to llvm.func, and then
% the JIT path (ExecutionEngine::create) rejected them. Plain
% `-emit-llvm` translation tolerated the same attrs so the bug was
% specific to the DAP / REPL JIT entrypoint.
%
% Keep the program tiny and parameter-bearing — the bug fires at
% launch time, before any breakpoint hits, so the scenario only
% needs to assert that initialize + launch succeed.

disp(fact(4));

function y = fact(n)
    if n <= 1
        y = 1;
    else
        y = n * fact(n - 1);
    end
end
