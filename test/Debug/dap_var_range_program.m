% Regression fixture: for-loop range bound is a script-level variable.
% Before the loadBinding fix, REPL/DAP-mode reads of N routed through
% matlab_ws_get_mat returning !llvm.ptr, the matlab.range op ended up
% with a (f64, !llvm.ptr) signature that LowerSeqLoops rejected, and
% ExecutionEngine::create failed with "missing LLVMTranslationDialectInterface
% for op: matlab.range".
%
% Line numbers are referenced by absolute index from dap_scenarios.py.
N = 3;
total = 0;
for i = 1:N
    total = total + i;
end
disp(total);
