% regress_plot_handle_assign.m — regression for
% `hSurf = surfc(X, Y, Z)` and the surf / mesh / surfl variants.
%
% Before the fix, the void-returning plot rewriters in LowerPlot
% erased the original matlab.call_builtin op while its `none`-typed
% result was still consumed by a matlab.store (file-emit) or
% matlab.call_builtin @matlab_ws_set_obj (REPL) into the user's
% named slot. The dangling-operand chain crashed the MLIR verifier
% intermittently when it tried to format the offending op for a
% diagnostic — observed as a SIGSEGV in the REPL JIT and a flaky
% propagateLiveness crash in the file-emit pipeline's trailing
% LowerScalarsToArith sweep.
%
% This test exercises each affected handler shape end-to-end:
%   - surfc with assignment
%   - surf with assignment
%   - mesh with assignment
%   - surfl with assignment
% The disp sentinels confirm execution reached each subsequent
% statement (a JIT/pipeline crash on an earlier line would abort
% before the print). The numeric checks at the end confirm that
% the input matrices are still accessible after the surf chain
% (the dead-store sweep musn't take down the source operands).

[X, Y, Z] = peaks(8);

hSurf = surfc(X, Y, Z);
disp('SURFC_OK')

hSurf2 = surf(X, Y, Z);
disp('SURF_OK')

hMesh = mesh(X, Y, Z);
disp('MESH_OK')

hSurf3 = surfl(X, Y, Z);
disp('SURFL_OK')

% Inputs still readable — confirms the input chain wasn't accidentally
% reaped during user-erasure of the surf result.
disp(size(X, 1))
disp(size(Z, 2))
