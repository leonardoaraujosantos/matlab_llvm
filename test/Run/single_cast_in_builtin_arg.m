% Issue #22 regression: `single()` / `double()` casts in the argument
% list of a runtime-table builtin must dispatch cleanly.  Without the
% fix, the matcher in LowerTensorOps rejects the (f32, f32, f64) shape
% because the table tags 'fff' as strict-f64-only — and the call falls
% all the way through to LowerToLLVMIR's "unsupported call shape" error.
%
% The fix broadens 'f' to accept f32 (single-cast result) and emits an
% fpext at the call site to widen back to the runtime ABI's double.
%
% Also covers the companion PromoteNoneParams fix: function-param `n`
% with no in-module caller stays `none`-typed unless the gpuArray_*
% prefix is recognised as a scalar-numeric consumer.

% --- Inline literal casts feed linspace as (f32, f32, f64). ---
a = gpuArray.linspace(single(-1), single(1), 8);
fprintf('a-len = %d\n', numel(a));

% --- Mixed single/double cast in the same call. ---
b = gpuArray.linspace(double(-1), single(1), 16);
fprintf('b-len = %d\n', numel(b));

% --- Function-param `n` reaches gpuArray.linspace via the helper
%     below; PromoteNoneParams must promote it to f64. ---
c = exercise_param(32);
fprintf('c-len = %d\n', numel(c));

function out = exercise_param(n)
    out = gpuArray.linspace(single(-1), single(1), n);
end
