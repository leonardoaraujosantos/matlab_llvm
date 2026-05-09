% Regression test for the meshgrid / ndgrid multi-return type-inference
% bug. Without the fix, `[xx, yy] = meshgrid(...)` typed both `xx` and
% `yy` as the fallback (Any), so downstream `exp(xx)` fell through to
% scalar Double and a subsequent `scalar * exp(matrix)` produced an
% arith.mulf(f64, !llvm.ptr) op that crashed the LLVM lowering pipeline:
%
%   error: 'arith.mulf' op operand #0 must be floating-point-like,
%          but got '!llvm.ptr'
%
% The fix is in lib/Sema/TypeInference.cpp (multi-return AssignStmt
% type refinement) + lib/MLIR/Lowering.cpp (multi-return MLIR result
% type table). This test exercises both paths.
%
% Note: the more complex expressions where the multi-return output
% feeds directly into scalar-minus-matrix or scalar-divide-matrix
% (e.g. `1 - xx`, `xx / 5`) trigger a *separate* emit-cpp Matrix-wrap
% bug. They are kept out of this regression to keep the meshgrid
% type-inference coverage clean; the other bug has its own slice.

% --- 1. meshgrid → exp → scalar * matrix.
[xx, yy] = meshgrid(linspace(-1, 1, 5), linspace(-1, 1, 5));
e  = exp(xx);
t1 = 0.5 * e;
disp('t1(3,3) (centre — closed form 0.5):');
fprintf('%.6f\n', t1(3, 3));

% --- 2. meshgrid → element-wise expression involving both outputs.
% z(3,3) center = exp(-0^2 - 0^2) = 1.
z = exp(-xx.^2 - yy.^2);
disp('z(3,3) (closed form 1.0):');
fprintf('%.6f\n', z(3, 3));

% --- 3. ndgrid — same multi-return shape, different convention.
[ii, jj] = ndgrid(linspace(0, 4, 5), linspace(0, 4, 5));
w = 0.25 * exp(-ii.^2 - jj.^2);
disp('w(1,1) (closed form 0.25):');
fprintf('%.6f\n', w(1, 1));
