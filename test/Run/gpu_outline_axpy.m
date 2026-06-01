% GPU Coder issue #24 — flat element-wise `coder.gpu.kernelfun` kernel.
% Exercises the real array-capture outliner (MATLAB_GPU_OUTLINE=1): the
% loop body writes an output-array slot indexed by the induction
% variable, with no scalar temporaries or nested control flow — the
% canonical AXPY/map pattern the outliner lifts into a standalone
% __gpu_kernel_N llvm.func.  On the default lane it rewrites to a
% sequential matlab.for; both lanes must produce the same checksum.
coder.gpu.kernelfun();
n = 8;
y = zeros(1, n);
for i = 1:n
  y(i) = 2.0 * i + 1.0;
end
fprintf('axpy checksum = %.0f\n', sum(y));
