% GPU Coder issue #24 — flat element-wise kernel with an inner WHILE loop
% and scalar temporaries (v, k).  Exercises the emitter's matlab.while +
% relational (matlab.gt) + outer-local-slot translation: each scalar temp
% becomes a per-thread MSL/CUDA/OpenCL local, and the while loop maps to a
% device while loop.  Outlines on MATLAB_GPU_OUTLINE=1 and the
% -emit-{metal,cuda,opencl} passes produce a real body (no FALLBACK).
coder.gpu.kernelfun();
n = 6;
y = zeros(1, n);
for i = 1:n
  v = i;
  k = 0;
  while v > 1.5
    v = v / 2.0;
    k = k + 1;
  end
  y(i) = k;
end
fprintf('while checksum = %.0f\n', sum(y));
