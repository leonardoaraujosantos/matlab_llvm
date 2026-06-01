% GPU Coder issue #24 — flat element-wise kernel exercising unary minus
% (matlab.neg) and division (matlab.matdiv) in the body, plus a scalar
% capture (x).  Confirms the outliner lifts it (MATLAB_GPU_OUTLINE=1)
% AND that the -emit-metal/cuda/opencl passes translate neg/div to a real
% device-kernel body (no FALLBACK), not just the identity placeholder.
coder.gpu.kernelfun();
n = 8;
x = 5.0;
y = zeros(1, n);
for i = 1:n
  y(i) = -x * i / 2.0 + 1.0;
end
fprintf('scale checksum = %.4f\n', sum(y));
