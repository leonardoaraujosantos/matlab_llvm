% GPU Coder Tier-1 gate: Mandelbrot via `coder.gpu.kernelfun`,
% CPU-debug lane.  Verifies the parser fold, the matlab.gpu.kernel op
% emission, the LowerGpuKernels rewrite to matlab.for, and the
% matlab_gpu_launch_kernel runtime stub all wire together correctly.
%
% Smaller grid than the example file (24x24) so the test runs fast
% under cc + AOT (this lane builds + links + runs each fixture from
% scratch).
coder.gpu.kernelfun();
n = 24;
max_iter = 30;
count = zeros(n, n);
for i = 1:n
  for j = 1:n
    cr = -2.0 + 2.5 * (i - 1) / (n - 1);
    ci = -1.25 + 2.5 * (j - 1) / (n - 1);
    zr = 0.0; zi = 0.0; k = 0;
    while (zr*zr + zi*zi <= 4.0) && (k < max_iter)
      zr_new = zr*zr - zi*zi + cr;
      zi = 2.0*zr*zi + ci;
      zr = zr_new;
      k = k + 1;
    end
    count(i, j) = k;
  end
end
fprintf('mandelbrot checksum = %.0f\n', sum(sum(count)));
