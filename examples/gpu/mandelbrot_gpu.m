% mandelbrot_gpu.m — the canonical GPU Coder getting-started demo.
%
% Reference: MathWorks "GPU Code Generation: The Mandelbrot Set"
% (https://www.mathworks.com/help/gpucoder/gs/gpu-code-generation-the-mandelbrot-set.html)
%
% The `coder.gpu.kernelfun` pragma tells the GPU lane to map every
% `for` loop in this function to a device kernel. In T1, the
% CPU-debug lane (MATLAB_GPU_TARGET unset or "cpu") rewrites the
% kernel back to a sequential `matlab.for` and runs it on the host
% — same code path as a normal for-loop, but with the GPU pragma
% recognised, AST + IR carried through the pipeline, and the
% kernel-launch ABI live.  T2 (Metal) will outline the body to an
% MSL kernel and dispatch through MTLCommandQueue.
%
% Verification: the final image checksum (sum of all escape-iteration
% counts) is printed.  The test fixture compares this single number
% against a known-good reference so the kernel rewrite is bit-exact
% vs the CPU pipeline.
function mandelbrot_gpu()
  coder.gpu.kernelfun();

  % Grid: a 64x64 sub-region of the complex plane, ranging from
  % -2 to 0.5 on the real axis and -1.25 to 1.25 on the imaginary.
  % MathWorks uses 1000x1000 but 64x64 is enough to cover the main
  % cardioid + the period-2 bulb (the visible Mandelbrot landmarks)
  % while keeping the test fixture's runtime under a second.
  n_grid = 64;
  max_iter = 50;
  count = zeros(n_grid, n_grid);

  re_min = -2.0;
  re_max = 0.5;
  im_min = -1.25;
  im_max = 1.25;

  for i = 1:n_grid
    for j = 1:n_grid
      cr = re_min + (re_max - re_min) * (i - 1) / (n_grid - 1);
      ci = im_min + (im_max - im_min) * (j - 1) / (n_grid - 1);
      zr = 0.0;
      zi = 0.0;
      k = 0;
      while (zr*zr + zi*zi <= 4.0) && (k < max_iter)
        zr_new = zr*zr - zi*zi + cr;
        zi = 2.0*zr*zi + ci;
        zr = zr_new;
        k = k + 1;
      end
      count(i, j) = k;
    end
  end

  % Single-number summary for the test fixture.
  s = sum(sum(count));
  fprintf('mandelbrot_gpu: checksum = %.0f\n', s);
end
