% bench_mandelbrot.m — scalar-inner-loop benchmark. Complements the
% LAPACK kernels: where those show matlab_llvm matching NumPy on dense
% linalg (because BLAS / LAPACK win the kernel race), this shows the
% opposite story — pure scalar arithmetic per pixel, no vectorisable
% structure for NumPy to exploit cheaply. matlab_llvm's LLVM JIT
% compiles the inner loop down to tight scalar code.
%
% N is the image side; max_iter=100 caps the escape-time iteration.
N = __BENCH_N__;
max_iter = 100;
re_min = -2.0; re_max = 1.0;
im_min = -1.5; im_max = 1.5;
counts = zeros(N, N);
best = Inf;
for trial = 1:3
    tic;
    for py = 1:N
        cim = im_min + (im_max - im_min) * (py - 1) / (N - 1);
        for px = 1:N
            cre = re_min + (re_max - re_min) * (px - 1) / (N - 1);
            zre = 0.0; zim = 0.0;
            count = 0;
            for k = 1:max_iter
                zre2 = zre * zre - zim * zim + cre;
                zim2 = 2.0 * zre * zim + cim;
                zre = zre2;
                zim = zim2;
                if zre * zre + zim * zim > 4.0
                    break;
                end
                count = k;
            end
            counts(py, px) = count;
        end
    end
    elapsed = toc;
    if elapsed < best
        best = elapsed;
    end
end
fprintf('mandelbrot N=%d best=%.9f s\n', N, best);
