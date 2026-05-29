# GPU Coder / gpuArray — Tutorial

The GPU lane lets you write GPU-Coder-style MATLAB — `gpuArray` data, the
`coder.gpu.kernelfun` pragma, `arrayfun` element-wise kernels — and compile it
through the same MLIR pipeline as the rest of the project. The design treats the
**kernel IR as the single source of truth** and targets three backends (Apple
Metal on macOS, CUDA + OpenCL on Linux). On the CPU-debug lane (the default, with
`MATLAB_GPU_TARGET` unset), kernels are rewritten back to sequential host loops
so the program runs and verifies numerically anywhere — the device backends
consume the identical front-end IR.

## Supported features

- **gpuArray allocation**: `gpuArray.rand`, `gpuArray.linspace`, single-precision
  storage (`'single'`).
- **Host↔device transfer**: `gather`.
- **Element-wise / BLAS-like ops on gpuArrays**: `.*`, `+`, matrix `*` (GEMM
  dispatch), `norm(·, 'fro')`.
- **Kernel generation**: `coder.gpu.kernelfun()` pragma (maps `for`/`while`
  loops to device kernels), `arrayfun(@kernel, x)` element-wise kernel launch.
- **Device management**: `gpuDevice`, `gpuDeviceCount`.
- **Parallel scheduling**: `parfor` over GPU batches, round-robin multi-GPU
  selection.

## Build & run

```bash
build/matlabc -emit-llvm examples/gpu/mandelbrot_gpu.m > /tmp/mandelbrot_gpu.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/mandelbrot_gpu.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/mandelbrot_gpu
/tmp/mandelbrot_gpu
```

The CPU-debug lane runs with no device toolkit installed. Device emission (Metal
/ CUDA / OpenCL) selects via `MATLAB_GPU_TARGET`; the same `.m` source feeds
every backend.

## Worked examples

### Mandelbrot — the canonical kernelfun demo  (`examples/gpu/mandelbrot_gpu.m`)

The MathWorks getting-started demo. `coder.gpu.kernelfun()` tags the function so
every loop maps to a device kernel; on the CPU lane the body runs as a sequential
loop with the kernel-launch ABI live.

```matlab
function mandelbrot_gpu()
  coder.gpu.kernelfun();
  n_grid = 64;  max_iter = 50;
  count = zeros(n_grid, n_grid);
  for i = 1:n_grid
    for j = 1:n_grid
      cr = -2.0 + 2.5 * (i - 1) / (n_grid - 1);
      ci = -1.25 + 2.5 * (j - 1) / (n_grid - 1);
      zr = 0.0;  zi = 0.0;  k = 0;
      while (zr*zr + zi*zi <= 4.0) && (k < max_iter)
        zr_new = zr*zr - zi*zi + cr;
        zi = 2.0*zr*zi + ci;
        zr = zr_new;
        k = k + 1;
      end
      count(i, j) = k;
    end
  end
  s = sum(sum(count));
  fprintf('mandelbrot_gpu: checksum = %.0f\n', s);
end
```

The test fixture compares the escape-count checksum against a known-good
reference so the kernel rewrite is bit-exact vs the CPU pipeline. This single
example exercises kernel outlining, grid sizing, and the launch ABI.

### AXPY on gpuArrays  (`examples/gpu/test_gpuarray_axpy.m`)

Element-wise `a.*x + b` on `gpuArray.rand` data, validated with a `gather`
round-trip against the host reference within fp32 tolerance.

```matlab
function err = test_gpuarray_axpy(n)
    a = single(2.5);
    x = gpuArray.rand(n, 1, 'single');
    b = gpuArray.rand(n, 1, 'single');
    y_gpu = a .* x + b;
    y = gather(y_gpu);
    y_ref = a .* gather(x) + gather(b);
    err = max(abs(y - y_ref));
    fprintf('AXPY error = %.8g\n', err);
end
```

This is the GPU-Coder validation rubric in miniature: allocate on device, do the
element-wise op, gather, and confirm numerical correctness.

### Matrix multiply — GEMM dispatch  (`examples/gpu/test_gpuarray_gemm.m`)

Matrix-matrix `*` on gpuArrays dispatches to the BLAS-like backend (cuBLAS on
CUDA, MPSMatrixMultiplication on Metal, clBlast on OpenCL; host `matlab_matmul`
on the CPU lane).

```matlab
function C = test_gpuarray_gemm(n)
    A = gpuArray.rand(n, n, 'single');
    B = gpuArray.rand(n, n, 'single');
    tic;
    Cgpu = A * B;
    gpuTime = toc;
    C = gather(Cgpu);
    fprintf('GPU matrix multiply time = %.4f s\n', gpuTime);
end
```

### arrayfun element-wise kernel  (`examples/gpu/test_gpuarray_arrayfun.m`)

`arrayfun(@kernel, x)` over a gpuArray generates a per-element grid launch; the
scalar kernel body is what gets lowered to MSL / CUDA-C / OpenCL-C at codegen.

```matlab
function y = test_gpuarray_arrayfun(n)
    x = gpuArray.linspace(single(-10), single(10), n);
    y_gpu = arrayfun(@activation_kernel, x);
    y = gather(y_gpu);
end

function y = activation_kernel(x)
    y = 1 ./ (1 + exp(-x));   % sigmoid — exercises exp() in the kernel
end
```

The kernel uses `exp()`, which each backend binds to its per-target math
library.

### Multi-GPU parfor scheduling  (`examples/gpu/test_parfor_multigpu.m`)

Round-robin device selection across `parfor` workers: each iteration picks a
device by modular hashing against `gpuDeviceCount`, runs a GEMM, and gathers the
Frobenius norm.

```matlab
function results = test_parfor_multigpu(numBatches, n)
    gcount = gpuDeviceCount;
    results = zeros(numBatches, 1);
    parfor i = 1:numBatches
        gpuId = mod(i - 1, gcount) + 1;
        gpuDevice(gpuId);
        A = gpuArray.rand(n, n, 'single');
        B = gpuArray.rand(n, n, 'single');
        C = A * B;
        results(i) = gather(norm(C, 'fro'));
    end
end
```

On a single-GPU box `gpuDeviceCount` = 1 and all workers fall back to the same
device. The sibling `test_parfor_gpu_batches.m` and `test_gpuarray_stencil2d.m`
cover batched parfor and stencil patterns; `benchmark_gpu_backend.m` and
`run_validation_suite.m` drive the numerical-equivalence sweep.

## Limitations & carve-outs

- Default lane is **CPU-debug**: kernels run as sequential host loops. Device
  backends (Metal / CUDA / OpenCL) are the codegen target; this is a
  kernel-codegen roadmap, not a deployment one.
- **Deep Learning on GPU** (cuDNN / TensorRT / MPSCNN, `dlarray`/`dlnetwork`
  half/INT8) is carved out to the DL roadmap.
- **Embedded boards** (Jetson / DRIVE), the **GPU Coder App GUI**, **External
  Mode** parameter tuning, and **packNGo-over-network** are out of scope. The
  headless `coder.gpuConfig` carrier is the CLI equivalent of the app.
- Code-generation **Reports** beyond a basic kernel summary (`-gpu-report`) are
  out of scope.

## See also

- Roadmap / design: [`../gpu_coder_roadmap.md`](../gpu_coder_roadmap.md)
- Examples: `examples/gpu/`
