# Building matlab_llvm

Prerequisites:

- LLVM 22.x and MLIR
- CMake 3.20+
- Ninja
- a C++20 compiler (Clang recommended)
- Python 3 with NumPy if you want `-emit-python`

## Generic (LLVM 22 already installed)

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Or via [`just`](https://github.com/casey/just): `just build` · `just test` ·
`just repl` · `just examples`.

## Ubuntu 24.04

LLVM 22 + MLIR are not in Ubuntu's default repos. Install from
[apt.llvm.org](https://apt.llvm.org/):

```bash
# Add the LLVM 22 repository
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/llvm.gpg
echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" \
    | sudo tee /etc/apt/sources.list.d/llvm-22.list

# Install LLVM 22, MLIR, and build dependencies
sudo apt-get update
sudo apt-get install -y \
    clang-22 lld-22 llvm-22-dev libmlir-22-dev mlir-22-tools \
    cmake ninja-build libcairo2-dev libzstd-dev

# Configure and build
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-22 \
    -DCMAKE_CXX_COMPILER=clang++-22 \
    -DMATLAB_LLVM_WITH_MLIR=ON \
    -DMATLAB_LLVM_WITH_PLOT=ON \
    -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
    -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir

ninja -C build
ctest --test-dir build --output-on-failure
```

> **Video export.** `getframe` / `VideoWriter` (MP4/AVI) is on by default within a
> `WITH_PLOT` build (`MATLAB_LLVM_WITH_PLOT_FFMPEG=ON`), which links libav and needs
> the FFmpeg dev libraries (`apt install libavcodec-dev libavformat-dev libavutil-dev
> libswscale-dev`). Opt out with `-DMATLAB_LLVM_WITH_PLOT_FFMPEG=OFF`. See
> [`plotting.md`](plotting.md) §4.

## Docker

```bash
docker build --target builder -t matlab_llvm .
docker run --rm matlab_llvm ./build/matlabc -emit-llvm your_file.m
```

## Build variants

Frontend-only, without MLIR/LLVM:

```bash
cmake -S . -B build -G Ninja -DMATLAB_LLVM_WITH_MLIR=OFF
cmake --build build
```

Sanitized runtime tests (`AddressSanitizer` + `UndefinedBehaviorSanitizer`):

```bash
cmake -S . -B build-asan -G Ninja \
    -DMATLAB_LLVM_RUNTIME_ASAN=ON -DMATLAB_LLVM_WITH_MLIR=OFF
cmake --build build-asan
ctest --test-dir build-asan -R '^runtime-tests-'
```

ASan + UBSan flags are wired per-test via `ENVIRONMENT` so a single fault doesn't
abort the rest of the lane.

## Symbolic & GPU (opt-in)

- **Symbolic Math** (`-DMATLAB_LLVM_WITH_SYM=ON`): backed by
  [SymPP](https://github.com/leonardoaraujosantos/SymPP). See [`sym.md`](sym.md).
- **GPU CUDA backend** (`-DMATLAB_LLVM_GPU_CUDA=ON`): auto-discovers a system CUDA
  toolkit, else pip-wheel CUDA libs (driver API + NVRTC, no nvcc). Validate on
  hardware with `bash test/Run/run_gpu_cuda_validation.sh` (HW-gated, skips with no
  GPU). The `-emit-{cuda,metal,opencl} [-o <dir>]` bundles are emission-structural
  checked in CI (`gpu-emit-tests`); see [`gpu_coder_roadmap.md`](gpu_coder_roadmap.md).
