# GPU Code Generation Spec

## Purpose
Document the observed behavior of the AOT GPU emit lanes
`matlabc -emit-cuda` / `-emit-metal` / `-emit-opencl`, which translate a
`coder.gpu.kernelfun`-tagged MATLAB function into a self-contained bundle: a
device-kernel source file, a host driver, and a `Makefile`. This is the GPU
analogue of the `-emit-cocotb` lane — a single MATLAB source produces a buildable
standalone project for each of the three device backends.
(doc: docs/gpu_coder_roadmap.md, src: tools/matlabc/main.cpp, src: runtime/gpu)

## Requirements

### Requirement: Three GPU emit targets
The system SHALL provide three emit modes — `-emit-cuda`, `-emit-metal`, and
`-emit-opencl` — selected via CLI flags that map to distinct internal modes.
(src: tools/matlabc/main.cpp:417-419)

#### Scenario: CUDA target selected
- **WHEN** the user runs `matlabc -emit-cuda foo.m`
- **THEN** the system SHALL select the CUDA emit mode and produce a `.cu` kernel plus a host driver

#### Scenario: Metal and OpenCL targets selected
- **WHEN** the user runs `matlabc -emit-metal foo.m` or `matlabc -emit-opencl foo.m`
- **THEN** the system SHALL select the respective mode and produce a `.metal` or `.cl` kernel with the matching host driver

### Requirement: Self-contained bundle layout
The system SHALL write the bundle as `<stem>_kernel.<ext>` (device kernel),
`<stem>_main.<host-ext>` (host driver), and a `Makefile` wiring the native
toolchain. (src: tools/matlabc/main.cpp:13432-13434)

#### Scenario: CUDA bundle files
- **WHEN** the CUDA bundle is emitted
- **THEN** the system SHALL write `<stem>_kernel.cu`, `<stem>_main.cpp`, and a `Makefile` that uses `nvcc` and links `-lcublas -lcufft -lcusolver`

#### Scenario: Metal bundle host extension
- **WHEN** the Metal bundle is emitted
- **THEN** the system SHALL write the host driver with the `.mm` extension and a `Makefile` invoking `xcrun metal` / `clang++` with the Metal frameworks

### Requirement: Output directory selection
The system SHALL default the bundle output directory to `<stem>_<target>` next to
the input file, and SHALL use an explicit `-o <dir>` value when provided (a
trailing slash is tolerated). (src: tools/matlabc/main.cpp:562, :13494-13499)

#### Scenario: Default output directory
- **WHEN** `matlabc -emit-cuda foo.m` is run without `-o`
- **THEN** the system SHALL create and write the bundle into `foo_cuda`

#### Scenario: Explicit output directory
- **WHEN** `matlabc -emit-cuda foo.m -o build/gpu/` is run
- **THEN** the system SHALL write the bundle into `build/gpu` (trailing slash stripped)

### Requirement: NVRTC/JIT host driver without offline compiler
The system SHALL emit a CUDA host driver that JIT-compiles the emitted kernel via
NVRTC at runtime (no `nvcc` required to run), and an OpenCL host driver that
JIT-compiles the `.cl` via `clBuildProgram` from the bundle directory.
(src: tools/matlabc/main.cpp:13585-13667, :13846-13885)

#### Scenario: CUDA host driver JIT-compiles the kernel
- **WHEN** the emitted CUDA `<stem>_main.cpp` is built and run
- **THEN** the system SHALL read `<stem>_kernel.cu`, compile it via NVRTC to PTX, launch it via the driver API, and print the result without requiring `nvcc`

### Requirement: Validated on-hardware backends
The system SHALL gate the CUDA and OpenCL runtime backends behind opt-in CMake
flags (`-DMATLAB_LLVM_GPU_CUDA=ON` / `-DMATLAB_LLVM_GPU_OPENCL=ON`, default OFF),
with hardware-gated validation lanes that skip cleanly when no device is present.
(doc: docs/gpu_coder_roadmap.md §Tier-3/Tier-4)

#### Scenario: CUDA validation lane on NVIDIA hardware
- **WHEN** `run_gpu_cuda_validation.sh` runs on an NVIDIA GPU
- **THEN** the system SHALL exercise the driver-API lifecycle, NVRTC AXPY kernel, and `cublasDgemm`, matching the host lane

#### Scenario: Validation skips without a device
- **WHEN** a GPU validation lane runs with no capable device present
- **THEN** the system SHALL skip cleanly rather than fail
