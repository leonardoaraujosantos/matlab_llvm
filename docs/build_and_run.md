# Compiling and running a `.m` source

After #50 Phase 1-5, the build-and-run flow collapses to a fixed two-command invocation. No wrapper script, no grep-based toolbox detection, no `pkg-config` discovery — the linker dead-strips unused toolboxes; native deps (Cairo, SymPP) load lazily via `dlopen` on first use.

## Prerequisites

A CMake build of matlab_llvm:

```bash
cmake -S . -B build -G Ninja \
    -DMATLAB_LLVM_WITH_MLIR=ON \
    -DMATLAB_LLVM_WITH_PLOT=ON \
    -DMATLAB_LLVM_WITH_SYM=ON    # optional, builds libmatlab_sym.dylib
ninja -C build
```

The build produces (in `build/`):

| Artefact | Always | Notes |
|---|---|---|
| `matlabc` | ✅ | the compiler / REPL |
| `libMatlabRuntime.a` | ✅ | every shipped runtime TU (core + 19 toolboxes + GPU dispatcher) consolidated into one static archive, compiled with `-ffunction-sections -fdata-sections` so external link can dead-strip per-symbol |
| `libmatlab_sym.dylib` (or `.so`) | only if `MATLAB_LLVM_WITH_SYM=ON` | SymPP-backed shared library — `dlopen`'d on first sym call by the stub layer baked into `libMatlabRuntime.a` |

## Compile + run

```bash
# 1. Lower the .m to LLVM IR.
build/matlabc -emit-llvm my_script.m > my_script.ll

# 2. Link against the runtime archive with dead-strip enabled.
clang++ -std=c++20 -O2 -Wno-override-module \
    my_script.ll \
    build/libMatlabRuntime.a \
    -ldl -lpthread \
    -Wl,-dead_strip \
    -o my_script

# 3. Run.
./my_script
```

That's all of it.

* `-Wl,-dead_strip` on macOS / `-Wl,--gc-sections` on Linux drops every runtime TU your program doesn't reference. A `disp(1)` program ends up at ~50 KB; a Mandelbrot example links only the matrix + scalar-loop code. No need to grep the source for which toolboxes are in use — the linker is ground truth.
* `-ldl` is the only non-libc external dep. macOS ignores it (libdl is part of libSystem); Linux needs it for the `dlopen` inside the plot + sym wrappers.
* `-lpthread` is needed for the `parfor` runtime even if your program doesn't call `parfor`; the runtime uses a thread pool internally.

## Plot example

```bash
build/matlabc -emit-llvm examples/plot/hello.m > /tmp/hello.ll
clang++ -std=c++20 -O2 -Wno-override-module \
    /tmp/hello.ll build/libMatlabRuntime.a \
    -ldl -lpthread -Wl,-dead_strip -o /tmp/hello
/tmp/hello                                  # writes hello.pdf
```

The binary has **no `LC_LOAD_DYLIB` for libcairo** (`otool -L /tmp/hello | grep -i cairo` is empty). The first `plot` / `figure` / `savefig` call inside the binary `dlopen`s `libcairo.dylib` from Homebrew / system paths. A program that never plots launches even on hosts without Cairo installed.

## Sym example

```bash
build/matlabc -emit-llvm /tmp/diff.m > /tmp/diff.ll
clang++ -std=c++20 -O2 -Wno-override-module \
    /tmp/diff.ll build/libMatlabRuntime.a \
    -ldl -lpthread -Wl,-dead_strip -o /tmp/diff
/tmp/diff
```

where `/tmp/diff.m` is:

```matlab
syms x
disp(diff(x^2 + 3*x, x))
```

The binary has **no `LC_LOAD_DYLIB` for libsympp**. The first `syms` / `sym` / `diff` / etc. call `dlopen`s `libmatlab_sym.dylib` (which itself transitively pulls in libsympp + GMP + MPFR). The dlopen probes `@executable_path/libmatlab_sym.dylib` first, then system paths — so a packaged install ships matlabc + libmatlab_sym.dylib side by side.

## Pre-#50 invocation

Before this work, the build-and-run was driven by `runtime/scripts/build_and_run.sh` — a 236-line shell wrapper that:

* greped the `.m` source for `syms` / `plot` / `mflowlink_run` to decide which runtime TUs to compile
* picked between `build/matlabc` and `build-sym/matlabc` based on whether the program used sym
* discovered Cairo via `pkg-config` (with GUI-PATH workarounds)
* walked `$SYMPP_PREFIX` candidates for the SymPP install
* compiled ~10 runtime TUs + the conditional plot/sym sources from source every invocation

All of that is gone. See [#50](https://github.com/leonardoaraujosantos/matlab_llvm/issues/50) for the architectural argument.
