// runtime/gpu/metal/runtime_gpu_metal.mm — Metal backend host driver
//
// T2 of docs/gpu_coder_roadmap.md.  Strong-overrides the weak
// matlab_gpu_launch_metal stub in runtime/gpu/runtime_gpu.cpp.
//
// This is an Objective-C++ (.mm) TU so we can `#import <Metal/Metal.h>`
// and use ObjC ARC for the MTLDevice / MTLBuffer / MTLLibrary lifecycle.
// The rest of the runtime stays plain C++; this file is the only ObjC++
// TU and is only compiled on macOS.
//
// T2.A scope (this commit):
//   - Lazy MTLDevice + MTLCommandQueue singletons
//   - matlab_gpu_metal_alloc / free / h2d / d2h: MTLBuffer round-trip
//     using MTLResourceStorageModeShared on Apple Silicon (zero-copy)
//     and StorageModeManaged on Intel Macs (with explicit syncs).
//   - matlab_gpu_launch_metal: T2 v1 is the CPU-debug fallback — calls
//     the outlined kernel function pointer in a sequential host loop
//     (same as the runtime_gpu.cpp CPU lane).  This proves Metal is
//     *active* (the dispatch routes through this file, not the weak
//     stub) and is ready for the MSL JIT + real launch in T2.B/C.
//
// T2.B (next): EmitMetal.cpp prints MSL source; matlab_gpu_register_kernel
//     stashes the (kernel_id, MSL source, fn name) tuple; first launch
//     compiles via MTLLibrary newLibraryWithSource: + caches the
//     MTLComputePipelineState.
// T2.C: encode + commit + waitUntilCompleted on the active queue.
// T2.D: MPS — wire mtimes / fft on gpuArray to MPSMatrixMultiplication
//     / MPSGraph FFT.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unistd.h>

namespace {

id<MTLDevice> g_Device = nil;
id<MTLCommandQueue> g_Queue = nil;
std::atomic<bool> g_Initialized{false};

bool isAppleSilicon() {
#if defined(__arm64__) || defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

bool ensureMetalDevice() {
  if (g_Initialized.load(std::memory_order_acquire)) return g_Device != nil;
  static std::atomic_flag InitLock = ATOMIC_FLAG_INIT;
  while (InitLock.test_and_set(std::memory_order_acquire)) { /* spin */ }
  if (!g_Initialized.load(std::memory_order_relaxed)) {
    @autoreleasepool {
      g_Device = MTLCreateSystemDefaultDevice();
      if (g_Device) {
        g_Queue = [g_Device newCommandQueue];
        if (std::getenv("MATLAB_GPU_DEBUG"))
          std::fprintf(stderr,
              "matlab_gpu_metal: device=%s, unified-memory=%d\n",
              g_Device.name.UTF8String, (int)isAppleSilicon());
      }
    }
    g_Initialized.store(true, std::memory_order_release);
  }
  InitLock.clear(std::memory_order_release);
  return g_Device != nil;
}

}  // namespace

extern "C" {

/* T2.C — MSL JIT compile + cache.  Compiles MSL source via
 * `MTLDevice newLibraryWithSource:options:error:` on first call,
 * caches the MTLComputePipelineState by source-hash so subsequent
 * launches skip the compile.  The kernel name inside the source
 * must match `kernel_name`. */
struct CachedKernel {
  id<MTLComputePipelineState> pso;
};

static std::unordered_map<std::string, CachedKernel> *kernelCache() {
  static std::unordered_map<std::string, CachedKernel> Cache;
  return &Cache;
}

extern "C" int matlab_gpu_metal_jit_compile(const char *src, const char *name,
                                            void **out_pso) {
  if (!ensureMetalDevice()) return -1;
  std::string Key(src);
  Key += "\n@@@KNAME@@@";
  Key += name;
  auto *C = kernelCache();
  auto It = C->find(Key);
  if (It != C->end()) {
    *out_pso = (__bridge void *)It->second.pso;
    return 0;
  }
  @autoreleasepool {
    NSError *err = nil;
    NSString *NsSrc = [NSString stringWithUTF8String:src];
    NSString *NsName = [NSString stringWithUTF8String:name];
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [g_Device newLibraryWithSource:NsSrc options:opts
                                                   error:&err];
    if (!lib) {
      std::fprintf(stderr, "matlab_gpu_metal_jit_compile: MSL compile "
                            "failed for kernel '%s'\n  %s\n",
                   name,
                   err ? err.localizedDescription.UTF8String : "(no error)");
      return -2;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:NsName];
    if (!fn) {
      std::fprintf(stderr, "matlab_gpu_metal_jit_compile: function '%s' "
                            "not found in compiled library\n", name);
      return -3;
    }
    id<MTLComputePipelineState> pso =
        [g_Device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      std::fprintf(stderr, "matlab_gpu_metal_jit_compile: pipeline-state "
                            "create failed for '%s': %s\n", name,
                   err ? err.localizedDescription.UTF8String : "(no error)");
      return -4;
    }
    CachedKernel Ck = {pso};
    /* Bridge-retain to keep the pso alive past the autorelease pool. */
    (*C)[Key] = Ck;
    *out_pso = (__bridge_retained void *)pso;
    return 0;
  }
}

/* Launch a precompiled kernel.  `pso_handle` is an opaque
 * MTLComputePipelineState* returned by jit_compile; `buffer_ptrs` is
 * an array of MTLBuffer* (or null for scalar args), `buffer_count` is
 * the count, `grid_size` is the number of threads to dispatch
 * (1-D, matches the end-start+1 of the MATLAB range). */
extern "C" int matlab_gpu_metal_dispatch(void *pso_handle,
                                          void **buffer_ptrs,
                                          int buffer_count,
                                          int grid_size) {
  if (!ensureMetalDevice()) return -1;
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)pso_handle;
    id<MTLCommandBuffer> cmdbuf = [g_Queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (int i = 0; i < buffer_count; ++i) {
      id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
      [enc setBuffer:buf offset:0 atIndex:i];
    }
    NSUInteger tpgw = pso.threadExecutionWidth;
    if (tpgw == 0) tpgw = 32;
    [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tpgw, 1, 1)];
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    if (cmdbuf.error) {
      std::fprintf(stderr, "matlab_gpu_metal_dispatch: error %s\n",
                   cmdbuf.error.localizedDescription.UTF8String);
      return -2;
    }
  }
  return 0;
}

/* Strong override of the weak stub in runtime_gpu.cpp.  Linker
 * selects this when the Metal TU is in the link line on macOS. */
int matlab_gpu_launch_metal(double start, double step, double end,
                            void *fn_ptr, void *state, int kernel_id) {
  if (!ensureMetalDevice()) {
    std::fprintf(stderr,
        "matlab_gpu_metal: MTLCreateSystemDefaultDevice() returned nil.\n"
        "  No Metal-capable GPU.  Fall back: MATLAB_GPU_TARGET=cpu\n");
    std::abort();
  }
  /* T2.A/C: until the AOT codegen embeds the kernel source + calls
   * jit_compile/dispatch, this dispatch arm still falls back to the
   * sequential host loop calling fn_ptr(iv, state).  The JIT compile
   * + dispatch ABI above is exercised by the standalone T2.C smoke
   * test (test/Run/gpu_metal_jit.mm). */
  if (std::getenv("MATLAB_GPU_DEBUG"))
    std::fprintf(stderr,
        "matlab_gpu_metal: launch kernel_id=%d range=[%g:%g:%g] "
        "(host fallback; AOT-to-JIT linkage is T2.C v1.1)\n",
        kernel_id, start, step, end);

  using KernelFn = void(*)(double, void *);
  KernelFn Fn = reinterpret_cast<KernelFn>(fn_ptr);
  if (step == 0.0) std::abort();
  if (step > 0.0) {
    for (double iv = start; iv <= end; iv += step) Fn(iv, state);
  } else {
    for (double iv = start; iv >= end; iv += step) Fn(iv, state);
  }
  return 0;
}

/* Device-buffer ABI.  T2.A round-trips via MTLBuffer with
 * StorageModeShared on Apple Silicon (zero-copy) and StorageModeManaged
 * on Intel Macs (explicit syncs).  Caller treats the return value as
 * an opaque void* — we cast to/from id<MTLBuffer> via __bridge_retained
 * / __bridge_transfer so ARC retain/release stays balanced.
 *
 * T2.A note: these are not yet exercised by the kernel launch arm
 * (T2.B will route gpuArray uploads through them).  Shipping the ABI
 * now lets gpuArray ctor swap matlab_gpu_upload → matlab_gpu_metal_alloc
 * with no further runtime change. */
void *matlab_gpu_metal_alloc(std::size_t bytes) {
  if (!ensureMetalDevice()) return std::malloc(bytes);
  MTLResourceOptions opts = isAppleSilicon()
      ? MTLResourceStorageModeShared
      : MTLResourceStorageModeManaged;
  @autoreleasepool {
    id<MTLBuffer> buf = [g_Device newBufferWithLength:bytes options:opts];
    return (__bridge_retained void *)buf;
  }
}

void matlab_gpu_metal_free(void *ptr) {
  if (!ptr) return;
  @autoreleasepool {
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)ptr;  /* ARC release */
    (void)buf;
  }
}

void matlab_gpu_metal_h2d(void *dst, const void *src, std::size_t bytes) {
  if (!ensureMetalDevice()) {
    std::memcpy(dst, src, bytes);
    return;
  }
  @autoreleasepool {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)dst;
    void *contents = [buf contents];
    std::memcpy(contents, src, bytes);
#if !defined(__arm64__) && !defined(__aarch64__)
    /* Intel Mac: explicitly sync the managed buffer. */
    [buf didModifyRange:NSMakeRange(0, bytes)];
#endif
  }
}

void matlab_gpu_metal_d2h(void *dst, const void *src, std::size_t bytes) {
  if (!ensureMetalDevice()) {
    std::memcpy(dst, src, bytes);
    return;
  }
  @autoreleasepool {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)src;
    void *contents = [buf contents];
    std::memcpy(dst, contents, bytes);
  }
}

void matlab_gpu_metal_sync(void) {
  /* T2.A: no pending command buffers yet (no real kernels).  T2.C
   * will wait on the current command buffer here. */
  if (g_Queue) {
    @autoreleasepool {
      id<MTLCommandBuffer> b = [g_Queue commandBuffer];
      [b commit];
      [b waitUntilCompleted];
    }
  }
}

/* T2.D — MPS GEMM.  Computes C = A * B on Apple GPU via
 * MPSMatrixMultiplication.  Inputs / output are row-major fp32
 * MTLBuffers (caller owns).  Matches MathWorks GPU Coder's cuBLAS
 * Sgemm shape; the runtime can route `mtimes(gpuArray,gpuArray)` here
 * when the active backend is Metal.
 *
 * Returns 0 on success, non-zero on error.
 *
 * v1 is row-major fp32; fp64 isn't supported by Apple GPUs.  fp16
 * (half) lane is a future addition using MPSDataTypeFloat16. */
extern "C" int matlab_gpu_metal_gemm_f32(
    void *a_buf, void *b_buf, void *c_buf,
    int M, int N, int K)
{
  if (!ensureMetalDevice()) return -1;
  @autoreleasepool {
    id<MTLBuffer> A = (__bridge id<MTLBuffer>)a_buf;
    id<MTLBuffer> B = (__bridge id<MTLBuffer>)b_buf;
    id<MTLBuffer> C = (__bridge id<MTLBuffer>)c_buf;

    MPSMatrixDescriptor *Da =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                              rowBytes:K * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *Db =
        [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                              rowBytes:N * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *Dc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                              rowBytes:N * sizeof(float)
                                              dataType:MPSDataTypeFloat32];

    MPSMatrix *MA = [[MPSMatrix alloc] initWithBuffer:A descriptor:Da];
    MPSMatrix *MB = [[MPSMatrix alloc] initWithBuffer:B descriptor:Db];
    MPSMatrix *MC = [[MPSMatrix alloc] initWithBuffer:C descriptor:Dc];

    MPSMatrixMultiplication *Mul =
        [[MPSMatrixMultiplication alloc] initWithDevice:g_Device
                                          transposeLeft:NO
                                         transposeRight:NO
                                             resultRows:M
                                          resultColumns:N
                                        interiorColumns:K
                                                  alpha:1.0
                                                   beta:0.0];

    id<MTLCommandBuffer> cmdbuf = [g_Queue commandBuffer];
    [Mul encodeToCommandBuffer:cmdbuf
                    leftMatrix:MA
                   rightMatrix:MB
                  resultMatrix:MC];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    if (cmdbuf.error) {
      std::fprintf(stderr, "matlab_gpu_metal_gemm_f32: %s\n",
                   cmdbuf.error.localizedDescription.UTF8String);
      return -2;
    }
  }
  return 0;
}

/* Device-name probe for gpuDevice() / DAP. */
const char *matlab_gpu_metal_device_name(void) {
  if (!ensureMetalDevice()) return "Metal (no device)";
  /* Buffer the name once on first call — the NSString backing storage
   * survives as long as g_Device, but we copy to a static so the
   * caller can hold the const char* indefinitely. */
  static char Name[256] = {0};
  if (!Name[0]) {
    @autoreleasepool {
      const char *N = g_Device.name.UTF8String;
      std::snprintf(Name, sizeof(Name), "%s", N ? N : "Apple GPU");
    }
  }
  return Name;
}

}  // extern "C"
