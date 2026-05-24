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
#import <Foundation/Foundation.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  /* T2.A: Metal device + queue are live, but kernel-source JIT lands
   * in T2.B/C.  Until then, run the body sequentially on the host
   * (same as the CPU-debug lane).  This proves the Metal dispatch
   * arm is active — the weak stub aborts with "not built in", but
   * with this strong override we run cleanly.  An env-var sentinel
   * lets the test harness assert that the Metal path was selected. */
  if (std::getenv("MATLAB_GPU_DEBUG"))
    std::fprintf(stderr,
        "matlab_gpu_metal: launch kernel_id=%d range=[%g:%g:%g] (CPU fallback "
        "until T2.B/C MSL JIT lands)\n", kernel_id, start, step, end);

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
