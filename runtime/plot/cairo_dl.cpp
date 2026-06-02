// runtime/plot/cairo_dl.cpp — lazy Cairo loader.
//
// #50 Phase 2.  Replaces direct link against libcairo with a runtime
// dlopen on first plot call.  The compiled binary has no
// LC_LOAD_DYLIB / DT_NEEDED entry for libcairo — programs that never
// plot launch on hosts without Cairo installed.  First plot call
// performs the dlopen + dlsym resolution; subsequent calls hit cached
// function pointers.
//
// Each wrapper here defines the regular `cairo_<name>` symbol; the
// linker uses these definitions instead of any libcairo it would
// otherwise be told to bring in (Cairo isn't on the link line at all
// after this phase).  The wrappers are extern "C" so the call ABI
// matches Cairo's published one.
//
// On failure to dlopen libcairo we hard-exit with a clear diagnostic
// suggesting how to install it on the host's platform — same policy
// the runtime uses for any other missing native dep.

#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cairo/cairo-svg.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

void *libcairo = nullptr;
std::atomic_flag init_lock = ATOMIC_FLAG_INIT;

void ensure_loaded() {
  if (libcairo) return;
  while (init_lock.test_and_set(std::memory_order_acquire)) { /* spin */ }
  if (!libcairo) {
    static const char *candidates[] = {
        /* macOS Homebrew + system locations. */
        "/opt/homebrew/lib/libcairo.dylib",
        "/usr/local/lib/libcairo.dylib",
        /* Linux distro locations. */
        "libcairo.so.2",
        "libcairo.so",
        /* Bare name — let the dynamic linker search its default path. */
        "libcairo.2.dylib",
        "libcairo.dylib",
        nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
      libcairo = dlopen(candidates[i], RTLD_LAZY | RTLD_GLOBAL);
      if (libcairo) break;
    }
    if (!libcairo) {
      std::fprintf(stderr,
          "matlab plot: libcairo not found. Install via\n"
          "  macOS: brew install cairo\n"
          "  Linux: apt install libcairo2\n"
          "  (dlerror: %s)\n",
          dlerror());
      std::exit(1);
    }
  }
  init_lock.clear(std::memory_order_release);
}

void *resolve(const char *name) {
  ensure_loaded();
  void *sym = dlsym(libcairo, name);
  if (!sym) {
    std::fprintf(stderr,
        "matlab plot: symbol '%s' not found in libcairo (%s)\n",
        name, dlerror());
    std::exit(1);
  }
  return sym;
}

/* DLF(<return-type>, <name>, <(params)>, <(call-args)>) emits a
 * wrapper that lazy-resolves the named libcairo symbol on first
 * call and caches the function pointer.  Each call site keeps the
 * static cache local to its own wrapper, so signature collisions
 * between different cairo functions can't share the cache. */
#define DLF(R, NAME, PARAMS, CALL) \
  extern "C" R NAME PARAMS { \
    using fp_t = R (*) PARAMS; \
    static fp_t fp = nullptr; \
    if (!fp) fp = reinterpret_cast<fp_t>(resolve(#NAME)); \
    return fp CALL; \
  }
#define DLV(NAME, PARAMS, CALL) \
  extern "C" void NAME PARAMS { \
    using fp_t = void (*) PARAMS; \
    static fp_t fp = nullptr; \
    if (!fp) fp = reinterpret_cast<fp_t>(resolve(#NAME)); \
    fp CALL; \
  }

}  // namespace

/* ----------- cairo core (alphabetised) --------------------------------- */
DLV(cairo_arc, (cairo_t *cr, double xc, double yc, double radius, double a1, double a2), (cr, xc, yc, radius, a1, a2))
DLV(cairo_clip, (cairo_t *cr), (cr))
DLV(cairo_close_path, (cairo_t *cr), (cr))
DLF(cairo_t *, cairo_create, (cairo_surface_t *target), (target))
DLV(cairo_destroy, (cairo_t *cr), (cr))
DLV(cairo_fill, (cairo_t *cr), (cr))
DLV(cairo_fill_preserve, (cairo_t *cr), (cr))
DLF(int, cairo_format_stride_for_width, (cairo_format_t format, int width), (format, width))
DLF(cairo_pattern_t *, cairo_get_source, (cairo_t *cr), (cr))
DLF(cairo_surface_t *, cairo_image_surface_create, (cairo_format_t format, int width, int height), (format, width, height))
DLF(cairo_surface_t *, cairo_image_surface_create_for_data, (unsigned char *data, cairo_format_t format, int width, int height, int stride), (data, format, width, height, stride))
DLF(unsigned char *, cairo_image_surface_get_data, (cairo_surface_t *surface), (surface))
DLF(int, cairo_image_surface_get_stride, (cairo_surface_t *surface), (surface))
DLV(cairo_line_to, (cairo_t *cr, double x, double y), (cr, x, y))
DLV(cairo_move_to, (cairo_t *cr, double x, double y), (cr, x, y))
DLV(cairo_paint, (cairo_t *cr), (cr))
DLV(cairo_pattern_set_filter, (cairo_pattern_t *p, cairo_filter_t filter), (p, filter))
DLV(cairo_rectangle, (cairo_t *cr, double x, double y, double w, double h), (cr, x, y, w, h))
DLV(cairo_restore, (cairo_t *cr), (cr))
DLV(cairo_rotate, (cairo_t *cr, double angle), (cr, angle))
DLV(cairo_save, (cairo_t *cr), (cr))
DLV(cairo_scale, (cairo_t *cr, double sx, double sy), (cr, sx, sy))
DLV(cairo_select_font_face, (cairo_t *cr, const char *family, cairo_font_slant_t slant, cairo_font_weight_t weight), (cr, family, slant, weight))
DLV(cairo_set_dash, (cairo_t *cr, const double *dashes, int n, double offset), (cr, dashes, n, offset))
DLV(cairo_set_font_size, (cairo_t *cr, double size), (cr, size))
DLV(cairo_set_line_width, (cairo_t *cr, double w), (cr, w))
DLV(cairo_set_source_rgb, (cairo_t *cr, double r, double g, double b), (cr, r, g, b))
DLV(cairo_set_source_rgba, (cairo_t *cr, double r, double g, double b, double a), (cr, r, g, b, a))
DLV(cairo_set_source_surface, (cairo_t *cr, cairo_surface_t *surface, double x, double y), (cr, surface, x, y))
DLV(cairo_show_text, (cairo_t *cr, const char *utf8), (cr, utf8))
DLV(cairo_stroke, (cairo_t *cr), (cr))
DLV(cairo_surface_destroy, (cairo_surface_t *surface), (surface))
DLV(cairo_surface_finish, (cairo_surface_t *surface), (surface))
DLV(cairo_surface_flush, (cairo_surface_t *surface), (surface))
DLF(cairo_status_t, cairo_surface_status, (cairo_surface_t *surface), (surface))
DLF(cairo_status_t, cairo_surface_write_to_png, (cairo_surface_t *surface, const char *filename), (surface, filename))
DLF(cairo_status_t, cairo_surface_write_to_png_stream, (cairo_surface_t *surface, cairo_write_func_t write_func, void *closure), (surface, write_func, closure))
DLV(cairo_text_extents, (cairo_t *cr, const char *utf8, cairo_text_extents_t *extents), (cr, utf8, extents))
DLV(cairo_translate, (cairo_t *cr, double tx, double ty), (cr, tx, ty))

/* ----------- cairo-pdf ------------------------------------------------- */
DLF(cairo_surface_t *, cairo_pdf_surface_create, (const char *filename, double width_in_points, double height_in_points), (filename, width_in_points, height_in_points))
DLF(cairo_surface_t *, cairo_pdf_surface_create_for_stream, (cairo_write_func_t write_func, void *closure, double width_in_points, double height_in_points), (write_func, closure, width_in_points, height_in_points))

/* ----------- cairo-svg ------------------------------------------------- */
DLF(cairo_surface_t *, cairo_svg_surface_create, (const char *filename, double width_in_points, double height_in_points), (filename, width_in_points, height_in_points))
DLF(cairo_surface_t *, cairo_svg_surface_create_for_stream, (cairo_write_func_t write_func, void *closure, double width_in_points, double height_in_points), (write_func, closure, width_in_points, height_in_points))
