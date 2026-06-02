#ifndef MATLAB_PLOT_CAIRO_RENDER_H
#define MATLAB_PLOT_CAIRO_RENDER_H

#include "figure.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace matlab_plot {

enum class Format { Png, Svg, Pdf };

/* Render `fig` to a freshly-allocated byte buffer in the requested format.
 * The buffer uses malloc so it can be handed to C callers, who release it
 * via free() (matlab_plot_buffer_free in the public API). */
struct RenderResult {
    uint8_t *data = nullptr;
    size_t   size = 0;
};

RenderResult render(const Figure &fig, Format fmt);

/* Raw uncompressed frame: a freshly-painted ARGB32 image of `fig`, copied
 * out of the Cairo surface so the caller owns it independent of Cairo
 * state. `data` is malloc'd (width*height*4 bytes, premultiplied ARGB in
 * native byte order — i.e. B,G,R,A on little-endian), released with free().
 * This is the capture path behind getframe()/VideoWriter — video encoders
 * want pixels, not a re-decoded PNG. */
struct RawFrame {
    uint8_t *data = nullptr;   // malloc'd, w*h*4 bytes; null on failure
    int width = 0;
    int height = 0;
    int stride = 0;            // bytes per row (== width*4 for the copy)
};

RawFrame render_raw(const Figure &fig);

/* Write the figure to a file path (UTF-8). Returns 0 on success, -2 on
 * I/O error. */
int render_to_file(const Figure &fig, Format fmt, const char *path);

}  // namespace matlab_plot

#endif
