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

/* Write the figure to a file path (UTF-8). Returns 0 on success, -2 on
 * I/O error. */
int render_to_file(const Figure &fig, Format fmt, const char *path);

}  // namespace matlab_plot

#endif
