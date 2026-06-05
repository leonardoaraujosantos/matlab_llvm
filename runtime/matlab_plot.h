#ifndef MATLAB_PLOT_H
#define MATLAB_PLOT_H

/* matlab_plot — cross-platform plotting runtime for matlab_llvm.
 *
 * Backend: Cairo (image / svg / pdf surfaces). Headless by default; no
 * subprocess, no temp files, no display server required. Targets macOS,
 * Linux, and iOS (sandbox-safe).
 *
 * ABI: pure C, mirrors runtime/matlab_runtime.h conventions. Strings are
 * passed as (const char *, int64_t length) — not zero-terminated. Matrix
 * args are matlab_mat* opaque pointers shared with matlab_runtime.h.
 *
 * Threading: each figure is non-thread-safe. The "current figure" is a
 * per-thread slot, matching MATLAB's gcf semantics.
 */

#include <stdint.h>
#include <stddef.h>

#include "matlab_runtime.h"  /* matlab_mat, matlab_cell */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque figure handle. */
typedef struct matlab_figure matlab_figure;

/* Opaque captured-frame handle (getframe). Holds one rendered raster of a
 * figure; consumed by matlab_videowriter_write. */
typedef struct matlab_frame matlab_frame;

/* Opaque video-writer handle (VideoWriter). Encodes a sequence of frames
 * to a container file via libav (when built with
 * MATLAB_LLVM_WITH_PLOT_FFMPEG). */
typedef struct matlab_videowriter matlab_videowriter;

/* In-memory render buffer. Caller must free via matlab_plot_buffer_free. */
typedef struct {
    uint8_t *data;
    size_t   size;
} matlab_plot_buffer;

/* --------------------------------------------------------------------------
 * Figure lifecycle
 * -------------------------------------------------------------------------- */

/* figure() — create a new figure and make it current. */
matlab_figure *matlab_figure_new(void);

/* figure(n) — select / create figure with integer id; matches MATLAB's
 * `figure(N)` which raises or creates figure number N. */
matlab_figure *matlab_figure_by_id(double id);

/* gcf — current figure on this thread; lazily creates one if none. */
matlab_figure *matlab_gcf(void);

/* close all — destroys every figure created on this thread. */
void matlab_close_all(void);

/* subplot(rows, cols, i) — switch the active axes within the current
 * figure to the i-th cell of an rows×cols grid (1-based, row-major).
 * Reshapes the grid if dimensions changed; out-of-range i is a no-op. */
void matlab_subplot(double rows, double cols, double i);

/* --------------------------------------------------------------------------
 * Plot primitives — operate on the current figure / current axes.
 * -------------------------------------------------------------------------- */

/* plot(y) — plot vector y against its 1-based index. */
void matlab_plot1(matlab_mat *y);

/* plot(x, y) — line plot of y vs x. */
void matlab_plot2(matlab_mat *x, matlab_mat *y);

/* plot(x, y, fmt) — line plot with MATLAB linespec, e.g. 'r--', 'ko-'. */
void matlab_plot_fmt(matlab_mat *x, matlab_mat *y,
                     const char *fmt, int64_t flen);

/* plot3(x, y, z) — 3-D line plot. Switches the axes into 3D mode and
 * uses the active view (default az=-37.5°, el=30°, MATLAB's default). */
void matlab_plot3   (matlab_mat *x, matlab_mat *y, matlab_mat *z);
void matlab_plot3_fmt(matlab_mat *x, matlab_mat *y, matlab_mat *z,
                      const char *fmt, int64_t flen);

/* view(az, el) — set 3D camera azimuth and elevation in degrees.
 * Affects the active axes; no-op for 2D axes. */
void matlab_view(double az, double el);

/* zlabel — only meaningful on 3D axes. */
void matlab_zlabel(const char *s, int64_t n);

/* surf(Z) / surf(X, Y, Z) — filled 3-D surface mesh. Tessellated to
 * m×n quads, sorted back-to-front (painter's algorithm), shaded by
 * per-face Z value through the active colormap. matlab_surf1 takes
 * only Z (auto x = 1..cols, y = 1..rows). */
void matlab_surf1(matlab_mat *Z);
void matlab_surf3(matlab_mat *X, matlab_mat *Y, matlab_mat *Z);

/* pdeplot(nodes, triangles, nodal_data) — Unstructured 2-D
 * triangle-mesh painter for Tier-1 FEM solution visualisation.
 *
 *   nodes      : Nn × 2 — x, y per vertex.
 *   triangles  : Nt × 3 (1-based vertex indices) or Nt × 4 (with a
 *                 leading face-id column matching the STL/GLB
 *                 convention).
 *   nodal_data : Nn × 1 per-vertex scalar (Gouraud-mean per
 *                 triangle) or NULL (depth-tint fallback).
 */
void matlab_pdeplot(matlab_mat *nodes, matlab_mat *triangles,
                    matlab_mat *nodal_data);

/* pdeplot3D(nodes, triangles, nodal_data) — Unstructured 3-D
 * triangle-mesh painter for FEM / STL / GLB visualisation.
 *
 *   nodes      : Nn × 3 — x, y, z per vertex.
 *   triangles  : Nt × 3 (or Nt × 4 with a face_id column at column 0,
 *                 matching the runtime_pde Faces convention) — 1-based
 *                 vertex indices per triangle.
 *   nodal_data : either Nn × 1 (per-vertex scalar, Gouraud-shaded) or
 *                 Nt × 1 (per-face scalar) or NULL (depth tint only).
 *
 * Optional companions set the deformation field + scale factor before
 * the next pdeplot3D call:
 *   pdeplot3d_deformation(disp_Nx3)  — Nn × 3 displacement.
 *   pdeplot3d_deform_scale(s)        — multiplier (default 1).
 */
void matlab_pdeplot3d(matlab_mat *nodes, matlab_mat *triangles,
                      matlab_mat *nodal_data);
void matlab_pdeplot3d_deformation(matlab_mat *disp);
void matlab_pdeplot3d_deform_scale(double s);

/* mesh(Z) / mesh(X, Y, Z) — wireframe surface (edges only, no fill). */
void matlab_mesh1(matlab_mat *Z);
void matlab_mesh3(matlab_mat *X, matlab_mat *Y, matlab_mat *Z);

void matlab_scatter(matlab_mat *x, matlab_mat *y);

/* errorbar(x, y, e) — line plot with symmetric vertical error bars
 * (y ± e). Each point gets a vertical bar from y-e to y+e plus hat
 * caps at both ends. */
void matlab_errorbar (matlab_mat *x, matlab_mat *y, matlab_mat *e);

/* errorbar(x, y, neg, pos) — asymmetric error bars; bar runs from
 * y - neg[i] to y + pos[i]. */
void matlab_errorbar2(matlab_mat *x, matlab_mat *y,
                      matlab_mat *neg, matlab_mat *pos);

/* stem(x, y) / stem(y) — vertical line from y=0 to each point with a
 * circle marker at the top. Common in DSP signal display. */
void matlab_stem1(matlab_mat *y);
void matlab_stem2(matlab_mat *x, matlab_mat *y);

/* stairs(x, y) / stairs(y) — staircase polyline (horizontal then vertical
 * segments between adjacent points). */
void matlab_stairs1(matlab_mat *y);
void matlab_stairs2(matlab_mat *x, matlab_mat *y);

/* histogram(data, nbins) — bin `data` into `nbins` equal-width buckets
 * and draw the counts as flush bars. nbins <= 0 picks a Sturges-like
 * default of round(sqrt(numel(data))). */
void matlab_histogram(matlab_mat *data, double nbins);

/* area(x, y) / area(y) — line plot with the region between the curve
 * and the y=0 baseline filled. */
void matlab_area1(matlab_mat *y);
void matlab_area2(matlab_mat *x, matlab_mat *y);

/* text(x, y, str) — place an annotation at data coordinates (x, y).
 * Multiple text() calls accumulate; cleared on next plot when the axes
 * is not in hold mode. */
void matlab_text(double x, double y, const char *s, int64_t n);

/* yyaxis('left') / yyaxis('right') — switch the active y-axis. New
 * series go to whichever side is active; each side keeps its own
 * y-limits and ylabel. */
void matlab_yyaxis(const char *side, int64_t n);

/* xticks(v) / yticks(v) — override the auto-generated tick locations
 * with the values in `v` (a row vector). Empty matrix restores auto. */
void matlab_xticks(matlab_mat *v);
void matlab_yticks(matlab_mat *v);

/* xticklabels({'a','b',...}) / yticklabels(...) — labels printed at the
 * current tick locations. Array form takes parallel ptr/len arrays
 * (the same shape codegen produces for legend's varargs path). */
void matlab_xticklabels_strs(const char *const *labels,
                             const int64_t *lens, int64_t n);
void matlab_yticklabels_strs(const char *const *labels,
                             const int64_t *lens, int64_t n);

/* xline(value) / yline(value) — reference lines at constant x or y.
 * The labelled variants render the supplied text alongside the line. */
void matlab_xline      (double v);
void matlab_yline      (double v);
void matlab_xline_label(double v, const char *s, int64_t n);
void matlab_yline_label(double v, const char *s, int64_t n);
void matlab_bar1   (matlab_mat *y);
void matlab_bar2   (matlab_mat *x, matlab_mat *y);
void matlab_imshow (matlab_mat *image);

/* imagesc(C) — pseudocolor display of matrix C with autoscaled data
 * range and axis tick labels (vs imshow which fills the plot area
 * without ticks). Honors the active colormap. */
void matlab_imagesc(matlab_mat *image);

/* pcolor(C) — pseudocolor cell-shaded display. Same model as imagesc
 * for our purposes (no separate vertex semantics yet). */
void matlab_pcolor (matlab_mat *image);

/* colorbar — add a vertical color-scale strip to the right of the plot
 * area. No-op if there's no image series to read the scale from. */
void matlab_colorbar(void);

/* contour(Z) — auto-pick 10 levels evenly spaced from min(Z) to max(Z)
 * and draw the level sets as line segments via marching squares.
 * Z is interpreted with rows as y, columns as x, math convention
 * (row 1 at the bottom of the plot). */
void matlab_contour(matlab_mat *Z);

/* contour(Z, V) — same, but use the explicit level vector V. */
void matlab_contour_levels(matlab_mat *Z, matlab_mat *levels);

/* contourf(Z) — filled contour: heatmap shaded by colormap with level
 * lines drawn on top. Implemented as imagesc + contour internally. */
void matlab_contourf(matlab_mat *Z);

/* quiver(x, y, u, v) — 2-D vector field. For each point (x[i], y[i])
 * draws an arrow with direction (u[i], v[i]) and length proportional
 * to the magnitude. */
void matlab_quiver(matlab_mat *x, matlab_mat *y,
                   matlab_mat *u, matlab_mat *v);

/* hold on/off — controls whether next plot replaces or overlays. */
void matlab_hold_on (void);
void matlab_hold_off(void);

/* Override the most-recently-added series' color via an RGB triplet.
 * Each component is clamped to [0, 1]. Mirrors MATLAB's
 *   plot(x, y, 'Color', [0.1 0.2 0.3])   →   matlab_set_series_color(...)
 * applied immediately after the plot() call. */
void matlab_set_series_color(double r, double g, double b);

/* RGB triplet form: read from a 3-element matlab_mat. The codegen for
 * `plot(..., 'Color', [r g b])` lowers the literal vector into a
 * matlab_mat* at IR level, so this is the form the JIT sees. */
void matlab_set_series_color_mat(matlab_mat *rgb);

/* Property/value setters — applied to the most-recently-added series
 * (i.e. the last call to plot/scatter/bar/etc.). They map MATLAB's
 *   plot(x, y, 'LineWidth', 2, 'LineStyle', '--', ...)
 * Name/Value pairs to runtime mutations. */
void matlab_set_series_linewidth (double width);
void matlab_set_series_markersize(double size);
void matlab_set_series_linestyle (const char *spec, int64_t n);
void matlab_set_series_marker    (const char *spec, int64_t n);
void matlab_set_series_displayname(const char *name, int64_t n);

/* --------------------------------------------------------------------------
 * Axes decoration
 * -------------------------------------------------------------------------- */

void matlab_xlabel(const char *s, int64_t n);
void matlab_ylabel(const char *s, int64_t n);
void matlab_title (const char *s, int64_t n);

/* xlim([lo hi]) — limits passed as a 2-element matlab_mat. */
void matlab_xlim(matlab_mat *limits);
void matlab_ylim(matlab_mat *limits);
/* zlim([lo hi]) — z-axis limits for 3-D axes (plot3/surf/mesh/...). */
void matlab_zlim(matlab_mat *limits);

/* grid on/off. */
void matlab_grid_on (void);
void matlab_grid_off(void);

/* Axis scale — log-spaced ticks (decade ticks 1, 10, 100, ...) when set
 * to log. Non-positive data is skipped on a log axis. Convenience
 * combinators (loglog/semilogx/semilogy) match MATLAB's plot family. */
void matlab_xscale_linear(void);
void matlab_xscale_log   (void);
void matlab_yscale_linear(void);
void matlab_yscale_log   (void);
void matlab_loglog       (void);  /* both axes log */
void matlab_semilogx     (void);  /* x log, y linear */
void matlab_semilogy     (void);  /* x linear, y log */

/* axis([xmin xmax ymin ymax]) — set both x and y limits from a single
 * 4-element matrix. The numeric form of MATLAB's `axis(...)`. */
void matlab_axis_lims(matlab_mat *limits);

/* axis(opt) — parses MATLAB's axis option strings. Recognised:
 *   "equal"  — 1 unit data = 1 unit screen on both axes
 *   "tight"  — drop the 5% autoscale padding
 *   "off"    — hide frame, ticks, labels (axes invisible)
 *   "on"     — show axes (the default)
 *   "square" — alias of "equal" for now
 * Numeric form (`axis([xmin xmax ymin ymax])`) is exposed via
 * matlab_xlim / matlab_ylim. */
void matlab_axis(const char *opt, int64_t n);

/* box on/off — frame visibility independent of axis ticks/labels. */
void matlab_box_on (void);
void matlab_box_off(void);

/* colormap('name') — set the active colormap for the current axes.
 * Recognised: "gray", "parula", "jet", "viridis", "hot", "cool".
 * Unknown names fall back to parula (MATLAB's default since R2014b). */
void matlab_colormap(const char *name, int64_t n);

/* legend({'a','b',...}) — labels from a cell array. */
void matlab_legend(matlab_cell *labels);

/* Array-form variant: labels[i] is a (char*, lens[i]) string, n entries.
 * This is what JIT'd MATLAB can lower a cell-of-strings into without
 * needing the full cell accessor surface, and what C-side tests use. */
void matlab_legend_strs(const char *const *labels,
                        const int64_t *lens, int64_t n);

/* --------------------------------------------------------------------------
 * Output
 * -------------------------------------------------------------------------- */

/* In-memory render of the current figure. Format chosen by call site.
 * Returned buffer.data is malloc'd; caller releases with the corresponding
 * free routine. */
matlab_plot_buffer matlab_render_png(void);
matlab_plot_buffer matlab_render_svg(void);
matlab_plot_buffer matlab_render_pdf(void);
void               matlab_plot_buffer_free(matlab_plot_buffer buf);

/* saveas / print — write current figure to file. Format is inferred from
 * the path extension (.png / .svg / .pdf). Returns 0 on success, -1 on
 * unsupported extension, -2 on I/O error. */
int matlab_savefig(const char *path, int64_t plen);

/* --------------------------------------------------------------------------
 * IDE integration (matlab_llvm_ide / Matlab_llvm_ide)
 *
 * When the env var MATLAB_LLVM_IDE_FIGURES=1 is set, the runtime streams
 * every open figure as a base64-encoded PNG over stdout, sentinel-bracketed
 * so the IDE's REPL / Run readers can intercept the block before it lands
 * in the console. The format on stdout is:
 *
 *   ___MF_FIG_BEGIN___ id=<id> w=<px> h=<px>
 *   <base64 PNG, chunked at 76 chars/line>
 *   ___MF_FIG_END___
 *
 * Each emit walks the per-thread figure registry; figures are matched by
 * `id` on the IDE side so re-emits replace the existing thumbnail in
 * place rather than appending. Trigger points:
 *   - process exit (atexit) — auto-registered on first figure when the
 *     env var is set, so compiled programs flush figures even without
 *     explicit drawnow / saveas;
 *   - matlab_drawnow() — manual flush;
 *   - in REPL mode, matlabc calls this after every input (see
 *     tools/matlabc/main.cpp:runRepl).
 * -------------------------------------------------------------------------- */

/* Render and stream every figure on the calling thread. No-op when the
 * env var isn't set. Safe to call from any of: program code (drawnow),
 * the matlabc REPL driver (post-input), atexit. */
void matlab_ide_emit_all_figures(void);

/* drawnow — flush any pending figure state to the IDE. The MATLAB
 * builtin's traditional role is "render now"; with the Cairo backend
 * there is no event loop to pump, so this is a thin wrapper around
 * matlab_ide_emit_all_figures. Always safe to call; no-op outside the
 * IDE-integration path. */
void matlab_drawnow(void);

/* --------------------------------------------------------------------------
 * Animation capture & video export (getframe / VideoWriter)
 *
 * Capture the current figure as a raster frame, then stream frames into a
 * video container. Mirrors MATLAB:
 *
 *   v = VideoWriter('out.mp4', 'MPEG-4');
 *   v.FrameRate = 30;
 *   open(v);
 *   for k = 1:N
 *       plot(...);
 *       writeVideo(v, getframe(gcf));
 *   end
 *   close(v);
 *
 * Frame structs returned by matlab_getframe are owned by a per-thread
 * registry and released by matlab_close_all / thread exit, so MATLAB
 * scripts never have to free them explicitly.
 * -------------------------------------------------------------------------- */

/* getframe / getframe(h) — render the current figure to an in-memory raster
 * and return an opaque frame handle. Returns null only on render failure. */
matlab_frame *matlab_getframe(void);

/* VideoWriter('path') / VideoWriter('path', 'profile').
 *  - matlab_videowriter_new picks a profile from the path extension
 *    (.mp4/.m4v -> MPEG-4 H.264, .avi -> Motion JPEG AVI).
 *  - matlab_videowriter_new_profile takes an explicit MATLAB profile name
 *    ("MPEG-4", "Motion JPEG AVI"); unknown names fall back to the
 *    extension default.
 * Strings are (ptr, len), not zero-terminated. Returns null on bad args. */
matlab_videowriter *matlab_videowriter_new(const char *path, int64_t plen);
matlab_videowriter *matlab_videowriter_new_profile(const char *path,
                                                   int64_t plen,
                                                   const char *profile,
                                                   int64_t prlen);

/* Property setters (v.FrameRate = fps / v.Quality = q). Must be called
 * before the first frame is written; ignored afterwards. Quality is
 * 0..100 (MATLAB convention), default 75. FrameRate default 30. */
void matlab_videowriter_set_framerate(matlab_videowriter *v, double fps);
void matlab_videowriter_set_quality(matlab_videowriter *v, double q);

/* open(v) — ready the writer for frames. The encoder itself is initialised
 * lazily from the first frame's dimensions. Returns 0 on success. */
int matlab_videowriter_open(matlab_videowriter *v);

/* writeVideo(v, frame) — append one frame. The first frame fixes the video
 * dimensions; later frames must match. Returns 0 on success, non-zero on
 * encode error or dimension mismatch. */
int matlab_videowriter_write(matlab_videowriter *v, matlab_frame *frame);

/* close(v) — flush the encoder, finalise the container, free the writer.
 * The handle is invalid after this call. Returns 0 on success.
 *
 * When the IDE env gate (MATLAB_LLVM_IDE_FIGURES=1) is set and the file
 * was written successfully, close() also emits a one-line sentinel on
 * stdout so the IDE can surface the finished video in its Plots panel:
 *
 *   ___MF_VID___ w=<px> h=<px> fps=<rate> path=<absolute path>
 *
 * `path=` is last and runs to end-of-line, so paths with spaces survive.
 * This mirrors the figure-emit protocol (matlab_ide_emit_all_figures)
 * but points at an on-disk artifact instead of streaming a bitmap. */
int matlab_videowriter_close(matlab_videowriter *v);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MATLAB_PLOT_H */
