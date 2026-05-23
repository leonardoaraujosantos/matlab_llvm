#ifndef MATLAB_PLOT_FIGURE_H
#define MATLAB_PLOT_FIGURE_H

#include "colormap.h"

#include <memory>
#include <string>
#include <vector>

namespace matlab_plot {

enum class SeriesKind {
    Line, Scatter, Bar, Image, Contour, ErrorBar, Stem, Stairs, Area,
    Line3D, Mesh, Surf, Quiver,
    /* TriMesh3D — unstructured 3D triangle mesh, for pdeplot3D /
     * STL / GLB visualisation.  Node coordinates live in x, y, z;
     * per-vertex scalar in `image`; 0-based triangle indices (3 per
     * triangle) in `mesh_tris`; per-vertex deformation in u, v, plus
     * a third axis stored in `e_pos` (the existing 3rd scalar slot). */
    TriMesh3D,
    /* TriMesh2D — unstructured 2D triangle mesh, for pdeplot of
     * Poisson-tier FEM solutions.  Node coordinates in x, y; per-
     * vertex scalar in `image`; 0-based triangle indices in
     * `mesh_tris`.  No depth sort, drawn in mesh order. */
    TriMesh2D
};

enum class Scale { Linear, Log };

enum class LineStyle { Solid, Dashed, Dotted, DashDot };

struct Series {
    SeriesKind kind = SeriesKind::Line;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;  /* 3-D series only (Line3D, Mesh, Surf) */
    /* Image series: row-major pixel data, dims in rows/cols.
     * with_axes distinguishes imshow (false — fills area, no ticks)
     * from imagesc / pcolor (true — heatmap with axis labels). */
    std::vector<double> image;
    int rows = 0;
    int cols = 0;
    bool with_axes = false;
    /* ErrorBar series: per-point negative and positive offsets. */
    std::vector<double> e_neg;
    std::vector<double> e_pos;
    /* Quiver series: per-point direction components (u, v). */
    std::vector<double> u;
    std::vector<double> v;
    /* Bar / Histogram series: half-width factor of the smallest x-gap.
     * 0.35 = standard bar (70% width); 0.49 = histogram (flush bins). */
    double bar_half_factor = 0.35;
    /* MATLAB linespec parsed: color RGBA, line style, marker.
     * color_set tracks whether an explicit color has been chosen so the
     * auto-cycle code can pick a default only when the user didn't. */
    float r = 0.f, g = 0.4470f, b = 0.7410f, a = 1.f;  // MATLAB default blue
    bool color_set = false;
    LineStyle style = LineStyle::Solid;
    /* line_off: linespec like 'o' (marker only, no connecting line). */
    bool line_off = false;
    /* MATLAB marker glyph: 0=none, 'o' '+' '*' 'x' 's' 'd' '^' as ASCII. */
    char marker = 0;
    /* Property/value overrides (MATLAB plot Name/Value pairs). */
    double line_width  = 1.5;
    double marker_size = 4.0;
    std::string display_name;  /* legend label override */
    /* yyaxis: which y-axis this series is bound to. */
    bool on_right_axis = false;
    /* TriMesh3D series: 0-based triangle indices, 3 ints per triangle. */
    std::vector<int> mesh_tris;
    /* TriMesh3D deformation: per-vertex displacement vector (uz column
     * stored in mesh_dz; ux/uy reuse u/v).  Scale factor applied
     * pre-projection. */
    std::vector<double> mesh_dz;
    double deform_scale = 1.0;
    /* TriMesh3D: a per-face value override.  If non-empty, used for
     * colormap lookup per triangle (Gouraud shading not applied).
     * Otherwise the painter uses the per-vertex `image` for Gouraud
     * shading. */
    std::vector<double> face_data;
};

struct TextAnnotation {
    double      x = 0;
    double      y = 0;
    std::string s;
};

/* xline / yline reference markers — vertical (xline) or horizontal
 * (yline) infinite lines clipped to the plot area, with optional label. */
struct RefLine {
    bool        vertical = true;
    double      value    = 0;
    std::string label;
    LineStyle   style    = LineStyle::Dashed;
    float       r = 0.f, g = 0.f, b = 0.f;
};

struct Axes {
    std::vector<Series> series;
    std::vector<TextAnnotation> annotations;
    std::vector<RefLine> reflines;
    std::string xlabel;
    std::string ylabel;
    std::string title;
    std::vector<std::string> legend;
    bool grid = false;
    bool xlim_set = false, ylim_set = false;
    double xlim_lo = 0, xlim_hi = 1;
    double ylim_lo = 0, ylim_hi = 1;
    bool hold = false;
    Colormap cmap = Colormap::Parula;
    /* Image semantics: row 1 at top → y-axis inverted compared to math
     * convention. Set automatically when an image series is added. */
    bool y_inverted = false;
    /* matlab_colorbar() adds a colorbar strip on the right of the plot
     * area when there's an image series to read the scale from. */
    bool colorbar = false;
    /* axis options: equal aspect ratio, hide axes entirely, hide just
     * the frame, suppress the autoscale y-padding. */
    bool axis_equal = false;
    bool axis_off   = false;
    bool box_off    = false;
    bool tight      = false;
    /* MATLAB auto color cycle position. Bumped on every new_series whose
     * color isn't explicitly set; the renderer reads back through cycle
     * index when laying down the default palette colors. */
    int color_cycle_idx = 0;
    /* Custom tick overrides — empty means use the auto Heckbert ticks /
     * decade ticks. xtick_labels.size() must match xticks.size() (same
     * for y); a renderer fallback to numeric formatting fills any gap. */
    std::vector<double>      xticks_v;
    std::vector<double>      yticks_v;
    std::vector<std::string> xtick_labels;
    std::vector<std::string> ytick_labels;
    /* yyaxis state: y_active_right toggles which axis subsequent plot
     * calls write to; y_right_used is set the first time the right
     * axis is touched, to know whether to render its frame and ticks. */
    bool   y_active_right = false;
    bool   y_right_used   = false;
    bool   ylim_right_set = false;
    double ylim_right_lo  = 0;
    double ylim_right_hi  = 1;
    std::string ylabel_right;
    Scale xscale = Scale::Linear;
    Scale yscale = Scale::Linear;
    /* 3-D state — set automatically by plot3 / mesh / surf. The view
     * angles match MATLAB's default (az=-37.5°, el=30°). */
    bool   is_3d  = false;
    double view_az = -37.5;
    double view_el =  30.0;
    std::string zlabel;
};

struct Figure {
    int id = 0;
    int width_px  = 800;
    int height_px = 600;
    /* Subplot grid: rows × cols cells, indexed row-major from top-left.
     * Default 1×1 with one Axes. subplot(m, n, i) reshapes the grid. */
    int rows = 1;
    int cols = 1;
    std::vector<std::unique_ptr<Axes>> axes_list;
    int active = 0;  // 0-based index into axes_list

    /* Convenience for call sites that previously used `fig.axes`. */
    Axes &axes() { return *axes_list[active]; }
    const Axes &axes() const { return *axes_list[active]; }
};

/* Per-thread figure registry. */
Figure *current_figure();
Figure *figure_new();
Figure *figure_by_id(int id);
void    close_all();

/* Snapshot of every figure on the calling thread, in creation order.
 * Used by the IDE-emit path to walk the figure list without exposing
 * the thread-local owning vector. Pointers are valid until the next
 * call into the figure registry on this thread. */
std::vector<Figure *> figures_snapshot();

/* Active axes accessor — equivalent to current_figure()->axes(). */
Axes   &current_axes();

/* subplot(rows, cols, idx_1based). Reshapes the grid if (rows, cols)
 * differ from the current figure's grid; then sets `active` to idx-1.
 * If idx is out of range, behaves as a no-op. */
void    subplot(int rows, int cols, int idx_1based);

}  // namespace matlab_plot

#endif
