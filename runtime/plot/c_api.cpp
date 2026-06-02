/* C ABI shim — connects matlab_plot.h symbols to the C++ figure / Cairo
 * implementation. Every public symbol is declared extern "C" so JIT'd
 * MATLAB code can resolve it via DynamicLibrarySearchGenerator alongside
 * the existing matlab_* runtime symbols. */

#include "../matlab_plot.h"
#include "../runtime_internal.h"

#include "cairo_render.h"
#include "colormap.h"
#include "contour.h"
#include "figure.h"

#include <string_view>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using matlab_plot::Axes;
using matlab_plot::Figure;
using matlab_plot::LineStyle;
using matlab_plot::Series;
using matlab_plot::SeriesKind;

namespace {

/* MATLAB's default color order (parula / 'lines' colormap, R2014b+). */
static const float kDefaultColors[7][3] = {
    {0.0000f, 0.4470f, 0.7410f},
    {0.8500f, 0.3250f, 0.0980f},
    {0.9290f, 0.6940f, 0.1250f},
    {0.4940f, 0.1840f, 0.5560f},
    {0.4660f, 0.6740f, 0.1880f},
    {0.3010f, 0.7450f, 0.9330f},
    {0.6350f, 0.0780f, 0.1840f},
};

void apply_cycle_color(Axes &ax, Series &s) {
    int i = ax.color_cycle_idx % 7;
    s.r = kDefaultColors[i][0];
    s.g = kDefaultColors[i][1];
    s.b = kDefaultColors[i][2];
    ax.color_cycle_idx++;
}

/* Replace-or-append based on the `hold` flag. Clears text annotations
 * too — MATLAB's `plot` without `hold on` resets the axes. With yyaxis
 * active, a non-hold plot replaces only the *same-side* series so the
 * other axis's data is preserved (matches MATLAB's documented yyaxis
 * semantics). The color cycle restarts only on a full clear. */
Series &new_series(Axes &ax, SeriesKind kind) {
    if (!ax.hold) {
        if (ax.y_right_used) {
            const bool side = ax.y_active_right;
            ax.series.erase(
                std::remove_if(ax.series.begin(), ax.series.end(),
                    [side](const Series &s) { return s.on_right_axis == side; }),
                ax.series.end());
        } else {
            ax.series.clear();
            ax.color_cycle_idx = 0;
        }
        ax.annotations.clear();
    }
    ax.series.emplace_back();
    ax.series.back().kind = kind;
    ax.series.back().on_right_axis = ax.y_active_right;
    if (ax.y_active_right) ax.y_right_used = true;
    /* Pre-assign the cycle color; explicit linespec / set_series_color /
     * Name/Value 'Color' overrides flip color_set true and replace RGB.
     * Image/Bar series consume a cycle slot too in MATLAB; matching that. */
    apply_cycle_color(ax, ax.series.back());
    return ax.series.back();
}

void copy_mat_to_vec(matlab_mat *m, std::vector<double> &out) {
    if (!m || !m->data) { out.clear(); return; }
    int64_t n = m->rows * m->cols;
    out.assign(m->data, m->data + n);
}

/* MATLAB's plot(y) with vector y treats y vs 1..numel(y). */
void fill_index(std::vector<double> &x, size_t n) {
    x.resize(n);
    for (size_t i = 0; i < n; ++i) x[i] = static_cast<double>(i + 1);
}

/* Minimal MATLAB linespec parser: color letter (rgbcmykw), line style
 * (-, --, :, -.), marker (o, +, *, x, s, d, ^). Anything unrecognized
 * is ignored. */
void parse_linespec(const char *fmt, int64_t flen, Series &s) {
    if (!fmt || flen <= 0) return;
    std::string spec(fmt, fmt + flen);
    for (char c : spec) {
        bool matched = true;
        switch (c) {
            case 'r': s.r = 0.85f; s.g = 0.33f; s.b = 0.10f; break;
            case 'g': s.r = 0.47f; s.g = 0.67f; s.b = 0.19f; break;
            case 'b': s.r = 0.00f; s.g = 0.45f; s.b = 0.74f; break;
            case 'c': s.r = 0.30f; s.g = 0.75f; s.b = 0.93f; break;
            case 'm': s.r = 0.49f; s.g = 0.18f; s.b = 0.56f; break;
            case 'y': s.r = 0.93f; s.g = 0.69f; s.b = 0.13f; break;
            case 'k': s.r = 0.00f; s.g = 0.00f; s.b = 0.00f; break;
            case 'w': s.r = 1.00f; s.g = 1.00f; s.b = 1.00f; break;
            default:  matched = false;
        }
        if (matched) { s.color_set = true; break; }
    }
    /* Order matters: '--' must be checked before single '-', and '-.' before '-'. */
    if (spec.find("--") != std::string::npos) s.style = LineStyle::Dashed;
    else if (spec.find("-.") != std::string::npos) s.style = LineStyle::DashDot;
    else if (spec.find(':')  != std::string::npos) s.style = LineStyle::Dotted;
    /* '-' alone leaves Solid (the default). */
    /* Line-style detection: explicit '-' (solid, default), '--' (dashed,
     * already), ':' / '-.' not yet rendered distinctly. A spec with a
     * marker but no '-' / '--' / ':' / '-.' means marker-only. */
    bool has_line = (spec.find('-') != std::string::npos) ||
                    (spec.find(':') != std::string::npos);
    for (char c : spec) {
        if (c == 'o' || c == '+' || c == '*' || c == 'x' ||
            c == 's' || c == 'd' || c == '^') {
            s.marker = c;
            if (!has_line) s.line_off = true;
            break;
        }
    }
}

/* Auto-pick 10 levels evenly spaced strictly between min(Z) and max(Z),
 * skipping the endpoints — they collapse to no segments under marching
 * squares anyway. */
std::vector<double> auto_levels(const double *Z, int64_t n) {
    double lo =  std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        if (Z[i] < lo) lo = Z[i];
        if (Z[i] > hi) hi = Z[i];
    }
    std::vector<double> out;
    if (!std::isfinite(lo) || hi <= lo) return out;
    const int N = 10;
    for (int k = 1; k <= N; ++k)
        out.push_back(lo + (hi - lo) * k / (N + 1));
    return out;
}

void contour_impl(matlab_mat *Z, const std::vector<double> &levels) {
    if (!Z || Z->rows < 2 || Z->cols < 2 || levels.empty()) return;
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Contour);
    matlab_plot::marching_squares(Z->data,
                                  static_cast<int>(Z->rows),
                                  static_cast<int>(Z->cols),
                                  levels, s.x, s.y);
    s.r = 0.f; s.g = 0.f; s.b = 0.f; s.a = 1.f;
}

}  // namespace

extern "C" {

/* -------- Figure lifecycle -------- */

matlab_figure *matlab_figure_new(void) {
    return reinterpret_cast<matlab_figure *>(matlab_plot::figure_new());
}

matlab_figure *matlab_figure_by_id(double id) {
    return reinterpret_cast<matlab_figure *>(
        matlab_plot::figure_by_id(static_cast<int>(id)));
}

matlab_figure *matlab_gcf(void) {
    return reinterpret_cast<matlab_figure *>(matlab_plot::current_figure());
}

void matlab_close_all(void) { matlab_plot::close_all(); }

void matlab_subplot(double rows, double cols, double i) {
    matlab_plot::subplot(static_cast<int>(rows),
                         static_cast<int>(cols),
                         static_cast<int>(i));
}

/* -------- Plot primitives -------- */

void matlab_plot1(matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Line);
    copy_mat_to_vec(y, s.y);
    fill_index(s.x, s.y.size());
}

void matlab_plot2(matlab_mat *x, matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Line);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
}

void matlab_plot_fmt(matlab_mat *x, matlab_mat *y,
                     const char *fmt, int64_t flen) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Line);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
    parse_linespec(fmt, flen, s);
}

void matlab_plot3(matlab_mat *x, matlab_mat *y, matlab_mat *z) {
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Line3D);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
    copy_mat_to_vec(z, s.z);
    ax.is_3d = true;
}

void matlab_plot3_fmt(matlab_mat *x, matlab_mat *y, matlab_mat *z,
                      const char *fmt, int64_t flen) {
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Line3D);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
    copy_mat_to_vec(z, s.z);
    parse_linespec(fmt, flen, s);
    ax.is_3d = true;
}

void matlab_view(double az, double el) {
    auto &ax = matlab_plot::current_axes();
    ax.view_az = az;
    ax.view_el = el;
}

void matlab_zlabel(const char *s, int64_t n) {
    if (!s || n <= 0) return;
    matlab_plot::current_axes().zlabel.assign(s, s + n);
}

namespace {
/* Internal helpers for mesh/surf data ingest. Z is a row-major rows×cols
 * matrix; X and Y, when supplied, are the same shape. When omitted we
 * auto-generate X = 1..cols (replicated), Y = 1..rows. */
void ingest_grid_z(matlab_mat *Z, Series &s) {
    s.rows = static_cast<int>(Z->rows);
    s.cols = static_cast<int>(Z->cols);
    int64_t n = Z->rows * Z->cols;
    s.z.assign(Z->data, Z->data + n);
    s.x.resize(n);
    s.y.resize(n);
    for (int r = 0; r < s.rows; ++r) {
        for (int c = 0; c < s.cols; ++c) {
            s.x[r * s.cols + c] = c + 1;  /* 1-based column index */
            s.y[r * s.cols + c] = r + 1;
        }
    }
}

void ingest_grid_xyz(matlab_mat *X, matlab_mat *Y, matlab_mat *Z, Series &s) {
    s.rows = static_cast<int>(Z->rows);
    s.cols = static_cast<int>(Z->cols);
    int64_t n = Z->rows * Z->cols;
    s.x.assign(X->data, X->data + n);
    s.y.assign(Y->data, Y->data + n);
    s.z.assign(Z->data, Z->data + n);
}
}  // namespace

/* --- pdeplot 2-D unstructured-mesh painter ------------------------ */

void matlab_pdeplot(matlab_mat *nodes, matlab_mat *triangles,
                    matlab_mat *nodal_data) {
    if (!nodes || nodes->cols < 2 || nodes->rows < 3) return;
    if (!triangles || triangles->rows < 1) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::TriMesh2D);
    s.color_set = true;
    int64_t Nn = nodes->rows;
    int64_t ncols = nodes->cols;  /* 2 or 3 (z ignored) */
    s.x.resize((size_t)Nn);
    s.y.resize((size_t)Nn);
    for (int64_t i = 0; i < Nn; ++i) {
        s.x[(size_t)i] = nodes->data[i * ncols + 0];
        s.y[(size_t)i] = nodes->data[i * ncols + 1];
    }
    int64_t Nt = triangles->rows;
    int idx_off = (triangles->cols == 4) ? 1 : 0;
    s.mesh_tris.resize((size_t)(Nt * 3));
    for (int64_t t = 0; t < Nt; ++t) {
        for (int k = 0; k < 3; ++k) {
            int64_t v = (int64_t)triangles->data[t * triangles->cols
                                                  + idx_off + k] - 1;
            if (v < 0) v = 0;
            if (v >= Nn) v = Nn - 1;
            s.mesh_tris[(size_t)(t * 3 + k)] = (int)v;
        }
    }
    if (nodal_data) {
        int64_t nd = nodal_data->rows * nodal_data->cols;
        if (nd == Nn) {
            s.image.resize((size_t)Nn);
            for (int64_t i = 0; i < Nn; ++i) s.image[(size_t)i] = nodal_data->data[i];
        } else if (nd == Nt) {
            s.face_data.resize((size_t)Nt);
            for (int64_t i = 0; i < Nt; ++i) s.face_data[(size_t)i] = nodal_data->data[i];
        }
    }
    ax.colorbar = true;
    ax.axis_equal = true;  /* Honest mesh aspect ratio */
}

/* --- pdeplot3D unstructured-mesh painter -------------------------- *
 *
 * Stashes the (nodes, triangles, nodal_data) triple in the current
 * axes as a new TriMesh3D series.  Deformation + scale are picked up
 * from per-thread sticky state.
 */
namespace {
struct Pdeplot3dState {
    matlab_mat *deformation = nullptr;  /* Nn × 3 displacement */
    double scale            = 1.0;
};
static thread_local Pdeplot3dState g_pdeplot3d;
}  /* anonymous namespace */

void matlab_pdeplot3d_deformation(matlab_mat *disp) {
    g_pdeplot3d.deformation = disp;
}
void matlab_pdeplot3d_deform_scale(double s) {
    g_pdeplot3d.scale = s;
}

void matlab_pdeplot3d(matlab_mat *nodes, matlab_mat *triangles,
                       matlab_mat *nodal_data) {
    if (!nodes || nodes->cols != 3 || nodes->rows < 3) return;
    if (!triangles || triangles->rows < 1) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::TriMesh3D);
    s.color_set = true;  /* TriMesh3D never auto-cycles a colour */
    int64_t Nn = nodes->rows;
    s.x.resize((size_t)Nn);
    s.y.resize((size_t)Nn);
    s.z.resize((size_t)Nn);
    for (int64_t i = 0; i < Nn; ++i) {
        s.x[(size_t)i] = nodes->data[i * 3 + 0];
        s.y[(size_t)i] = nodes->data[i * 3 + 1];
        s.z[(size_t)i] = nodes->data[i * 3 + 2];
    }
    /* Triangle indices: 1-based.  Source layout is either Nt × 3
     * (n1, n2, n3) or Nt × 4 ([face_id, n1, n2, n3] from STL/GLB
     * loaders + multicuboid faces). */
    int64_t Nt = triangles->rows;
    int idx_off = (triangles->cols == 4) ? 1 : 0;
    s.mesh_tris.resize((size_t)(Nt * 3));
    for (int64_t t = 0; t < Nt; ++t) {
        for (int k = 0; k < 3; ++k) {
            int64_t v = (int64_t)triangles->data[t * triangles->cols
                                                  + idx_off + k] - 1;
            if (v < 0) v = 0;
            if (v >= Nn) v = Nn - 1;
            s.mesh_tris[(size_t)(t * 3 + k)] = (int)v;
        }
    }
    /* Nodal data: per-vertex if matches Nn, per-face if matches Nt. */
    if (nodal_data) {
        int64_t nd = nodal_data->rows * nodal_data->cols;
        if (nd == Nn) {
            s.image.resize((size_t)Nn);
            for (int64_t i = 0; i < Nn; ++i) s.image[(size_t)i] = nodal_data->data[i];
        } else if (nd == Nt) {
            s.face_data.resize((size_t)Nt);
            for (int64_t i = 0; i < Nt; ++i) s.face_data[(size_t)i] = nodal_data->data[i];
        }
    }
    /* Deformation (sticky state). */
    if (g_pdeplot3d.deformation &&
        g_pdeplot3d.deformation->rows == Nn &&
        g_pdeplot3d.deformation->cols == 3) {
        s.u.resize((size_t)Nn);
        s.v.resize((size_t)Nn);
        s.mesh_dz.resize((size_t)Nn);
        for (int64_t i = 0; i < Nn; ++i) {
            s.u[(size_t)i]       = g_pdeplot3d.deformation->data[i * 3 + 0];
            s.v[(size_t)i]       = g_pdeplot3d.deformation->data[i * 3 + 1];
            s.mesh_dz[(size_t)i] = g_pdeplot3d.deformation->data[i * 3 + 2];
        }
        s.deform_scale = g_pdeplot3d.scale;
    }
    ax.is_3d = true;
    ax.colorbar = true;
}

/* --- model/mesh-struct plot forms (issue #28) -------------------------
 * The high-level `pdeplot(model, XYData=..)` / `pdeplot3D(mesh, ColorMapData=..)`
 * calls pass a geometry/result struct rather than raw node/element
 * matrices.  These wrappers pull the Nodes + Triangles/Faces out of the
 * struct and forward to the painters above.  `colordata` is optional and
 * passed through only when it looks like a real matrix (a scalar / null
 * arrives when the field read defaulted to f64 — render uncoloured then).
 */
extern "C" {
matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name, int64_t len);
}

namespace {
/* A struct field read can defensively return a 1×1 / empty box; only
 * treat a genuinely vector-sized matrix as colour data. */
matlab_mat *plot_colordata_or_null(matlab_mat *m) {
    if (!m) return nullptr;
    if (m->rows * m->cols <= 1) return nullptr;
    return m;
}
/* `R.Mesh` may itself be the mesh struct, or the arg may already be a
 * mesh struct — return whichever carries Nodes. */
matlab_struct *plot_mesh_of(matlab_struct *s) {
    if (!s) return nullptr;
    if (matlab_struct_get_mat(s, "Nodes", 5)->rows > 0) return s;
    matlab_struct *m = (matlab_struct *)matlab_struct_get_mat(s, "Mesh", 4);
    if (m && matlab_struct_get_mat(m, "Nodes", 5)->rows > 0) return m;
    return s;
}
}  /* namespace */

void matlab_pdeplot_struct(matlab_struct *model, matlab_mat *colordata) {
    matlab_struct *mesh = plot_mesh_of(model);
    if (!mesh) return;
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *tris  = matlab_struct_get_mat(mesh, "Triangles", 9);
    matlab_pdeplot(nodes, tris, plot_colordata_or_null(colordata));
}

void matlab_pdeplot3d_struct(matlab_struct *mesharg, matlab_mat *colordata) {
    matlab_struct *mesh = plot_mesh_of(mesharg);
    if (!mesh) return;
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *faces = matlab_struct_get_mat(mesh, "Faces", 5);
    if (!faces || faces->rows == 0)
        faces = matlab_struct_get_mat(mesh, "Tets", 4);  /* fall back */
    matlab_pdeplot3d(nodes, faces, plot_colordata_or_null(colordata));
}

void matlab_surf1(matlab_mat *Z) {
    if (!Z || Z->rows < 2 || Z->cols < 2) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Surf);
    ingest_grid_z(Z, s);
    ax.is_3d = true;
}

void matlab_surf3(matlab_mat *X, matlab_mat *Y, matlab_mat *Z) {
    if (!X || !Y || !Z || Z->rows < 2 || Z->cols < 2) return;
    if (X->rows != Z->rows || X->cols != Z->cols) return;
    if (Y->rows != Z->rows || Y->cols != Z->cols) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Surf);
    ingest_grid_xyz(X, Y, Z, s);
    ax.is_3d = true;
}

void matlab_mesh1(matlab_mat *Z) {
    if (!Z || Z->rows < 2 || Z->cols < 2) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Mesh);
    ingest_grid_z(Z, s);
    ax.is_3d = true;
}

void matlab_mesh3(matlab_mat *X, matlab_mat *Y, matlab_mat *Z) {
    if (!X || !Y || !Z || Z->rows < 2 || Z->cols < 2) return;
    if (X->rows != Z->rows || X->cols != Z->cols) return;
    if (Y->rows != Z->rows || Y->cols != Z->cols) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Mesh);
    ingest_grid_xyz(X, Y, Z, s);
    ax.is_3d = true;
}

/* surfc / meshc — Tier-3 plotting roadmap entry (docs/plotting.md §3).
 * Combo of a surface (or mesh) plus a contour drawn at the base
 * plane. Compose the existing surf/mesh + contour painters on the
 * same axes; the painter side already supports overlay because every
 * series shares the same axes object. */
void matlab_surfc1(matlab_mat *Z) {
    matlab_surf1(Z);
    matlab_contour(Z);
}
void matlab_surfc3(matlab_mat *X, matlab_mat *Y, matlab_mat *Z) {
    matlab_surf3(X, Y, Z);
    /* Use Z alone for the contour map — the X/Y axes are already
     * established by the surf call above. */
    matlab_contour(Z);
}
void matlab_meshc1(matlab_mat *Z) {
    matlab_mesh1(Z);
    matlab_contour(Z);
}
void matlab_meshc3(matlab_mat *X, matlab_mat *Y, matlab_mat *Z) {
    matlab_mesh3(X, Y, Z);
    matlab_contour(Z);
}

void matlab_scatter(matlab_mat *x, matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Scatter);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
}

void matlab_errorbar(matlab_mat *x, matlab_mat *y, matlab_mat *e) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::ErrorBar);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
    copy_mat_to_vec(e, s.e_pos);
    s.e_neg = s.e_pos;  /* symmetric */
}

void matlab_errorbar2(matlab_mat *x, matlab_mat *y,
                      matlab_mat *neg, matlab_mat *pos) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::ErrorBar);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
    copy_mat_to_vec(neg, s.e_neg);
    copy_mat_to_vec(pos, s.e_pos);
}

void matlab_stem1(matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Stem);
    copy_mat_to_vec(y, s.y);
    fill_index(s.x, s.y.size());
}
void matlab_stem2(matlab_mat *x, matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Stem);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
}
void matlab_stairs1(matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Stairs);
    copy_mat_to_vec(y, s.y);
    fill_index(s.x, s.y.size());
}
void matlab_stairs2(matlab_mat *x, matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Stairs);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
}

void matlab_histogram(matlab_mat *data, double nbins_d) {
    if (!data || data->rows * data->cols == 0) return;
    int64_t n = data->rows * data->cols;
    int nbins = static_cast<int>(nbins_d);
    if (nbins <= 0) {
        nbins = std::max(1, static_cast<int>(std::round(std::sqrt(static_cast<double>(n)))));
    }
    /* Bin range over finite samples. */
    double lo =  std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        double v = data->data[i];
        if (std::isfinite(v)) {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }
    if (!std::isfinite(lo) || hi <= lo) { lo = 0; hi = 1; }
    double width = (hi - lo) / nbins;
    std::vector<int> counts(nbins, 0);
    for (int64_t i = 0; i < n; ++i) {
        double v = data->data[i];
        if (!std::isfinite(v)) continue;
        int b = static_cast<int>((v - lo) / width);
        if (b < 0) b = 0; else if (b >= nbins) b = nbins - 1;
        counts[b]++;
    }
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Bar);
    s.x.resize(nbins);
    s.y.resize(nbins);
    for (int b = 0; b < nbins; ++b) {
        s.x[b] = lo + (b + 0.5) * width;       /* bin center */
        s.y[b] = static_cast<double>(counts[b]);
    }
    s.bar_half_factor = 0.49;  /* flush bars */
}

void matlab_area1(matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Area);
    copy_mat_to_vec(y, s.y);
    fill_index(s.x, s.y.size());
}
void matlab_area2(matlab_mat *x, matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Area);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
}

void matlab_text(double x, double y, const char *s, int64_t n) {
    if (!s || n <= 0) return;
    auto &ax = matlab_plot::current_axes();
    matlab_plot::TextAnnotation a;
    a.x = x; a.y = y;
    a.s.assign(s, s + n);
    ax.annotations.push_back(std::move(a));
}

namespace {
void add_refline(double v, bool vertical, const char *label, int64_t llen) {
    auto &ax = matlab_plot::current_axes();
    matlab_plot::RefLine R;
    R.vertical = vertical;
    R.value = v;
    if (label && llen > 0) R.label.assign(label, label + llen);
    ax.reflines.push_back(std::move(R));
}
}  // namespace

void matlab_xticks(matlab_mat *v) {
    auto &ax = matlab_plot::current_axes();
    if (!v) { ax.xticks_v.clear(); return; }
    int64_t n = v->rows * v->cols;
    ax.xticks_v.assign(v->data, v->data + n);
}
void matlab_yticks(matlab_mat *v) {
    auto &ax = matlab_plot::current_axes();
    if (!v) { ax.yticks_v.clear(); return; }
    int64_t n = v->rows * v->cols;
    ax.yticks_v.assign(v->data, v->data + n);
}

namespace {
void store_labels(std::vector<std::string> &dst,
                  const char *const *labels,
                  const int64_t *lens, int64_t n) {
    dst.clear();
    dst.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        if (!labels[i] || lens[i] <= 0) { dst.emplace_back(); continue; }
        dst.emplace_back(labels[i], labels[i] + lens[i]);
    }
}
}  // namespace

void matlab_xticklabels_strs(const char *const *labels,
                             const int64_t *lens, int64_t n) {
    store_labels(matlab_plot::current_axes().xtick_labels, labels, lens, n);
}
void matlab_yticklabels_strs(const char *const *labels,
                             const int64_t *lens, int64_t n) {
    store_labels(matlab_plot::current_axes().ytick_labels, labels, lens, n);
}

void matlab_xline(double v) { add_refline(v, true,  nullptr, 0); }
void matlab_yline(double v) { add_refline(v, false, nullptr, 0); }
void matlab_xline_label(double v, const char *s, int64_t n) { add_refline(v, true,  s, n); }
void matlab_yline_label(double v, const char *s, int64_t n) { add_refline(v, false, s, n); }

void matlab_bar1(matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Bar);
    copy_mat_to_vec(y, s.y);
    fill_index(s.x, s.y.size());
}

void matlab_bar2(matlab_mat *x, matlab_mat *y) {
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Bar);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
}

/* Internal: shared image-ingest path. with_axes distinguishes the three
 * MATLAB calls — imshow (fills area, no ticks) vs imagesc / pcolor
 * (heatmap with axis labels). All three set y_inverted on the axes so
 * row 1 of the image is at the top. */
static void ingest_image(matlab_mat *image, bool with_axes) {
    if (!image) return;
    auto &ax = matlab_plot::current_axes();
    Series &s = new_series(ax, SeriesKind::Image);
    s.rows = static_cast<int>(image->rows);
    s.cols = static_cast<int>(image->cols);
    s.with_axes = with_axes;
    int64_t n = image->rows * image->cols;
    s.image.assign(image->data, image->data + n);
    ax.y_inverted = true;
}

void matlab_imshow (matlab_mat *image) { ingest_image(image, /*with_axes=*/false); }
void matlab_imagesc(matlab_mat *image) { ingest_image(image, /*with_axes=*/true);  }
void matlab_pcolor (matlab_mat *image) { ingest_image(image, /*with_axes=*/true);  }

void matlab_colorbar(void) {
    matlab_plot::current_axes().colorbar = true;
}

void matlab_contour(matlab_mat *Z) {
    if (!Z) return;
    contour_impl(Z, auto_levels(Z->data, Z->rows * Z->cols));
}

void matlab_contour_levels(matlab_mat *Z, matlab_mat *lv) {
    if (!Z || !lv) return;
    int64_t n = lv->rows * lv->cols;
    std::vector<double> levels(lv->data, lv->data + n);
    contour_impl(Z, levels);
}

void matlab_contourf(matlab_mat *Z) {
    /* Pragmatic implementation: heatmap (imagesc) for the filled bands +
     * contour() lines on top. Visually equivalent to MATLAB's contourf
     * for smooth fields without the stepped-band look the band-fill
     * algorithm would give. */
    if (!Z || Z->rows < 2 || Z->cols < 2) return;
    matlab_plot::current_axes().hold = true;  /* both layers on same axes */
    matlab_imagesc(Z);
    matlab_contour(Z);
    matlab_plot::current_axes().hold = false;
}

void matlab_quiver(matlab_mat *x, matlab_mat *y,
                   matlab_mat *u, matlab_mat *v) {
    if (!x || !y || !u || !v) return;
    Series &s = new_series(matlab_plot::current_axes(), SeriesKind::Quiver);
    copy_mat_to_vec(x, s.x);
    copy_mat_to_vec(y, s.y);
    copy_mat_to_vec(u, s.u);
    copy_mat_to_vec(v, s.v);
}

void matlab_hold_on (void) { matlab_plot::current_axes().hold = true;  }
void matlab_hold_off(void) { matlab_plot::current_axes().hold = false; }

void matlab_set_series_color(double r, double g, double b) {
    auto &ax = matlab_plot::current_axes();
    if (ax.series.empty()) return;
    auto clamp = [](double v) {
        return v < 0 ? 0.f : v > 1 ? 1.f : static_cast<float>(v);
    };
    Series &s = ax.series.back();
    s.r = clamp(r); s.g = clamp(g); s.b = clamp(b);
    s.color_set = true;
}

void matlab_set_series_color_mat(matlab_mat *rgb) {
    if (!rgb || rgb->rows * rgb->cols < 3) return;
    matlab_set_series_color(rgb->data[0], rgb->data[1], rgb->data[2]);
}

void matlab_set_series_linewidth(double width) {
    auto &ax = matlab_plot::current_axes();
    if (ax.series.empty()) return;
    if (width > 0) ax.series.back().line_width = width;
}

void matlab_set_series_markersize(double size) {
    auto &ax = matlab_plot::current_axes();
    if (ax.series.empty()) return;
    if (size > 0) ax.series.back().marker_size = size;
}

void matlab_set_series_linestyle(const char *spec, int64_t n) {
    auto &ax = matlab_plot::current_axes();
    if (!spec || n <= 0 || ax.series.empty()) return;
    std::string_view sv(spec, n);
    Series &s = ax.series.back();
    if      (sv == "-")     s.style = LineStyle::Solid;
    else if (sv == "--")    s.style = LineStyle::Dashed;
    else if (sv == ":")     s.style = LineStyle::Dotted;
    else if (sv == "-.")    s.style = LineStyle::DashDot;
    else if (sv == "none")  s.line_off = true;
}

void matlab_set_series_marker(const char *spec, int64_t n) {
    auto &ax = matlab_plot::current_axes();
    if (!spec || n <= 0 || ax.series.empty()) return;
    Series &s = ax.series.back();
    if (n == 1) {
        char c = spec[0];
        if (c == 'o' || c == '+' || c == '*' || c == 'x' ||
            c == 's' || c == 'd' || c == '^') {
            s.marker = c;
        } else if (c == '.') {
            s.marker = 'o';  /* MATLAB '.' is a small filled dot — closest
                              * to our 'o' painter. */
        }
        return;
    }
    if (n == 4 && std::string_view(spec, 4) == "none") s.marker = 0;
}

void matlab_set_series_displayname(const char *name, int64_t n) {
    auto &ax = matlab_plot::current_axes();
    if (!name || n <= 0 || ax.series.empty()) return;
    ax.series.back().display_name.assign(name, name + n);
}

/* -------- Decoration -------- */

void matlab_xlabel(const char *s, int64_t n) {
    matlab_plot::current_axes().xlabel.assign(s, s + n);
}
void matlab_ylabel(const char *s, int64_t n) {
    auto &ax = matlab_plot::current_axes();
    /* Route to whichever y-axis is active; matches MATLAB's yyaxis flow. */
    if (ax.y_active_right) ax.ylabel_right.assign(s, s + n);
    else                   ax.ylabel.assign(s, s + n);
}
void matlab_title(const char *s, int64_t n) {
    matlab_plot::current_axes().title.assign(s, s + n);
}

void matlab_xlim(matlab_mat *limits) {
    if (!limits || limits->rows * limits->cols < 2) return;
    auto &ax = matlab_plot::current_axes();
    ax.xlim_lo = limits->data[0];
    ax.xlim_hi = limits->data[1];
    ax.xlim_set = true;
}
void matlab_ylim(matlab_mat *limits) {
    if (!limits || limits->rows * limits->cols < 2) return;
    auto &ax = matlab_plot::current_axes();
    if (ax.y_active_right) {
        ax.ylim_right_lo  = limits->data[0];
        ax.ylim_right_hi  = limits->data[1];
        ax.ylim_right_set = true;
    } else {
        ax.ylim_lo  = limits->data[0];
        ax.ylim_hi  = limits->data[1];
        ax.ylim_set = true;
    }
}

void matlab_yyaxis(const char *side, int64_t n) {
    if (!side || n <= 0) return;
    auto &ax = matlab_plot::current_axes();
    std::string_view sv(side, n);
    ax.y_active_right = (sv == "right");
    if (ax.y_active_right) ax.y_right_used = true;
}

void matlab_grid_on (void) { matlab_plot::current_axes().grid = true;  }
void matlab_grid_off(void) { matlab_plot::current_axes().grid = false; }

void matlab_xscale_linear(void) { matlab_plot::current_axes().xscale = matlab_plot::Scale::Linear; }
void matlab_xscale_log   (void) { matlab_plot::current_axes().xscale = matlab_plot::Scale::Log;    }
void matlab_yscale_linear(void) { matlab_plot::current_axes().yscale = matlab_plot::Scale::Linear; }
void matlab_yscale_log   (void) { matlab_plot::current_axes().yscale = matlab_plot::Scale::Log;    }
void matlab_loglog  (void) { matlab_xscale_log();    matlab_yscale_log();    }
void matlab_semilogx(void) { matlab_xscale_log();    matlab_yscale_linear(); }
void matlab_semilogy(void) { matlab_xscale_linear(); matlab_yscale_log();    }

void matlab_axis_lims(matlab_mat *limits) {
    if (!limits || limits->rows * limits->cols < 4) return;
    auto &ax = matlab_plot::current_axes();
    ax.xlim_lo = limits->data[0];
    ax.xlim_hi = limits->data[1];
    ax.ylim_lo = limits->data[2];
    ax.ylim_hi = limits->data[3];
    ax.xlim_set = ax.ylim_set = true;
}

void matlab_axis(const char *opt, int64_t n) {
    if (!opt || n <= 0) return;
    std::string_view s(opt, n);
    auto &ax = matlab_plot::current_axes();
    if      (s == "equal" || s == "square") { ax.axis_equal = true;                   }
    else if (s == "tight")                  { ax.tight      = true;                   }
    else if (s == "off")                    { ax.axis_off   = true;                   }
    else if (s == "on")                     { ax.axis_off   = false; ax.box_off = false; }
    /* unknown options are ignored — MATLAB warns but continues. */
}

void matlab_box_on (void) { matlab_plot::current_axes().box_off = false; }
void matlab_box_off(void) { matlab_plot::current_axes().box_off = true;  }

void matlab_colormap(const char *name, int64_t n) {
    if (!name || n <= 0) return;
    matlab_plot::current_axes().cmap =
        matlab_plot::cmap_from_name(std::string_view(name, n));
}

/* Cell-form legend: legend({'a', 'b', ...}). Walk the matlab_cell with
 * the existing matlab_cell_get_mat / matlab_cell_size_dim accessors,
 * decode each char-vector entry into a C++ string, populate ax.legend.
 * Char vectors are row-major doubles holding ASCII codes (MATLAB's
 * conventional char representation). */
extern "C" double      matlab_cell_size_dim(matlab_cell *c, double dim);
extern "C" matlab_mat *matlab_cell_get_mat (matlab_cell *c, double i1);

void matlab_legend(matlab_cell *labels) {
    if (!labels) return;
    auto &ax = matlab_plot::current_axes();
    /* Cell may be 1-D (rows=1, cols=N or vice versa); use total numel. */
    int64_t rows = static_cast<int64_t>(matlab_cell_size_dim(labels, 1));
    int64_t cols = static_cast<int64_t>(matlab_cell_size_dim(labels, 2));
    int64_t n    = rows * cols;
    if (n <= 0) return;
    ax.legend.clear();
    ax.legend.reserve(static_cast<size_t>(n));
    for (int64_t i = 1; i <= n; ++i) {
        matlab_mat *m = matlab_cell_get_mat(labels, static_cast<double>(i));
        if (!m || m->rows * m->cols <= 0) {
            ax.legend.emplace_back();
            continue;
        }
        std::string s;
        int64_t k = m->rows * m->cols;
        s.reserve(static_cast<size_t>(k));
        for (int64_t j = 0; j < k; ++j) {
            int c = static_cast<int>(m->data[j]);
            if (c >= 0 && c < 0x80) s.push_back(static_cast<char>(c));
        }
        ax.legend.push_back(std::move(s));
    }
}

void matlab_legend_strs(const char *const *labels,
                        const int64_t *lens, int64_t n) {
    auto &ax = matlab_plot::current_axes();
    ax.legend.clear();
    ax.legend.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        if (!labels[i] || lens[i] <= 0) { ax.legend.emplace_back(); continue; }
        ax.legend.emplace_back(labels[i], labels[i] + lens[i]);
    }
}

/* -------- Output -------- */

static matlab_plot_buffer to_c_buffer(matlab_plot::RenderResult r) {
    matlab_plot_buffer b{};
    b.data = r.data;
    b.size = r.size;
    return b;
}

matlab_plot_buffer matlab_render_png(void) {
    return to_c_buffer(matlab_plot::render(*matlab_plot::current_figure(),
                                            matlab_plot::Format::Png));
}
matlab_plot_buffer matlab_render_svg(void) {
    return to_c_buffer(matlab_plot::render(*matlab_plot::current_figure(),
                                            matlab_plot::Format::Svg));
}
matlab_plot_buffer matlab_render_pdf(void) {
    return to_c_buffer(matlab_plot::render(*matlab_plot::current_figure(),
                                            matlab_plot::Format::Pdf));
}
void matlab_plot_buffer_free(matlab_plot_buffer buf) {
    std::free(buf.data);
}

int matlab_savefig(const char *path, int64_t plen) {
    if (!path || plen <= 0) return -1;
    std::string p(path, path + plen);
    auto fmt = matlab_plot::Format::Png;
    if      (p.size() >= 4 && p.compare(p.size()-4, 4, ".png") == 0) fmt = matlab_plot::Format::Png;
    else if (p.size() >= 4 && p.compare(p.size()-4, 4, ".svg") == 0) fmt = matlab_plot::Format::Svg;
    else if (p.size() >= 4 && p.compare(p.size()-4, 4, ".pdf") == 0) fmt = matlab_plot::Format::Pdf;
    else return -1;
    return matlab_plot::render_to_file(*matlab_plot::current_figure(),
                                        fmt, p.c_str());
}

}  /* extern "C" */

/* -------- IDE integration ------------------------------------------------ */

namespace {

/* RFC 4648 base64 alphabet. Stream-style encoder: writes directly to
 * stdout in 76-char-wide lines so the IDE-side reader sees readable,
 * easy-to-buffer chunks. Padding ('=') is appended on the final line. */
constexpr const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void write_base64_chunked(const uint8_t *data, size_t n, FILE *out) {
    constexpr int kLine = 76;          /* multiple of 4 — keeps groups intact */
    char line[kLine + 1];
    int  col = 0;
    auto flush = [&]() {
        if (col == 0) return;
        std::fwrite(line, 1, col, out);
        std::fputc('\n', out);
        col = 0;
    };
    auto emit_byte = [&](char c) {
        line[col++] = c;
        if (col == kLine) flush();
    };

    size_t i = 0;
    while (i + 3 <= n) {
        uint32_t w = (uint32_t(data[i]) << 16) |
                     (uint32_t(data[i+1]) << 8) |
                      uint32_t(data[i+2]);
        emit_byte(kB64[(w >> 18) & 0x3F]);
        emit_byte(kB64[(w >> 12) & 0x3F]);
        emit_byte(kB64[(w >>  6) & 0x3F]);
        emit_byte(kB64[ w        & 0x3F]);
        i += 3;
    }
    if (i < n) {
        uint32_t w = uint32_t(data[i]) << 16;
        if (i + 1 < n) w |= uint32_t(data[i+1]) << 8;
        emit_byte(kB64[(w >> 18) & 0x3F]);
        emit_byte(kB64[(w >> 12) & 0x3F]);
        if (i + 1 < n) emit_byte(kB64[(w >> 6) & 0x3F]);
        else           emit_byte('=');
        emit_byte('=');
    }
    flush();
}

bool ide_figures_enabled() {
    const char *v = std::getenv("MATLAB_LLVM_IDE_FIGURES");
    return v && v[0] == '1';
}

/* End-of-process flush is handled by the destructor of figure.cpp's
 * thread_local ThreadFigures registry — std::atexit isn't viable on
 * macOS because thread_local destructors run before atexit handlers
 * on the main thread, so by the time the handler fires the figure
 * vector has already been cleared. The TLS destructor calls into
 * matlab_ide_emit_all_figures while its member storage is still live,
 * which is what we want for both Run mode (compiled exec) and the
 * matlabc REPL (per-input emit happens explicitly from the REPL
 * driver in tools/matlabc/main.cpp). */

}  // namespace

extern "C" {

void matlab_ide_emit_all_figures(void) {
    if (!ide_figures_enabled()) return;
    auto figs = matlab_plot::figures_snapshot();
    if (figs.empty()) return;

    FILE *out = stdout;
    for (matlab_plot::Figure *fig : figs) {
        if (!fig) continue;
        auto buf = matlab_plot::render(*fig, matlab_plot::Format::Png);
        if (!buf.data || buf.size == 0) continue;
        std::fprintf(out,
                     "___MF_FIG_BEGIN___ id=%d w=%d h=%d\n",
                     fig->id, fig->width_px, fig->height_px);
        write_base64_chunked(buf.data, buf.size, out);
        std::fputs("___MF_FIG_END___\n", out);
        std::free(buf.data);
    }
    std::fflush(out);
}

void matlab_drawnow(void) {
    matlab_ide_emit_all_figures();
}

}  /* extern "C" */
