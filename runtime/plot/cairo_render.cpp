#include "cairo_render.h"

#include "colormap.h"

#include <cairo.h>
#include <cairo-svg.h>
#include <cairo-pdf.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace matlab_plot {

namespace {

/* Growable byte sink used as the cairo write target. */
struct ByteSink {
    std::vector<uint8_t> buf;
};

cairo_status_t sink_write(void *closure, const unsigned char *data,
                          unsigned int length) {
    auto *sink = static_cast<ByteSink *>(closure);
    sink->buf.insert(sink->buf.end(), data, data + length);
    return CAIRO_STATUS_SUCCESS;
}

RenderResult sink_to_result(ByteSink &sink) {
    RenderResult r;
    r.size = sink.buf.size();
    r.data = static_cast<uint8_t *>(std::malloc(r.size));
    if (r.data) std::memcpy(r.data, sink.buf.data(), r.size);
    return r;
}

struct Layout {
    double left   = 70;
    double right  = 20;
    double top    = 40;
    double bottom = 60;
};

/* Tick suppression — only true imshow series (with_axes=false) hide the
 * axes; imagesc/pcolor still draw ticks over the colored heatmap. */
bool axes_skips_ticks(const Axes &ax) {
    for (const auto &s : ax.series)
        if (s.kind == SeriesKind::Image && !s.with_axes) return true;
    return false;
}

/* Image data range across all image series in the axes (used by
 * draw_colorbar to label the scale). Returns false if there is none. */
bool axes_image_range(const Axes &ax, double &lo, double &hi) {
    bool any = false;
    lo =  std::numeric_limits<double>::infinity();
    hi = -std::numeric_limits<double>::infinity();
    for (const auto &s : ax.series) {
        if (s.kind != SeriesKind::Image) continue;
        for (double v : s.image) { lo = std::min(lo, v); hi = std::max(hi, v); }
        any = true;
    }
    if (!any || lo == hi) { lo = 0; hi = 1; return any; }
    return any;
}

/* Compute the y-axis data range for a single side (left = false, right = true)
 * across only the series that belong to it. Caller wires the result into a
 * MapCtx for that side. */
void compute_y_range_side(const Axes &ax, bool right,
                          double &y_lo, double &y_hi) {
    y_lo =  std::numeric_limits<double>::infinity();
    y_hi = -std::numeric_limits<double>::infinity();
    for (const auto &s : ax.series) {
        if (s.on_right_axis != right) continue;
        for (double v : s.y) { y_lo = std::min(y_lo, v); y_hi = std::max(y_hi, v); }
        if (s.kind == SeriesKind::Bar || s.kind == SeriesKind::Stem ||
            s.kind == SeriesKind::Area) {
            y_lo = std::min(y_lo, 0.0);
            y_hi = std::max(y_hi, 0.0);
        }
        if (s.kind == SeriesKind::ErrorBar) {
            const size_t n = std::min({s.x.size(), s.y.size(),
                                       s.e_neg.size(), s.e_pos.size()});
            for (size_t i = 0; i < n; ++i) {
                y_lo = std::min(y_lo, s.y[i] - s.e_neg[i]);
                y_hi = std::max(y_hi, s.y[i] + s.e_pos[i]);
            }
        }
    }
    if (!std::isfinite(y_lo) || y_lo == y_hi) { y_lo = 0; y_hi = 1; }
    if (!ax.tight && ax.yscale == Scale::Linear) {
        double pad = (y_hi - y_lo) * 0.05;
        y_lo -= pad; y_hi += pad;
    }
    if (right) {
        if (ax.ylim_right_set) { y_lo = ax.ylim_right_lo; y_hi = ax.ylim_right_hi; }
    } else {
        if (ax.ylim_set)       { y_lo = ax.ylim_lo;       y_hi = ax.ylim_hi;       }
    }
}

void compute_data_range(const Axes &ax,
                        double &x_lo, double &x_hi,
                        double &y_lo, double &y_hi) {
    /* If any image series exists, axes follow the image's pixel coords.
     * The first image wins (subsequent ones are blitted to the same
     * frame). MATLAB's imshow places pixel (1,1) at the top-left, so
     * y is reversed downstream when the image is painted. */
    for (const auto &s : ax.series) {
        if (s.kind == SeriesKind::Image) {
            x_lo = 0.5;            x_hi = s.cols + 0.5;
            y_lo = 0.5;            y_hi = s.rows + 0.5;
            if (ax.xlim_set) { x_lo = ax.xlim_lo; x_hi = ax.xlim_hi; }
            if (ax.ylim_set) { y_lo = ax.ylim_lo; y_hi = ax.ylim_hi; }
            return;
        }
    }
    x_lo = y_lo =  std::numeric_limits<double>::infinity();
    x_hi = y_hi = -std::numeric_limits<double>::infinity();
    for (const auto &s : ax.series) {
        /* x always comes from every series (shared between left and right
         * y-axes); y only from series bound to the LEFT axis. The right
         * axis range is computed separately by compute_y_range_side. */
        for (double v : s.x) { x_lo = std::min(x_lo, v); x_hi = std::max(x_hi, v); }
        if (s.on_right_axis) continue;
        for (double v : s.y) { y_lo = std::min(y_lo, v); y_hi = std::max(y_hi, v); }
        if (s.kind == SeriesKind::Bar || s.kind == SeriesKind::Stem ||
            s.kind == SeriesKind::Area) {
            y_lo = std::min(y_lo, 0.0);
            y_hi = std::max(y_hi, 0.0);
        }
        if (s.kind == SeriesKind::ErrorBar) {
            const size_t n = std::min({s.x.size(), s.y.size(),
                                       s.e_neg.size(), s.e_pos.size()});
            for (size_t i = 0; i < n; ++i) {
                y_lo = std::min(y_lo, s.y[i] - s.e_neg[i]);
                y_hi = std::max(y_hi, s.y[i] + s.e_pos[i]);
            }
        }
    }
    if (!std::isfinite(x_lo) || x_lo == x_hi) { x_lo = 0; x_hi = 1; }
    if (!std::isfinite(y_lo) || y_lo == y_hi) { y_lo = 0; y_hi = 1; }
    /* axis tight suppresses the 5% autoscale padding. */
    if (!ax.ylim_set && !ax.tight && ax.yscale == Scale::Linear) {
        double pad = (y_hi - y_lo) * 0.05;
        y_lo -= pad; y_hi += pad;
    }
    if (ax.xlim_set) { x_lo = ax.xlim_lo; x_hi = ax.xlim_hi; }
    if (ax.ylim_set) { y_lo = ax.ylim_lo; y_hi = ax.ylim_hi; }

    /* Log axes: clamp non-positive bounds to a small positive value and
     * expand range to the nearest enclosing decade. Data points <= 0
     * are skipped at draw time. */
    if (ax.xscale == Scale::Log) {
        if (x_lo <= 0) {
            /* Re-scan for smallest positive x. */
            double pos = std::numeric_limits<double>::infinity();
            for (const auto &s : ax.series)
                for (double v : s.x)
                    if (v > 0 && v < pos) pos = v;
            x_lo = std::isfinite(pos) ? pos : 1.0;
        }
        if (x_hi <= 0) x_hi = 1.0;
        x_lo = std::pow(10.0, std::floor(std::log10(x_lo)));
        x_hi = std::pow(10.0, std::ceil (std::log10(x_hi)));
        if (x_hi <= x_lo) x_hi = x_lo * 10;
    }
    if (ax.yscale == Scale::Log) {
        if (y_lo <= 0) {
            double pos = std::numeric_limits<double>::infinity();
            for (const auto &s : ax.series)
                for (double v : s.y)
                    if (v > 0 && v < pos) pos = v;
            y_lo = std::isfinite(pos) ? pos : 1.0;
        }
        if (y_hi <= 0) y_hi = 1.0;
        y_lo = std::pow(10.0, std::floor(std::log10(y_lo)));
        y_hi = std::pow(10.0, std::ceil (std::log10(y_hi)));
        if (y_hi <= y_lo) y_hi = y_lo * 10;
    }
}

/* Heckbert-style "nice number" tick spacing. */
double nice_step(double range, int target_count) {
    if (range <= 0) return 1.0;
    double raw  = range / std::max(1, target_count);
    double mag  = std::pow(10.0, std::floor(std::log10(raw)));
    double norm = raw / mag;
    double step = (norm < 1.5) ? 1.0
                : (norm < 3.0) ? 2.0
                : (norm < 7.0) ? 5.0
                :                10.0;
    return step * mag;
}

std::vector<double> nice_ticks(double lo, double hi, int target = 6) {
    double step = nice_step(hi - lo, target);
    std::vector<double> ticks;
    double first = std::ceil(lo / step) * step;
    /* Guard against runaway loops on degenerate ranges. */
    for (double v = first; v <= hi + step * 1e-9 && ticks.size() < 64; v += step) {
        ticks.push_back(v);
    }
    return ticks;
}

std::string fmt_tick(double v, double step) {
    char buf[32];
    /* Choose decimals based on step magnitude. */
    int decimals = 0;
    if (step < 1.0) decimals = std::max(0, static_cast<int>(-std::floor(std::log10(step))));
    if (decimals > 6) decimals = 6;
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    /* Suppress -0.000 artifacts. */
    if (std::strncmp(buf, "-0", 2) == 0) {
        bool all_zero = true;
        for (const char *p = buf + 1; *p; ++p) {
            if (*p != '0' && *p != '.' && *p != '-') { all_zero = false; break; }
        }
        if (all_zero) std::snprintf(buf, sizeof(buf), "%.*f", decimals, 0.0);
    }
    return buf;
}

void set_sans_font(cairo_t *cr, double size, bool bold = false) {
    cairo_select_font_face(cr, "sans-serif",
                           CAIRO_FONT_SLANT_NORMAL,
                           bold ? CAIRO_FONT_WEIGHT_BOLD
                                : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
}

void draw_text(cairo_t *cr, double x, double y, const std::string &s) {
    cairo_move_to(cr, x, y);
    cairo_show_text(cr, s.c_str());
}

/* MATLAB markers: o (circle), + (plus), * (asterisk), x (cross),
 * s (square), d (diamond), ^ (triangle up). All drawn at radius `r`
 * centered at (cx, cy). */
void draw_marker(cairo_t *cr, char glyph, double cx, double cy, double r) {
    switch (glyph) {
        case 'o':
            cairo_arc(cr, cx, cy, r, 0, 2 * M_PI);
            cairo_set_line_width(cr, 1.2);
            cairo_stroke(cr);
            break;
        case '+':
            cairo_set_line_width(cr, 1.2);
            cairo_move_to(cr, cx - r, cy); cairo_line_to(cr, cx + r, cy);
            cairo_move_to(cr, cx, cy - r); cairo_line_to(cr, cx, cy + r);
            cairo_stroke(cr);
            break;
        case 'x':
            cairo_set_line_width(cr, 1.2);
            cairo_move_to(cr, cx - r, cy - r); cairo_line_to(cr, cx + r, cy + r);
            cairo_move_to(cr, cx - r, cy + r); cairo_line_to(cr, cx + r, cy - r);
            cairo_stroke(cr);
            break;
        case '*':
            cairo_set_line_width(cr, 1.2);
            cairo_move_to(cr, cx - r, cy);     cairo_line_to(cr, cx + r, cy);
            cairo_move_to(cr, cx, cy - r);     cairo_line_to(cr, cx, cy + r);
            cairo_move_to(cr, cx - r, cy - r); cairo_line_to(cr, cx + r, cy + r);
            cairo_move_to(cr, cx - r, cy + r); cairo_line_to(cr, cx + r, cy - r);
            cairo_stroke(cr);
            break;
        case 's':
            cairo_rectangle(cr, cx - r, cy - r, 2 * r, 2 * r);
            cairo_set_line_width(cr, 1.2);
            cairo_stroke(cr);
            break;
        case 'd':
            cairo_move_to(cr, cx,     cy - r);
            cairo_line_to(cr, cx + r, cy);
            cairo_line_to(cr, cx,     cy + r);
            cairo_line_to(cr, cx - r, cy);
            cairo_close_path(cr);
            cairo_set_line_width(cr, 1.2);
            cairo_stroke(cr);
            break;
        case '^':
            cairo_move_to(cr, cx,     cy - r);
            cairo_line_to(cr, cx + r, cy + r);
            cairo_line_to(cr, cx - r, cy + r);
            cairo_close_path(cr);
            cairo_set_line_width(cr, 1.2);
            cairo_stroke(cr);
            break;
        default: break;
    }
}

/* Configure cairo dashes from the Series LineStyle. Dash lengths scale
 * gently with line width so heavy lines still look right. */
void apply_line_style(cairo_t *cr, LineStyle st, double width) {
    cairo_set_line_width(cr, width);
    switch (st) {
        case LineStyle::Solid:
            cairo_set_dash(cr, nullptr, 0, 0); break;
        case LineStyle::Dashed: {
            double d[2] = {6.0 + width, 4.0 + width / 2};
            cairo_set_dash(cr, d, 2, 0); break;
        }
        case LineStyle::Dotted: {
            double d[2] = {1.5, 3.0};
            cairo_set_dash(cr, d, 2, 0); break;
        }
        case LineStyle::DashDot: {
            double d[4] = {6.0, 3.0, 1.5, 3.0};
            cairo_set_dash(cr, d, 4, 0); break;
        }
    }
}

void draw_line_series(cairo_t *cr, const Series &s,
                      double (*mx)(double, void *), double (*my)(double, void *),
                      void *ctx) {
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);

    /* Connecting line — skipped if linespec is marker-only ('o', 'r*', ...). */
    if (!s.line_off && s.x.size() >= 2) {
        apply_line_style(cr, s.style, s.line_width);
        cairo_move_to(cr, mx(s.x[0], ctx), my(s.y[0], ctx));
        for (size_t i = 1; i < s.x.size(); ++i)
            cairo_line_to(cr, mx(s.x[i], ctx), my(s.y[i], ctx));
        cairo_stroke(cr);
    }

    /* Markers — one per data point if the linespec has one. */
    if (s.marker) {
        cairo_set_dash(cr, nullptr, 0, 0);
        const size_t n = std::min(s.x.size(), s.y.size());
        for (size_t i = 0; i < n; ++i) {
            draw_marker(cr, s.marker,
                        mx(s.x[i], ctx), my(s.y[i], ctx), s.marker_size);
        }
    }
}

void draw_scatter_series(cairo_t *cr, const Series &s,
                         double (*mx)(double, void *),
                         double (*my)(double, void *),
                         void *ctx) {
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.0);
    const double n = static_cast<double>(std::min(s.x.size(), s.y.size()));
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        double cx = mx(s.x[i], ctx), cy = my(s.y[i], ctx);
        cairo_arc(cr, cx, cy, 3.5, 0, 2 * M_PI);
        cairo_fill(cr);
    }
}

/* Area series — line plot with the region under the curve filled down
 * to the y=0 baseline. Outline drawn after fill so the curve stays
 * sharp on top. */
void draw_area_series(cairo_t *cr, const Series &s,
                      double (*mx)(double, void *),
                      double (*my)(double, void *),
                      void *ctx) {
    const size_t n = std::min(s.x.size(), s.y.size());
    if (n < 2) return;
    const double y0 = my(0.0, ctx);

    cairo_set_dash(cr, nullptr, 0, 0);
    cairo_set_source_rgba(cr, s.r, s.g, s.b, 0.35f * s.a);  /* fill alpha */
    cairo_move_to(cr, mx(s.x[0], ctx), y0);
    for (size_t i = 0; i < n; ++i)
        cairo_line_to(cr, mx(s.x[i], ctx), my(s.y[i], ctx));
    cairo_line_to(cr, mx(s.x[n-1], ctx), y0);
    cairo_close_path(cr);
    cairo_fill(cr);

    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.5);
    cairo_move_to(cr, mx(s.x[0], ctx), my(s.y[0], ctx));
    for (size_t i = 1; i < n; ++i)
        cairo_line_to(cr, mx(s.x[i], ctx), my(s.y[i], ctx));
    cairo_stroke(cr);
}

/* Quiver series — arrow per (x[i], y[i]) pointing along (u[i], v[i]).
 * Auto-scales arrow length so the longest arrow is ~1 grid spacing,
 * matching MATLAB's default scale factor.  */
void draw_quiver_series(cairo_t *cr, const Series &s,
                        double (*mx)(double, void *),
                        double (*my)(double, void *),
                        void *ctx) {
    const size_t n = std::min({s.x.size(), s.y.size(), s.u.size(), s.v.size()});
    if (n == 0) return;

    /* Pick a global scale: longest arrow becomes ~70% of the median
     * grid spacing in x. */
    double max_mag = 0;
    for (size_t i = 0; i < n; ++i) {
        double m = std::sqrt(s.u[i] * s.u[i] + s.v[i] * s.v[i]);
        if (m > max_mag) max_mag = m;
    }
    double dx_typical = 1.0;
    if (n >= 2) {
        std::vector<double> xs = s.x;
        std::sort(xs.begin(), xs.end());
        dx_typical = xs[xs.size() / 2] - xs[0];
        if (dx_typical <= 0) dx_typical = 1.0;
    }
    double scale = (max_mag > 0) ? (0.7 * dx_typical / max_mag) : 1.0;

    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.0);
    cairo_set_dash(cr, nullptr, 0, 0);
    for (size_t i = 0; i < n; ++i) {
        double x0 = s.x[i],         y0 = s.y[i];
        double x1 = x0 + s.u[i] * scale, y1 = y0 + s.v[i] * scale;
        double sx0 = mx(x0, ctx), sy0 = my(y0, ctx);
        double sx1 = mx(x1, ctx), sy1 = my(y1, ctx);
        /* Shaft. */
        cairo_move_to(cr, sx0, sy0); cairo_line_to(cr, sx1, sy1);
        /* Arrowhead — two short lines at 25° from the shaft, length 6px. */
        double dx = sx1 - sx0, dy = sy1 - sy0;
        double len = std::sqrt(dx * dx + dy * dy);
        if (len > 1) {
            double ux = dx / len, uy = dy / len;
            const double head_len = 6.0;
            const double cosA =  0.9063, sinA =  0.4226;  /* 25° */
            double hx1 = sx1 - head_len * (ux * cosA + uy * sinA);
            double hy1 = sy1 - head_len * (uy * cosA - ux * sinA);
            double hx2 = sx1 - head_len * (ux * cosA - uy * sinA);
            double hy2 = sy1 - head_len * (uy * cosA + ux * sinA);
            cairo_move_to(cr, sx1, sy1); cairo_line_to(cr, hx1, hy1);
            cairo_move_to(cr, sx1, sy1); cairo_line_to(cr, hx2, hy2);
        }
    }
    cairo_stroke(cr);
}

/* Stem series — vertical line from y=0 baseline to each point with a
 * filled marker at the top. Standard DSP visualisation for discrete
 * samples. */
void draw_stem_series(cairo_t *cr, const Series &s,
                      double (*mx)(double, void *),
                      double (*my)(double, void *),
                      void *ctx) {
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.0);
    cairo_set_dash(cr, nullptr, 0, 0);
    const size_t n = std::min(s.x.size(), s.y.size());
    const double y0 = my(0.0, ctx);
    for (size_t i = 0; i < n; ++i) {
        double sx = mx(s.x[i], ctx), sy = my(s.y[i], ctx);
        cairo_move_to(cr, sx, y0); cairo_line_to(cr, sx, sy);
    }
    cairo_stroke(cr);
    /* Top markers — filled circles. */
    for (size_t i = 0; i < n; ++i) {
        cairo_arc(cr, mx(s.x[i], ctx), my(s.y[i], ctx), 3.0, 0, 2 * M_PI);
        cairo_fill(cr);
    }
}

/* Stairs series — horizontal-then-vertical polyline between adjacent
 * (x_i, y_i) → (x_{i+1}, y_i) → (x_{i+1}, y_{i+1}). */
void draw_stairs_series(cairo_t *cr, const Series &s,
                        double (*mx)(double, void *),
                        double (*my)(double, void *),
                        void *ctx) {
    const size_t n = std::min(s.x.size(), s.y.size());
    if (n < 2) return;
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.5);
    cairo_set_dash(cr, nullptr, 0, 0);
    cairo_move_to(cr, mx(s.x[0], ctx), my(s.y[0], ctx));
    for (size_t i = 1; i < n; ++i) {
        cairo_line_to(cr, mx(s.x[i], ctx), my(s.y[i-1], ctx));
        cairo_line_to(cr, mx(s.x[i], ctx), my(s.y[i],   ctx));
    }
    cairo_stroke(cr);
}

/* Error-bar series — connecting line + per-point vertical bar with
 * 6px hat caps at top and bottom. Reuses series color/dashing. */
void draw_errorbar_series(cairo_t *cr, const Series &s,
                          double (*mx)(double, void *),
                          double (*my)(double, void *),
                          void *ctx) {
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.5);
    cairo_set_dash(cr, nullptr, 0, 0);

    const size_t n = std::min({s.x.size(), s.y.size(),
                               s.e_neg.size(), s.e_pos.size()});
    if (n == 0) return;

    /* Connecting polyline. */
    if (n >= 2) {
        cairo_move_to(cr, mx(s.x[0], ctx), my(s.y[0], ctx));
        for (size_t i = 1; i < n; ++i)
            cairo_line_to(cr, mx(s.x[i], ctx), my(s.y[i], ctx));
        cairo_stroke(cr);
    }

    /* Bars + hat caps. */
    cairo_set_line_width(cr, 1.0);
    constexpr double cap_half = 4.0;
    for (size_t i = 0; i < n; ++i) {
        double sx = mx(s.x[i],                ctx);
        double y_lo = my(s.y[i] - s.e_neg[i], ctx);
        double y_hi = my(s.y[i] + s.e_pos[i], ctx);
        cairo_move_to(cr, sx, y_lo); cairo_line_to(cr, sx, y_hi);
        cairo_move_to(cr, sx - cap_half, y_lo); cairo_line_to(cr, sx + cap_half, y_lo);
        cairo_move_to(cr, sx - cap_half, y_hi); cairo_line_to(cr, sx + cap_half, y_hi);
    }
    cairo_stroke(cr);

    /* Center markers — small filled dots at each (x, y). */
    for (size_t i = 0; i < n; ++i) {
        cairo_arc(cr, mx(s.x[i], ctx), my(s.y[i], ctx), 2.5, 0, 2 * M_PI);
        cairo_fill(cr);
    }
}

/* Contour series — points come in pairs (start, end) per segment. */
void draw_contour_series(cairo_t *cr, const Series &s,
                         double (*mx)(double, void *),
                         double (*my)(double, void *),
                         void *ctx) {
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    cairo_set_line_width(cr, 1.0);
    cairo_set_dash(cr, nullptr, 0, 0);
    const size_t n = std::min(s.x.size(), s.y.size());
    for (size_t i = 0; i + 1 < n; i += 2) {
        cairo_move_to(cr, mx(s.x[i],   ctx), my(s.y[i],   ctx));
        cairo_line_to(cr, mx(s.x[i+1], ctx), my(s.y[i+1], ctx));
    }
    cairo_stroke(cr);
}

void draw_bar_series(cairo_t *cr, const Series &s,
                     double (*mx)(double, void *),
                     double (*my)(double, void *),
                     void *ctx) {
    if (s.x.empty()) return;
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    /* Bar width: 70% of the smallest neighbor gap (in data units) so groups
     * don't overlap. Falls back to a unit-step heuristic for a single bar. */
    double min_gap = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < s.x.size(); ++i)
        min_gap = std::min(min_gap, std::abs(s.x[i] - s.x[i-1]));
    if (!std::isfinite(min_gap) || min_gap == 0) min_gap = 1.0;
    const double half = s.bar_half_factor * min_gap;
    const double y0   = my(0.0, ctx);
    for (size_t i = 0; i < std::min(s.x.size(), s.y.size()); ++i) {
        double xL = mx(s.x[i] - half, ctx);
        double xR = mx(s.x[i] + half, ctx);
        double yT = my(s.y[i], ctx);
        cairo_rectangle(cr, xL, std::min(y0, yT), xR - xL, std::abs(y0 - yT));
        cairo_fill_preserve(cr);
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_line_width(cr, 0.5);
        cairo_stroke(cr);
        cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    }
}

struct MapCtx {
    double x_lo, x_hi, y_lo, y_hi;
    double left, top, plot_w, plot_h;
    bool   y_inverted = false;
    bool   x_log = false;
    bool   y_log = false;
};

/* Pre-transform a data value for the active scale. log10 of <=0 is
 * undefined; we treat it as the lower clamp so the segment endpoint
 * stays at the bottom of the plot rather than vanishing. */
inline double scaled(double v, bool log_axis, double lo) {
    if (!log_axis) return v;
    return v > 0 ? std::log10(v) : std::log10(lo > 0 ? lo : 1e-300);
}

double map_x(double v, void *p) {
    auto *c = static_cast<MapCtx *>(p);
    double lo = c->x_log ? std::log10(c->x_lo) : c->x_lo;
    double hi = c->x_log ? std::log10(c->x_hi) : c->x_hi;
    double sv = scaled(v, c->x_log, c->x_lo);
    return c->left + (sv - lo) / (hi - lo) * c->plot_w;
}
double map_y(double v, void *p) {
    auto *c = static_cast<MapCtx *>(p);
    double lo = c->y_log ? std::log10(c->y_lo) : c->y_lo;
    double hi = c->y_log ? std::log10(c->y_hi) : c->y_hi;
    double sv = scaled(v, c->y_log, c->y_lo);
    double t = (sv - lo) / (hi - lo);
    return c->y_inverted ? c->top + t * c->plot_h
                         : c->top + c->plot_h - t * c->plot_h;
}

/* Decade ticks for log axes — emits 10^k for each integer k whose value
 * lies in [lo, hi]. Caller must ensure lo > 0. */
std::vector<double> log_ticks(double lo, double hi) {
    std::vector<double> out;
    if (lo <= 0 || hi <= lo) return out;
    int k_lo = static_cast<int>(std::floor(std::log10(lo)));
    int k_hi = static_cast<int>(std::ceil (std::log10(hi)));
    for (int k = k_lo; k <= k_hi && out.size() < 32; ++k) {
        double v = std::pow(10.0, k);
        if (v >= lo && v <= hi) out.push_back(v);
    }
    return out;
}

std::string fmt_log_tick(double v) {
    /* For magnitudes near 1 print as integer or fixed-point; otherwise
     * scientific 1eK. */
    int k = static_cast<int>(std::round(std::log10(v)));
    char buf[32];
    if (k >= 0 && k <= 4) {
        std::snprintf(buf, sizeof(buf), "%.0f", v);
    } else if (k >= -2 && k < 0) {
        std::snprintf(buf, sizeof(buf), "%.*f", -k, v);
    } else {
        std::snprintf(buf, sizeof(buf), "1e%d", k);
    }
    return buf;
}

/* imshow / imagesc — autoscale data to [0, 1], look up RGB via the
 * supplied colormap, and blit row-major (pixel (0,0) at top-left). Cairo
 * ARGB32 pixels are native-endian 0xAARRGGBB; on little-endian boxes the
 * byte sequence is BGRA. */
void draw_image_series(cairo_t *cr, const Series &s, Colormap cm,
                       double left, double top,
                       double plot_w, double plot_h) {
    if (s.rows <= 0 || s.cols <= 0 ||
        s.image.size() != static_cast<size_t>(s.rows) * s.cols) return;

    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (double v : s.image) { lo = std::min(lo, v); hi = std::max(hi, v); }
    double range = (hi > lo) ? (hi - lo) : 1.0;

    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, s.cols);
    std::vector<uint8_t> buf(static_cast<size_t>(stride) * s.rows);
    for (int r = 0; r < s.rows; ++r) {
        auto *row = reinterpret_cast<uint32_t *>(buf.data() + r * stride);
        for (int c = 0; c < s.cols; ++c) {
            double n = (s.image[r * s.cols + c] - lo) / range;
            if (n < 0) n = 0; if (n > 1) n = 1;
            float fr, fg, fb;
            cmap_eval(cm, n, fr, fg, fb);
            uint32_t R = static_cast<uint32_t>(fr * 255.0f + 0.5f);
            uint32_t G = static_cast<uint32_t>(fg * 255.0f + 0.5f);
            uint32_t B = static_cast<uint32_t>(fb * 255.0f + 0.5f);
            row[c] = 0xFF000000u | (R << 16) | (G << 8) | B;
        }
    }

    cairo_surface_t *img = cairo_image_surface_create_for_data(
        buf.data(), CAIRO_FORMAT_ARGB32, s.cols, s.rows, stride);
    if (cairo_surface_status(img) != CAIRO_STATUS_SUCCESS) {
        cairo_surface_destroy(img);
        return;
    }
    cairo_save(cr);
    cairo_translate(cr, left, top);
    cairo_scale(cr, plot_w / s.cols, plot_h / s.rows);
    cairo_set_source_surface(cr, img, 0, 0);
    /* Nearest-neighbor: faithful for scientific image display, no blur. */
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_NEAREST);
    cairo_paint(cr);
    cairo_restore(cr);
    cairo_surface_destroy(img);
}

/* Vertical colorbar drawn just to the right of the plot area. Reads the
 * combined image data range from `ax` for the tick labels and uses the
 * active colormap for the strip itself. Width is fixed (16px); reserve
 * ~60px of plot-area shrinkage in the layout when colorbar is enabled. */
void draw_colorbar(cairo_t *cr, const Axes &ax,
                   double plot_x, double plot_y,
                   double plot_w, double plot_h) {
    double lo, hi;
    if (!axes_image_range(ax, lo, hi)) return;

    const double bar_w = 16;
    const double bar_x = plot_x + plot_w + 12;
    const double bar_y = plot_y;
    const double bar_h = plot_h;

    /* Render the strip via a small ARGB32 surface (1 column × N rows). */
    constexpr int N = 256;
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, 1);
    std::vector<uint8_t> buf(static_cast<size_t>(stride) * N);
    for (int r = 0; r < N; ++r) {
        double t = 1.0 - static_cast<double>(r) / (N - 1);  // top = high
        float fr, fg, fb; cmap_eval(ax.cmap, t, fr, fg, fb);
        uint32_t R = static_cast<uint32_t>(fr * 255 + 0.5);
        uint32_t G = static_cast<uint32_t>(fg * 255 + 0.5);
        uint32_t B = static_cast<uint32_t>(fb * 255 + 0.5);
        auto *row = reinterpret_cast<uint32_t *>(buf.data() + r * stride);
        row[0] = 0xFF000000u | (R << 16) | (G << 8) | B;
    }
    cairo_surface_t *strip = cairo_image_surface_create_for_data(
        buf.data(), CAIRO_FORMAT_ARGB32, 1, N, stride);
    cairo_save(cr);
    cairo_translate(cr, bar_x, bar_y);
    cairo_scale(cr, bar_w, bar_h / N);
    cairo_set_source_surface(cr, strip, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);
    cairo_surface_destroy(strip);

    /* Frame + tick labels (5 evenly-spaced values from hi → lo). */
    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_set_line_width(cr, 0.75);
    cairo_set_dash(cr, nullptr, 0, 0);
    cairo_rectangle(cr, bar_x, bar_y, bar_w, bar_h);
    cairo_stroke(cr);

    set_sans_font(cr, 10);
    constexpr int N_TICKS = 5;
    double step = (hi - lo) / 5;  // for fmt rounding
    for (int i = 0; i < N_TICKS; ++i) {
        double frac = static_cast<double>(i) / (N_TICKS - 1);
        double v    = hi - frac * (hi - lo);
        double sy   = bar_y + frac * bar_h;
        cairo_move_to(cr, bar_x + bar_w,     sy);
        cairo_line_to(cr, bar_x + bar_w + 4, sy);
        cairo_stroke(cr);
        std::string label = fmt_tick(v, std::abs(step));
        cairo_text_extents_t te; cairo_text_extents(cr, label.c_str(), &te);
        draw_text(cr, bar_x + bar_w + 6, sy + te.height / 2 - 1, label);
    }
}

void draw_legend(cairo_t *cr, const Axes &ax,
                 double left, double top,
                 double plot_w, double /*plot_h*/) {
    if (ax.legend.empty()) return;
    set_sans_font(cr, 11);

    /* Width = max label width + swatch + padding. Height = entries × 18 + pad. */
    double label_w = 0;
    for (const auto &s : ax.legend) {
        cairo_text_extents_t te; cairo_text_extents(cr, s.c_str(), &te);
        label_w = std::max(label_w, te.width);
    }
    const double swatch_w = 22, gap = 6, pad = 8;
    const double row_h = 18;
    const double box_w = swatch_w + gap + label_w + pad * 2;
    const double box_h = static_cast<double>(ax.legend.size()) * row_h + pad * 2;

    /* Anchor: inset 8px from upper-right of plot area. */
    const double box_x = left + plot_w - box_w - 8;
    const double box_y = top + 8;

    /* Frame. */
    cairo_set_source_rgba(cr, 1, 1, 1, 0.9);
    cairo_rectangle(cr, box_x, box_y, box_w, box_h);
    cairo_fill_preserve(cr);
    cairo_set_source_rgb(cr, 0.4, 0.4, 0.4);
    cairo_set_line_width(cr, 0.75);
    cairo_set_dash(cr, nullptr, 0, 0);
    cairo_stroke(cr);

    /* Per-entry: swatch (line + dash if applicable) + label. Pulls
     * style from the matching series by index. */
    for (size_t i = 0; i < ax.legend.size(); ++i) {
        double cy = box_y + pad + i * row_h + row_h / 2;
        if (i < ax.series.size()) {
            const Series &s = ax.series[i];
            cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
            cairo_set_line_width(cr, 1.5);
            if (s.kind == SeriesKind::Line) {
                if (!s.line_off) {
                    apply_line_style(cr, s.style, std::min(s.line_width, 1.5));
                    cairo_move_to(cr, box_x + pad,            cy);
                    cairo_line_to(cr, box_x + pad + swatch_w, cy);
                    cairo_stroke(cr);
                }
                if (s.marker) {
                    draw_marker(cr, s.marker,
                                box_x + pad + swatch_w / 2, cy, 3.0);
                }
            } else if (s.kind == SeriesKind::Scatter) {
                cairo_arc(cr, box_x + pad + swatch_w / 2, cy, 3.5, 0, 2 * M_PI);
                cairo_fill(cr);
            } else { /* Bar / Image */
                cairo_rectangle(cr, box_x + pad, cy - 5, swatch_w, 10);
                cairo_fill(cr);
            }
        }
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_dash(cr, nullptr, 0, 0);
        cairo_text_extents_t te;
        cairo_text_extents(cr, ax.legend[i].c_str(), &te);
        draw_text(cr, box_x + pad + swatch_w + gap,
                       cy + te.height / 2 - 1, ax.legend[i]);
    }
}

/* ---- 3D projection ----
 * Orthographic projection from data coords (x, y, z) to screen (sx, sy).
 * The view is parameterized by azimuth (rotation around z) and elevation
 * (tilt above the xy plane), matching MATLAB's view(az, el) convention.
 * Depth (away from camera) is also returned for painter's-algorithm
 * sorting in surf/mesh fills. */
struct Proj3 {
    double cx, cy, cz;          /* data center */
    double rx, ry, rz;           /* data half-range per axis */
    double right_x, right_y;    /* world basis vectors mapped to screen */
    double up_x,   up_y, up_z;
    double view_x, view_y, view_z;
    double scale;                /* world units → screen pixels */
    double screen_cx, screen_cy; /* center of plot rect */
};

void make_proj3(const Axes &ax, double plot_x, double plot_y,
                double plot_w, double plot_h,
                double x_lo, double x_hi,
                double y_lo, double y_hi,
                double z_lo, double z_hi,
                Proj3 &P) {
    P.cx = 0.5 * (x_lo + x_hi); P.rx = std::max(1e-12, 0.5 * (x_hi - x_lo));
    P.cy = 0.5 * (y_lo + y_hi); P.ry = std::max(1e-12, 0.5 * (y_hi - y_lo));
    P.cz = 0.5 * (z_lo + z_hi); P.rz = std::max(1e-12, 0.5 * (z_hi - z_lo));

    double a = ax.view_az * M_PI / 180.0;
    double e = ax.view_el * M_PI / 180.0;
    double sa = std::sin(a), ca = std::cos(a);
    double se = std::sin(e), ce = std::cos(e);

    /* Right axis: in xy-plane, perpendicular to view. */
    P.right_x =  ca; P.right_y = sa;
    /* Up axis: tilted by elevation. */
    P.up_x = -sa * se; P.up_y = ca * se; P.up_z = ce;
    /* View axis (away from camera). */
    P.view_x = -sa * ce; P.view_y = ca * ce; P.view_z = -se;

    /* Pick scale so the unit data box fits with margin. The world bounding
     * box's screen extent is bounded by sum(|right_i| * r_i) horizontally
     * and sum(|up_i| * r_i) vertically. */
    double half_w = std::abs(P.right_x) * P.rx + std::abs(P.right_y) * P.ry;
    double half_h = std::abs(P.up_x)    * P.rx + std::abs(P.up_y)    * P.ry
                  + std::abs(P.up_z)    * P.rz;
    P.scale = std::min(plot_w / (2 * half_w), plot_h / (2 * half_h)) * 0.9;
    P.screen_cx = plot_x + plot_w / 2;
    P.screen_cy = plot_y + plot_h / 2;
}

inline void project(const Proj3 &P, double x, double y, double z,
                    double &sx, double &sy, double &depth) {
    double dx = x - P.cx, dy = y - P.cy, dz = z - P.cz;
    sx    = P.screen_cx + P.scale * (P.right_x * dx + P.right_y * dy);
    /* Cairo screen y points down; up_* gives "up", so we subtract. */
    sy    = P.screen_cy - P.scale * (P.up_x * dx + P.up_y * dy + P.up_z * dz);
    depth = P.view_x * dx + P.view_y * dy + P.view_z * dz;
}

/* Compute (x, y, z) data range for a 3-D axes. */
void compute_data_range_3d(const Axes &ax,
                           double &x_lo, double &x_hi,
                           double &y_lo, double &y_hi,
                           double &z_lo, double &z_hi) {
    x_lo = y_lo = z_lo =  std::numeric_limits<double>::infinity();
    x_hi = y_hi = z_hi = -std::numeric_limits<double>::infinity();
    for (const auto &s : ax.series) {
        for (double v : s.x) { x_lo = std::min(x_lo, v); x_hi = std::max(x_hi, v); }
        for (double v : s.y) { y_lo = std::min(y_lo, v); y_hi = std::max(y_hi, v); }
        for (double v : s.z) { z_lo = std::min(z_lo, v); z_hi = std::max(z_hi, v); }
    }
    if (!std::isfinite(x_lo) || x_lo == x_hi) { x_lo = 0; x_hi = 1; }
    if (!std::isfinite(y_lo) || y_lo == y_hi) { y_lo = 0; y_hi = 1; }
    if (!std::isfinite(z_lo) || z_lo == z_hi) { z_lo = 0; z_hi = 1; }
    if (ax.xlim_set) { x_lo = ax.xlim_lo; x_hi = ax.xlim_hi; }
    if (ax.ylim_set) { y_lo = ax.ylim_lo; y_hi = ax.ylim_hi; }
}

void draw_box_3d(cairo_t *cr, const Proj3 &P,
                 double x_lo, double x_hi,
                 double y_lo, double y_hi,
                 double z_lo, double z_hi) {
    /* Eight corners of the box, projected. */
    const double cx[8] = {x_lo, x_hi, x_hi, x_lo, x_lo, x_hi, x_hi, x_lo};
    const double cy[8] = {y_lo, y_lo, y_hi, y_hi, y_lo, y_lo, y_hi, y_hi};
    const double cz[8] = {z_lo, z_lo, z_lo, z_lo, z_hi, z_hi, z_hi, z_hi};
    double sx[8], sy[8], dep[8];
    for (int i = 0; i < 8; ++i)
        project(P, cx[i], cy[i], cz[i], sx[i], sy[i], dep[i]);

    /* Twelve edges of the cube (pairs of corner indices). */
    const int E[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},   /* bottom */
        {4,5}, {5,6}, {6,7}, {7,4},   /* top */
        {0,4}, {1,5}, {2,6}, {3,7},   /* verticals */
    };
    cairo_set_source_rgb(cr, 0.3, 0.3, 0.3);
    cairo_set_line_width(cr, 0.6);
    cairo_set_dash(cr, nullptr, 0, 0);
    for (auto &e : E) {
        cairo_move_to(cr, sx[e[0]], sy[e[0]]);
        cairo_line_to(cr, sx[e[1]], sy[e[1]]);
    }
    cairo_stroke(cr);

    /* Axis labels at the front edges of the box. Use the corner with the
     * largest depth as the front-bottom-left for x/y placement. */
    set_sans_font(cr, 11);
    cairo_set_source_rgb(cr, 0, 0, 0);
    char buf[32];
    /* x ticks at z_lo, y_lo edge: corners 0 → 1. */
    std::snprintf(buf, sizeof(buf), "%.3g", x_lo); draw_text(cr, sx[0] - 8, sy[0] + 14, buf);
    std::snprintf(buf, sizeof(buf), "%.3g", x_hi); draw_text(cr, sx[1] - 8, sy[1] + 14, buf);
    /* y ticks at the other front edge: corners 1 → 2. */
    std::snprintf(buf, sizeof(buf), "%.3g", y_lo); draw_text(cr, sx[1] + 4, sy[1] + 14, buf);
    std::snprintf(buf, sizeof(buf), "%.3g", y_hi); draw_text(cr, sx[2] + 4, sy[2] + 4,  buf);
    /* z ticks on a vertical edge. */
    std::snprintf(buf, sizeof(buf), "%.3g", z_lo); draw_text(cr, sx[0] - 30, sy[0],     buf);
    std::snprintf(buf, sizeof(buf), "%.3g", z_hi); draw_text(cr, sx[4] - 30, sy[4],     buf);
}

/* Mesh/Surf rendering. Tessellate the m×n grid into (m-1)*(n-1) quads,
 * compute centroid depth, sort back-to-front, and either fill (Surf —
 * colormap-shaded by centroid z) or stroke (Mesh — wireframe). This is
 * a simple painter's algorithm: it gets interpenetrating geometry wrong
 * but is correct for any well-behaved height-field surface, which is
 * what surf/mesh actually receive. */
void draw_grid_series(cairo_t *cr, const Series &s, const Proj3 &P,
                      Colormap cm, double z_lo, double z_hi) {
    if (s.rows < 2 || s.cols < 2) return;
    const int M = s.rows, N = s.cols;
    const int n_quads = (M - 1) * (N - 1);

    struct Quad {
        double sx[4], sy[4];
        double depth;
        double z_avg;
    };
    std::vector<Quad> quads;
    quads.reserve(n_quads);

    for (int r = 0; r < M - 1; ++r) {
        for (int c = 0; c < N - 1; ++c) {
            const int idx[4] = {
                r * N + c,           /* TL */
                r * N + (c + 1),     /* TR */
                (r + 1) * N + (c+1), /* BR */
                (r + 1) * N + c,     /* BL */
            };
            Quad q;
            double dep_sum = 0, z_sum = 0;
            for (int k = 0; k < 4; ++k) {
                double sx, sy, d;
                project(P, s.x[idx[k]], s.y[idx[k]], s.z[idx[k]], sx, sy, d);
                q.sx[k] = sx; q.sy[k] = sy;
                dep_sum += d;
                z_sum += s.z[idx[k]];
            }
            q.depth = dep_sum * 0.25;
            q.z_avg = z_sum   * 0.25;
            quads.push_back(q);
        }
    }

    /* Painter's algorithm: largest depth (farthest) drawn first. */
    std::sort(quads.begin(), quads.end(),
              [](const Quad &a, const Quad &b) { return a.depth > b.depth; });

    const double z_range = (z_hi > z_lo) ? (z_hi - z_lo) : 1.0;
    const bool fill = (s.kind == SeriesKind::Surf);
    cairo_set_dash(cr, nullptr, 0, 0);
    cairo_set_line_width(cr, fill ? 0.4 : 1.0);
    for (const auto &q : quads) {
        cairo_move_to(cr, q.sx[0], q.sy[0]);
        cairo_line_to(cr, q.sx[1], q.sy[1]);
        cairo_line_to(cr, q.sx[2], q.sy[2]);
        cairo_line_to(cr, q.sx[3], q.sy[3]);
        cairo_close_path(cr);
        if (fill) {
            double t = (q.z_avg - z_lo) / z_range;
            float fr, fg, fb;
            cmap_eval(cm, t, fr, fg, fb);
            cairo_set_source_rgba(cr, fr, fg, fb, 1.0);
            cairo_fill_preserve(cr);
            /* Thin black edge for definition. */
            cairo_set_source_rgba(cr, 0, 0, 0, 0.4);
            cairo_stroke(cr);
        } else {
            cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
            cairo_stroke(cr);
        }
    }
}

void draw_line3d_series(cairo_t *cr, const Series &s, const Proj3 &P) {
    const size_t n = std::min({s.x.size(), s.y.size(), s.z.size()});
    if (n < 2) return;
    cairo_set_source_rgba(cr, s.r, s.g, s.b, s.a);
    apply_line_style(cr, s.style, s.line_width);
    double sx, sy, d;
    project(P, s.x[0], s.y[0], s.z[0], sx, sy, d);
    cairo_move_to(cr, sx, sy);
    for (size_t i = 1; i < n; ++i) {
        project(P, s.x[i], s.y[i], s.z[i], sx, sy, d);
        cairo_line_to(cr, sx, sy);
    }
    cairo_stroke(cr);
}

void draw_axes_3d(cairo_t *cr, const Axes &ax,
                  double cell_x, double cell_y,
                  double cell_w, double cell_h) {
    Layout L;
    const double plot_x = cell_x + L.left;
    const double plot_y = cell_y + L.top;
    const double plot_w = cell_w - L.left - L.right;
    const double plot_h = cell_h - L.top  - L.bottom;
    if (plot_w <= 0 || plot_h <= 0) return;

    double x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
    compute_data_range_3d(ax, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi);

    Proj3 P;
    make_proj3(ax, plot_x, plot_y, plot_w, plot_h,
               x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, P);

    /* Bounding box first (drawn under the data). */
    if (!ax.axis_off) draw_box_3d(cr, P, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi);

    /* Series — Line3D, Mesh, Surf. Surf/Mesh get drawn first so Line3D
     * overlays on top. */
    cairo_save(cr);
    cairo_rectangle(cr, plot_x, plot_y, plot_w, plot_h);
    cairo_clip(cr);
    for (const auto &s : ax.series) {
        if (s.kind == SeriesKind::Surf || s.kind == SeriesKind::Mesh)
            draw_grid_series(cr, s, P, ax.cmap, z_lo, z_hi);
    }
    for (const auto &s : ax.series) {
        if (s.kind == SeriesKind::Line3D) draw_line3d_series(cr, s, P);
    }
    cairo_restore(cr);

    if (!ax.title.empty()) {
        set_sans_font(cr, 14, /*bold=*/true);
        cairo_text_extents_t te; cairo_text_extents(cr, ax.title.c_str(), &te);
        cairo_set_source_rgb(cr, 0, 0, 0);
        draw_text(cr, plot_x + (plot_w - te.width) / 2,
                       plot_y - 12, ax.title);
    }
}

/* Per-axes painter — operates on a cell rect (cell_x, cell_y, cell_w,
 * cell_h) within the figure. Layout margins are subtracted from the
 * cell, not the whole figure, so subplot grids work uniformly. When the
 * axes has a colorbar enabled, ~60px of right-side space is reserved for
 * the strip and its tick labels. */
void draw_axes(cairo_t *cr, const Axes &ax,
               double cell_x, double cell_y,
               double cell_w, double cell_h) {
    Layout L;
    const double cbar_reserve = ax.colorbar ? 60.0 : 0.0;
    /* axis off: collapse outer padding to zero so the data fills the cell. */
    const double pad_l = ax.axis_off ? 0 : L.left;
    const double pad_r = ax.axis_off ? 0 : L.right;
    const double pad_t = ax.axis_off ? 0 : L.top;
    const double pad_b = ax.axis_off ? 0 : L.bottom;
    double plot_x = cell_x + pad_l;
    double plot_y = cell_y + pad_t;
    double plot_w = cell_w - pad_l - pad_r - cbar_reserve;
    double plot_h = cell_h - pad_t - pad_b;
    if (plot_w <= 0 || plot_h <= 0) return;

    double x_lo, x_hi, y_lo, y_hi;
    compute_data_range(ax, x_lo, x_hi, y_lo, y_hi);

    /* axis equal: pick the smaller of the two scales (px-per-data-unit)
     * and shrink the larger plot dimension to match. Center the result
     * in the cell so the plot doesn't get jammed against an edge. */
    if (ax.axis_equal) {
        double x_range = x_hi - x_lo;
        double y_range = y_hi - y_lo;
        if (x_range > 0 && y_range > 0) {
            double scale = std::min(plot_w / x_range, plot_h / y_range);
            double new_w = scale * x_range;
            double new_h = scale * y_range;
            plot_x += (plot_w - new_w) / 2;
            plot_y += (plot_h - new_h) / 2;
            plot_w = new_w;
            plot_h = new_h;
        }
    }

    /* axis off / box off: tick generation is wasted — gate it. */
    const bool draw_decoration = !ax.axis_off;
    const bool skip_ticks = axes_skips_ticks(ax) || !draw_decoration;
    const bool x_log = (ax.xscale == Scale::Log);
    const bool y_log = (ax.yscale == Scale::Log);
    auto x_ticks = skip_ticks ? std::vector<double>{}
                  : !ax.xticks_v.empty() ? ax.xticks_v
                  : x_log ? log_ticks(x_lo, x_hi)
                          : nice_ticks(x_lo, x_hi);
    auto y_ticks = skip_ticks ? std::vector<double>{}
                  : !ax.yticks_v.empty() ? ax.yticks_v
                  : y_log ? log_ticks(y_lo, y_hi)
                          : nice_ticks(y_lo, y_hi);
    double x_step = nice_step(x_hi - x_lo, 6);
    double y_step = nice_step(y_hi - y_lo, 6);

    MapCtx ctx{x_lo, x_hi, y_lo, y_hi, plot_x, plot_y, plot_w, plot_h,
               ax.y_inverted, x_log, y_log};

    /* Right-side y-axis context, computed only when any series uses it.
     * Shares x range with the left-axis context; only y range differs. */
    MapCtx ctx_right = ctx;
    if (ax.y_right_used) {
        double yr_lo, yr_hi;
        compute_y_range_side(ax, /*right=*/true, yr_lo, yr_hi);
        ctx_right.y_lo = yr_lo;
        ctx_right.y_hi = yr_hi;
    }

    if (ax.grid && !skip_ticks) {
        cairo_set_source_rgb(cr, 0.85, 0.85, 0.85);
        cairo_set_line_width(cr, 0.5);
        cairo_set_dash(cr, nullptr, 0, 0);
        for (double t : x_ticks) {
            double sx = map_x(t, &ctx);
            cairo_move_to(cr, sx, plot_y);
            cairo_line_to(cr, sx, plot_y + plot_h);
        }
        for (double t : y_ticks) {
            double sy = map_y(t, &ctx);
            cairo_move_to(cr, plot_x,          sy);
            cairo_line_to(cr, plot_x + plot_w, sy);
        }
        cairo_stroke(cr);
    }

    /* Frame — skipped under axis off / box off. */
    if (draw_decoration && !ax.box_off) {
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_line_width(cr, 1.0);
        cairo_set_dash(cr, nullptr, 0, 0);
        cairo_rectangle(cr, plot_x, plot_y, plot_w, plot_h);
        cairo_stroke(cr);
    }

    /* Ticks + labels. Custom labels (xtick_labels / ytick_labels) when
     * set replace the auto-formatted numeric strings; arrays shorter
     * than the tick list fall back to the numeric format for the tail. */
    auto label_for = [&](size_t i, double t, bool log_axis, double step,
                          const std::vector<std::string> &custom) {
        if (i < custom.size() && !custom[i].empty()) return custom[i];
        return log_axis ? fmt_log_tick(t) : fmt_tick(t, step);
    };
    set_sans_font(cr, 11);
    for (size_t i = 0; i < x_ticks.size(); ++i) {
        double t = x_ticks[i];
        double sx = map_x(t, &ctx);
        cairo_move_to(cr, sx, plot_y + plot_h);
        cairo_line_to(cr, sx, plot_y + plot_h + 5);
        cairo_stroke(cr);
        std::string label = label_for(i, t, x_log, x_step, ax.xtick_labels);
        cairo_text_extents_t te; cairo_text_extents(cr, label.c_str(), &te);
        draw_text(cr, sx - te.width / 2,
                       plot_y + plot_h + 5 + te.height + 4, label);
    }
    for (size_t i = 0; i < y_ticks.size(); ++i) {
        double t = y_ticks[i];
        double sy = map_y(t, &ctx);
        cairo_move_to(cr, plot_x,     sy);
        cairo_line_to(cr, plot_x - 5, sy);
        cairo_stroke(cr);
        std::string label = label_for(i, t, y_log, y_step, ax.ytick_labels);
        cairo_text_extents_t te; cairo_text_extents(cr, label.c_str(), &te);
        draw_text(cr, plot_x - 8 - te.width, sy + te.height / 2 - 1, label);
    }

    if (draw_decoration && !ax.xlabel.empty()) {
        set_sans_font(cr, 12);
        cairo_text_extents_t te;
        cairo_text_extents(cr, ax.xlabel.c_str(), &te);
        draw_text(cr, plot_x + (plot_w - te.width) / 2,
                       cell_y + cell_h - 12, ax.xlabel);
    }
    if (draw_decoration && !ax.ylabel.empty()) {
        set_sans_font(cr, 12);
        cairo_text_extents_t te;
        cairo_text_extents(cr, ax.ylabel.c_str(), &te);
        cairo_save(cr);
        cairo_translate(cr, cell_x + 16, plot_y + (plot_h + te.width) / 2);
        cairo_rotate(cr, -M_PI / 2);
        cairo_move_to(cr, 0, 0);
        cairo_show_text(cr, ax.ylabel.c_str());
        cairo_restore(cr);
    }

    if (draw_decoration && !ax.title.empty()) {
        set_sans_font(cr, 14, /*bold=*/true);
        cairo_text_extents_t te;
        cairo_text_extents(cr, ax.title.c_str(), &te);
        draw_text(cr, plot_x + (plot_w - te.width) / 2,
                       plot_y - 12, ax.title);
    }

    cairo_save(cr);
    cairo_rectangle(cr, plot_x, plot_y, plot_w, plot_h);
    cairo_clip(cr);
    for (const auto &s : ax.series) {
        MapCtx *active = s.on_right_axis ? &ctx_right : &ctx;
        switch (s.kind) {
            case SeriesKind::Line:    draw_line_series   (cr, s, map_x, map_y, active); break;
            case SeriesKind::Scatter: draw_scatter_series(cr, s, map_x, map_y, active); break;
            case SeriesKind::Bar:     draw_bar_series    (cr, s, map_x, map_y, active); break;
            case SeriesKind::Image:
                draw_image_series(cr, s, ax.cmap,
                                  plot_x, plot_y, plot_w, plot_h);
                break;
            case SeriesKind::Contour:
                draw_contour_series(cr, s, map_x, map_y, active);
                break;
            case SeriesKind::ErrorBar:
                draw_errorbar_series(cr, s, map_x, map_y, active);
                break;
            case SeriesKind::Stem:
                draw_stem_series(cr, s, map_x, map_y, active);
                break;
            case SeriesKind::Stairs:
                draw_stairs_series(cr, s, map_x, map_y, active);
                break;
            case SeriesKind::Area:
                draw_area_series(cr, s, map_x, map_y, active);
                break;
            case SeriesKind::Quiver:
                draw_quiver_series(cr, s, map_x, map_y, active);
                break;
            /* 3-D series belong on a 3-D axes; ignore in the 2-D path. */
            case SeriesKind::Line3D:
            case SeriesKind::Mesh:
            case SeriesKind::Surf:
                break;
        }
    }
    cairo_restore(cr);

    /* Right-side y-axis ticks + label, when in use. */
    if (ax.y_right_used) {
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_line_width(cr, 1.0);
        cairo_set_dash(cr, nullptr, 0, 0);
        auto y2_ticks = nice_ticks(ctx_right.y_lo, ctx_right.y_hi);
        double y2_step = nice_step(ctx_right.y_hi - ctx_right.y_lo, 6);
        set_sans_font(cr, 11);
        for (double t : y2_ticks) {
            double sy = map_y(t, &ctx_right);
            cairo_move_to(cr, plot_x + plot_w,     sy);
            cairo_line_to(cr, plot_x + plot_w + 5, sy);
            cairo_stroke(cr);
            std::string label = fmt_tick(t, y2_step);
            cairo_text_extents_t te; cairo_text_extents(cr, label.c_str(), &te);
            draw_text(cr, plot_x + plot_w + 8, sy + te.height / 2 - 1, label);
        }
        if (!ax.ylabel_right.empty()) {
            set_sans_font(cr, 12);
            cairo_text_extents_t te;
            cairo_text_extents(cr, ax.ylabel_right.c_str(), &te);
            cairo_save(cr);
            cairo_translate(cr, cell_x + cell_w - 16,
                                plot_y + (plot_h - te.width) / 2);
            cairo_rotate(cr, M_PI / 2);
            cairo_move_to(cr, 0, 0);
            cairo_show_text(cr, ax.ylabel_right.c_str());
            cairo_restore(cr);
        }
    }

    /* xline / yline reference lines — clipped to the plot area; thin,
     * dashed black by default. The optional label sits at the far end
     * for visibility. */
    if (!ax.reflines.empty()) {
        cairo_save(cr);
        cairo_rectangle(cr, plot_x, plot_y, plot_w, plot_h);
        cairo_clip(cr);
        for (const auto &R : ax.reflines) {
            cairo_set_source_rgba(cr, R.r, R.g, R.b, 0.85);
            apply_line_style(cr, R.style, 1.0);
            if (R.vertical) {
                double sx = map_x(R.value, &ctx);
                cairo_move_to(cr, sx, plot_y);
                cairo_line_to(cr, sx, plot_y + plot_h);
                cairo_stroke(cr);
                if (!R.label.empty()) {
                    cairo_set_dash(cr, nullptr, 0, 0);
                    set_sans_font(cr, 11);
                    draw_text(cr, sx + 4, plot_y + 14, R.label);
                }
            } else {
                double sy = map_y(R.value, &ctx);
                cairo_move_to(cr, plot_x,          sy);
                cairo_line_to(cr, plot_x + plot_w, sy);
                cairo_stroke(cr);
                if (!R.label.empty()) {
                    cairo_set_dash(cr, nullptr, 0, 0);
                    set_sans_font(cr, 11);
                    draw_text(cr, plot_x + plot_w - 70, sy - 4, R.label);
                }
            }
        }
        cairo_restore(cr);
    }

    /* Text annotations — clipped to plot area, drawn over the series.
     * MATLAB's text() left-anchors the string at the baseline, so we
     * shift down by the font ascent (~10px at 12pt) to match. */
    if (!ax.annotations.empty()) {
        cairo_save(cr);
        cairo_rectangle(cr, plot_x, plot_y, plot_w, plot_h);
        cairo_clip(cr);
        cairo_set_source_rgb(cr, 0, 0, 0);
        set_sans_font(cr, 12);
        for (const auto &a : ax.annotations) {
            cairo_text_extents_t te;
            cairo_text_extents(cr, a.s.c_str(), &te);
            draw_text(cr, map_x(a.x, &ctx), map_y(a.y, &ctx), a.s);
            (void)te;  /* extents unused — left-baseline anchor */
        }
        cairo_restore(cr);
    }

    if (ax.colorbar) draw_colorbar(cr, ax, plot_x, plot_y, plot_w, plot_h);

    draw_legend(cr, ax, plot_x, plot_y, plot_w, plot_h);
}

void draw_figure(cairo_t *cr, const Figure &fig) {
    /* White background. */
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);

    /* Subplot grid layout. Cells tile the figure with a small inter-cell
     * gap; each axes paints into its own cell rect. */
    const double gap   = 8.0;
    const int    rows  = std::max(1, fig.rows);
    const int    cols  = std::max(1, fig.cols);
    const double cell_w = (fig.width_px  - gap * (cols - 1)) / cols;
    const double cell_h = (fig.height_px - gap * (rows - 1)) / rows;

    const int n_axes = static_cast<int>(fig.axes_list.size());
    for (int i = 0; i < n_axes && i < rows * cols; ++i) {
        int row = i / cols;
        int col = i % cols;
        double cx = col * (cell_w + gap);
        double cy = row * (cell_h + gap);
        const Axes &ax = *fig.axes_list[i];
        if (ax.is_3d) draw_axes_3d(cr, ax, cx, cy, cell_w, cell_h);
        else          draw_axes   (cr, ax, cx, cy, cell_w, cell_h);
    }
}

cairo_surface_t *make_surface(Format fmt, ByteSink *sink, const Figure &fig,
                              const char *file_path) {
    switch (fmt) {
        case Format::Png:
            return cairo_image_surface_create(
                CAIRO_FORMAT_ARGB32, fig.width_px, fig.height_px);
        case Format::Svg:
            return file_path
                ? cairo_svg_surface_create(file_path, fig.width_px, fig.height_px)
                : cairo_svg_surface_create_for_stream(
                      sink_write, sink, fig.width_px, fig.height_px);
        case Format::Pdf:
            return file_path
                ? cairo_pdf_surface_create(file_path, fig.width_px, fig.height_px)
                : cairo_pdf_surface_create_for_stream(
                      sink_write, sink, fig.width_px, fig.height_px);
    }
    return nullptr;
}

}  // namespace

RenderResult render(const Figure &fig, Format fmt) {
    ByteSink sink;
    cairo_surface_t *surface = make_surface(fmt, &sink, fig, nullptr);
    if (!surface || cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        if (surface) cairo_surface_destroy(surface);
        return {};
    }
    cairo_t *cr = cairo_create(surface);
    draw_figure(cr, fig);
    cairo_destroy(cr);
    if (fmt == Format::Png)
        cairo_surface_write_to_png_stream(surface, sink_write, &sink);
    else
        cairo_surface_finish(surface);
    cairo_surface_destroy(surface);
    return sink_to_result(sink);
}

int render_to_file(const Figure &fig, Format fmt, const char *path) {
    cairo_surface_t *surface = make_surface(fmt, nullptr, fig, path);
    if (!surface || cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        if (surface) cairo_surface_destroy(surface);
        return -2;
    }
    cairo_t *cr = cairo_create(surface);
    draw_figure(cr, fig);
    cairo_destroy(cr);
    int rc = 0;
    if (fmt == Format::Png) {
        if (cairo_surface_write_to_png(surface, path) != CAIRO_STATUS_SUCCESS)
            rc = -2;
    } else {
        cairo_surface_finish(surface);
    }
    cairo_surface_destroy(surface);
    return rc;
}

}  // namespace matlab_plot
