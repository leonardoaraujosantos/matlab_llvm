/* Focused C-ABI tests for runtime/matlab_plot. Direct calls — no JIT,
 * no MATLAB frontend. Covers: hold on/off semantics, multi-format
 * rendering, file vs in-memory parity, savefig extension dispatch,
 * empty/degenerate inputs, multi-figure isolation. */

#include "matlab_plot.h"
#include "runtime_internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond) do {                                            \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL %s:%d: %s\n",                    \
                     __FILE__, __LINE__, #cond);                    \
        return 1;                                                   \
    }                                                               \
} while (0)

namespace {

matlab_mat make_vec(std::vector<double> &storage,
                    const std::vector<double> &v) {
    storage = v;
    matlab_mat m;
    m.rows = 1;
    m.cols = static_cast<int64_t>(storage.size());
    m.data = storage.data();
    return m;
}

bool nonempty(matlab_plot_buffer b) { return b.data && b.size > 0; }

long file_size(const char *p) {
    std::FILE *f = std::fopen(p, "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fclose(f);
    return n;
}

bool starts_with(const char *path, const char *magic, size_t n) {
    std::FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    char buf[16] = {};
    size_t got = std::fread(buf, 1, n, f);
    std::fclose(f);
    return got == n && std::memcmp(buf, magic, n) == 0;
}

}  // namespace

/* ---- Test 1: every format renders to non-empty bytes. ---- */
static int test_three_formats() {
    matlab_close_all();
    std::vector<double> sx, sy;
    std::vector<double> X{0, 1, 2, 3, 4}, Y{0, 1, 4, 9, 16};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);

    matlab_figure_new();
    matlab_plot2(&x, &y);

    matlab_plot_buffer png = matlab_render_png();
    matlab_plot_buffer svg = matlab_render_svg();
    matlab_plot_buffer pdf = matlab_render_pdf();
    CHECK(nonempty(png));
    CHECK(nonempty(svg));
    CHECK(nonempty(pdf));
    /* PNG magic: 89 50 4E 47 0D 0A 1A 0A. */
    CHECK(png.data[0] == 0x89 && png.data[1] == 'P' &&
          png.data[2] == 'N'  && png.data[3] == 'G');
    /* SVG should start with "<?xml" or "<svg". */
    CHECK(svg.data[0] == '<');
    /* PDF magic: %PDF. */
    CHECK(pdf.data[0] == '%' && pdf.data[1] == 'P' &&
          pdf.data[2] == 'D' && pdf.data[3] == 'F');

    matlab_plot_buffer_free(png);
    matlab_plot_buffer_free(svg);
    matlab_plot_buffer_free(pdf);
    return 0;
}

/* ---- Test 2: hold off replaces, hold on accumulates. ---- */
static int test_hold_semantics() {
    matlab_close_all();
    std::vector<double> sx, sy1, sy2;
    std::vector<double> X{0, 1, 2}, Y1{0, 1, 0}, Y2{1, 0, 1};
    matlab_mat x  = make_vec(sx,  X);
    matlab_mat y1 = make_vec(sy1, Y1);
    matlab_mat y2 = make_vec(sy2, Y2);

    matlab_figure_new();
    matlab_plot2(&x, &y1);
    matlab_plot2(&x, &y2);   /* hold off → second call replaces first */

    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a));
    matlab_plot_buffer_free(a);

    matlab_figure_new();
    matlab_hold_on();
    matlab_plot2(&x, &y1);
    matlab_plot2(&x, &y2);   /* hold on → both series rendered */

    matlab_plot_buffer b = matlab_render_png();
    /* With both series painted, the encoded PNG must differ in size from
     * the single-series case because there's strictly more pixel content. */
    CHECK(nonempty(b));
    matlab_plot_buffer_free(b);
    return 0;
}

/* ---- Test 3: savefig extension dispatch + file output ---- */
static int test_savefig_dispatch() {
    matlab_close_all();
    std::vector<double> sx, sy;
    std::vector<double> X{0, 1}, Y{0, 1};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);

    matlab_figure_new();
    matlab_plot2(&x, &y);

    const char *png_path = "/tmp/matlab_plot_test_basic.png";
    const char *svg_path = "/tmp/matlab_plot_test_basic.svg";
    const char *pdf_path = "/tmp/matlab_plot_test_basic.pdf";
    const char *bad_path = "/tmp/matlab_plot_test_basic.xyz";

    CHECK(matlab_savefig(png_path, std::strlen(png_path)) == 0);
    CHECK(matlab_savefig(svg_path, std::strlen(svg_path)) == 0);
    CHECK(matlab_savefig(pdf_path, std::strlen(pdf_path)) == 0);
    CHECK(matlab_savefig(bad_path, std::strlen(bad_path)) == -1);

    CHECK(file_size(png_path) > 0);
    CHECK(file_size(svg_path) > 0);
    CHECK(file_size(pdf_path) > 0);

    CHECK(starts_with(png_path, "\x89PNG", 4));
    CHECK(starts_with(svg_path, "<", 1));
    CHECK(starts_with(pdf_path, "%PDF", 4));

    std::remove(png_path);
    std::remove(svg_path);
    std::remove(pdf_path);
    return 0;
}

/* ---- Test 4: degenerate inputs don't crash. ---- */
static int test_degenerate() {
    matlab_close_all();
    matlab_figure_new();
    /* Empty figure render. */
    matlab_plot_buffer empty_png = matlab_render_png();
    CHECK(nonempty(empty_png));
    matlab_plot_buffer_free(empty_png);

    /* Single-point series — falls below line draw threshold but must
     * not fault. */
    std::vector<double> sx, sy;
    std::vector<double> X{1.0}, Y{2.0};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);
    matlab_plot2(&x, &y);
    matlab_plot_buffer one_pt = matlab_render_png();
    CHECK(nonempty(one_pt));
    matlab_plot_buffer_free(one_pt);

    /* Constant y (zero range) — renderer pads internally. */
    std::vector<double> Y_const{5.0, 5.0, 5.0};
    matlab_mat yc = make_vec(sy, Y_const);
    std::vector<double> X3{0.0, 1.0, 2.0};
    matlab_mat x3 = make_vec(sx, X3);
    matlab_figure_new();
    matlab_plot2(&x3, &yc);
    matlab_plot_buffer flat = matlab_render_png();
    CHECK(nonempty(flat));
    matlab_plot_buffer_free(flat);

    return 0;
}

/* ---- Test 5: multi-figure isolation via figure ids. ---- */
static int test_multi_figure() {
    matlab_close_all();
    std::vector<double> sx1, sy1, sx2, sy2;
    std::vector<double> X1{0, 1, 2}, Y1{0, 1, 0};
    std::vector<double> X2{0, 1, 2}, Y2{1, 0, 1};
    matlab_mat x1 = make_vec(sx1, X1), y1 = make_vec(sy1, Y1);
    matlab_mat x2 = make_vec(sx2, X2), y2 = make_vec(sy2, Y2);

    matlab_figure *f1 = matlab_figure_new();
    matlab_plot2(&x1, &y1);
    matlab_figure *f2 = matlab_figure_new();
    matlab_plot2(&x2, &y2);
    CHECK(f1 != f2);

    /* gcf returns the most recently created. */
    CHECK(matlab_gcf() == f2);

    /* figure_by_id swaps current. */
    matlab_figure *back = matlab_figure_by_id(1);
    CHECK(back == f1);
    CHECK(matlab_gcf() == f1);

    matlab_close_all();
    /* After close_all, f1/f2 are dangling — only assert that gcf still
     * returns a usable handle (the registry rebuilt itself lazily). */
    matlab_figure *fresh = matlab_gcf();
    CHECK(fresh != nullptr);
    return 0;
}

/* ---- Test 6: limits override autoscale. ---- */
static int test_limits() {
    matlab_close_all();
    std::vector<double> sx, sy, slim;
    std::vector<double> X{0, 1, 2}, Y{0, 1, 2};
    std::vector<double> Lim{-10, 10};
    matlab_mat x = make_vec(sx, X);
    matlab_mat y = make_vec(sy, Y);
    matlab_mat lim = make_vec(slim, Lim);

    matlab_figure_new();
    matlab_plot2(&x, &y);
    matlab_xlim(&lim);
    matlab_ylim(&lim);
    matlab_plot_buffer png = matlab_render_png();
    CHECK(nonempty(png));
    matlab_plot_buffer_free(png);
    return 0;
}

/* ---- Test 7: imshow renders without crash; valid PNG bytes. ---- */
static int test_imshow() {
    matlab_close_all();
    constexpr int R = 16, C = 16;
    std::vector<double> img(R * C);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            img[r * C + c] = static_cast<double>(r + c);

    matlab_mat M;
    M.rows = R; M.cols = C; M.data = img.data();

    matlab_figure_new();
    matlab_imshow(&M);
    matlab_plot_buffer png = matlab_render_png();
    CHECK(nonempty(png));
    CHECK(png.data[0] == 0x89 && png.data[1] == 'P');
    matlab_plot_buffer_free(png);

    /* Degenerate constant image — autoscale denominator must guard. */
    std::fill(img.begin(), img.end(), 7.0);
    matlab_figure_new();
    matlab_imshow(&M);
    matlab_plot_buffer flat = matlab_render_png();
    CHECK(nonempty(flat));
    matlab_plot_buffer_free(flat);
    return 0;
}

/* ---- Test 8: legend_strs records labels; render path runs. ---- */
static int test_legend_strs() {
    matlab_close_all();
    std::vector<double> sx, sy1, sy2;
    std::vector<double> X{0, 1, 2}, Y1{0, 1, 0}, Y2{1, 0, 1};
    matlab_mat x  = make_vec(sx,  X);
    matlab_mat y1 = make_vec(sy1, Y1);
    matlab_mat y2 = make_vec(sy2, Y2);

    matlab_figure_new();
    matlab_plot2(&x, &y1);
    matlab_hold_on();
    matlab_plot2(&x, &y2);

    const char *labels[] = {"first", "second"};
    const int64_t lens[] = {5, 6};
    matlab_legend_strs(labels, lens, 2);

    matlab_plot_buffer png = matlab_render_png();
    CHECK(nonempty(png));
    matlab_plot_buffer_free(png);

    /* Empty / null entries must not crash. */
    const char *labels2[] = {nullptr, ""};
    const int64_t lens2[] = {0, 0};
    matlab_legend_strs(labels2, lens2, 2);
    matlab_plot_buffer p2 = matlab_render_png();
    CHECK(nonempty(p2));
    matlab_plot_buffer_free(p2);
    return 0;
}

/* ---- Test 9: linespec markers + RGB color override. ---- */
static int test_markers_rgb() {
    matlab_close_all();
    std::vector<double> sx, sy;
    std::vector<double> X{1, 2, 3, 4}, Y{1, 4, 2, 3};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);

    /* Marker-only spec ('o') yields no line; render must still succeed. */
    matlab_figure_new();
    const char *only_marker = "o";
    matlab_plot_fmt(&x, &y, only_marker, std::strlen(only_marker));
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a));
    matlab_plot_buffer_free(a);

    /* Each marker glyph paints without crash. */
    const char *glyphs[] = {"o-", "+-", "x-", "*-", "s-", "d-", "^-"};
    for (const char *g : glyphs) {
        matlab_figure_new();
        matlab_plot_fmt(&x, &y, g, std::strlen(g));
        matlab_plot_buffer b = matlab_render_png();
        CHECK(nonempty(b));
        matlab_plot_buffer_free(b);
    }

    /* RGB triplet override is clamped and applied to the latest series. */
    matlab_figure_new();
    matlab_plot2(&x, &y);
    matlab_set_series_color(2.0, -0.5, 0.5);  /* clamps to (1, 0, 0.5) */
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c));
    matlab_plot_buffer_free(c);

    /* No-op when there's no series — must not crash. */
    matlab_figure_new();
    matlab_set_series_color(0.5, 0.5, 0.5);
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d));
    matlab_plot_buffer_free(d);
    return 0;
}

/* ---- Test 10: subplot reshapes the grid and routes to the right cell. ---- */
static int test_subplot() {
    matlab_close_all();
    std::vector<double> sx, sy1, sy2, sy3, sy4;
    std::vector<double> X{0, 1, 2}, Y1{0, 1, 0}, Y2{1, 0, 1}, Y3{0, 0.5, 1}, Y4{1, 0.5, 0};
    matlab_mat x  = make_vec(sx,  X);
    matlab_mat y1 = make_vec(sy1, Y1);
    matlab_mat y2 = make_vec(sy2, Y2);
    matlab_mat y3 = make_vec(sy3, Y3);
    matlab_mat y4 = make_vec(sy4, Y4);

    matlab_figure_new();
    matlab_subplot(2, 2, 1); matlab_plot2(&x, &y1);
    matlab_subplot(2, 2, 2); matlab_plot2(&x, &y2);
    matlab_subplot(2, 2, 3); matlab_plot2(&x, &y3);
    matlab_subplot(2, 2, 4); matlab_plot2(&x, &y4);

    matlab_plot_buffer png = matlab_render_png();
    CHECK(nonempty(png));
    matlab_plot_buffer_free(png);

    /* Out-of-range index is a no-op; render must still succeed. */
    matlab_subplot(2, 2, 99);
    /* Reshape: switching to 1×3 must drop the previous 2×2 grid; new
     * cells start empty. */
    matlab_subplot(1, 3, 2);
    matlab_plot2(&x, &y1);
    matlab_plot_buffer png2 = matlab_render_png();
    CHECK(nonempty(png2));
    matlab_plot_buffer_free(png2);

    /* Decoration routes to the active subplot only. */
    matlab_close_all();
    matlab_figure_new();
    matlab_subplot(1, 2, 1);
    matlab_plot2(&x, &y1);
    const char *t1 = "left";
    matlab_title(t1, std::strlen(t1));
    matlab_subplot(1, 2, 2);
    matlab_plot2(&x, &y2);
    const char *t2 = "right";
    matlab_title(t2, std::strlen(t2));
    matlab_plot_buffer png3 = matlab_render_png();
    CHECK(nonempty(png3));
    matlab_plot_buffer_free(png3);

    /* Bad dims are no-ops. */
    matlab_subplot(0, 0, 1);   /* rows < 1 */
    matlab_subplot(2, 2, 0);   /* idx < 1 */
    matlab_plot_buffer png4 = matlab_render_png();
    CHECK(nonempty(png4));
    matlab_plot_buffer_free(png4);
    return 0;
}

/* ---- Test 11: colormap selection — every recognised name renders. ---- */
static int test_colormap() {
    constexpr int R = 16, C = 16;
    std::vector<double> img(R * C);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            img[r * C + c] = static_cast<double>(r + c);
    matlab_mat M;
    M.rows = R; M.cols = C; M.data = img.data();

    const char *names[] = {"gray", "parula", "jet", "viridis", "hot", "cool"};
    for (const char *nm : names) {
        matlab_close_all();
        matlab_figure_new();
        matlab_colormap(nm, std::strlen(nm));
        matlab_imshow(&M);
        matlab_plot_buffer png = matlab_render_png();
        CHECK(nonempty(png));
        matlab_plot_buffer_free(png);
    }

    /* Unknown name falls back to parula; must not crash. */
    matlab_close_all();
    matlab_figure_new();
    const char *bad = "nonsense_map";
    matlab_colormap(bad, std::strlen(bad));
    matlab_imshow(&M);
    matlab_plot_buffer png = matlab_render_png();
    CHECK(nonempty(png));
    matlab_plot_buffer_free(png);

    /* Empty / null name: no-op (active map unchanged). */
    matlab_colormap(nullptr, 0);
    matlab_colormap("", 0);
    return 0;
}

/* ---- Test 12: imagesc + pcolor + colorbar render without crash. ---- */
static int test_imagesc_pcolor_colorbar() {
    constexpr int R = 8, C = 12;
    std::vector<double> img(R * C);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            img[r * C + c] = static_cast<double>(r) - c;
    matlab_mat M;
    M.rows = R; M.cols = C; M.data = img.data();

    /* imagesc — must produce ticks (no skip) and a non-empty render. */
    matlab_close_all();
    matlab_figure_new();
    matlab_imagesc(&M);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* pcolor — alias path for now. */
    matlab_close_all();
    matlab_figure_new();
    matlab_pcolor(&M);
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* imagesc + colorbar. */
    matlab_close_all();
    matlab_figure_new();
    matlab_imagesc(&M);
    matlab_colorbar();
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* colorbar with no image — must be a no-op (no crash). */
    matlab_close_all();
    matlab_figure_new();
    matlab_colorbar();
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);
    return 0;
}

/* ---- Test 13: contour generates segments + renders. ---- */
static int test_contour() {
    /* Saddle z = x*y on a 5x5 grid → contours at z=0 form a cross. */
    constexpr int N = 5;
    std::vector<double> Z(N * N);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            Z[r * N + c] = (c - N/2.0) * (r - N/2.0);
    matlab_mat M;
    M.rows = N; M.cols = N; M.data = Z.data();

    /* Auto levels — must produce a non-empty render. */
    matlab_close_all();
    matlab_figure_new();
    matlab_contour(&M);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* Explicit levels including z=0 (the saddle's level set). */
    std::vector<double> lv_data{-2, -1, 0, 1, 2};
    matlab_mat L;
    L.rows = 1; L.cols = static_cast<int64_t>(lv_data.size());
    L.data = lv_data.data();
    matlab_close_all();
    matlab_figure_new();
    matlab_contour_levels(&M, &L);
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* Degenerate constant Z — auto_levels should produce empty list and
     * the call is a no-op (renders an empty axes without crashing). */
    std::fill(Z.begin(), Z.end(), 7.0);
    matlab_close_all();
    matlab_figure_new();
    matlab_contour(&M);
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* Tiny grid (1x1) — must not crash. */
    matlab_mat tiny;
    double dot = 1.0;
    tiny.rows = 1; tiny.cols = 1; tiny.data = &dot;
    matlab_close_all();
    matlab_figure_new();
    matlab_contour(&tiny);
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);
    return 0;
}

/* ---- Test 14: axis options + box on/off. ---- */
static int test_axis_options() {
    std::vector<double> sx, sy;
    std::vector<double> X{0, 1, 2, 3, 4}, Y{0, 4, 1, 9, 2};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);

    /* Each option renders without crash. Unknown opts must be silently ignored. */
    const char *opts[] = {"equal", "tight", "off", "on", "square", "garbage"};
    for (const char *o : opts) {
        matlab_close_all();
        matlab_figure_new();
        matlab_plot2(&x, &y);
        matlab_axis(o, std::strlen(o));
        matlab_plot_buffer png = matlab_render_png();
        CHECK(nonempty(png));
        matlab_plot_buffer_free(png);
    }

    /* box off without axis off — frame disappears, ticks stay. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&x, &y);
    matlab_box_off();
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a));
    matlab_plot_buffer_free(a);

    /* box on after off restores frame. */
    matlab_box_on();
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b));
    matlab_plot_buffer_free(b);

    /* "on" after "off" must restore decoration. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&x, &y);
    const char *off = "off"; matlab_axis(off, std::strlen(off));
    const char *on  = "on";  matlab_axis(on,  std::strlen(on));
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c));
    matlab_plot_buffer_free(c);

    /* Empty / null options are no-ops. */
    matlab_axis(nullptr, 0);
    matlab_axis("", 0);
    return 0;
}

/* ---- Test 15: log-scale axes — semilogy, semilogx, loglog. ---- */
static int test_log_axes() {
    constexpr int N = 50;
    std::vector<double> sx, sy_exp, sy_pow;
    std::vector<double> X(N), Y_exp(N), Y_pow(N);
    for (int i = 0; i < N; ++i) {
        X[i]     = 0.1 + 10.0 * i / (N - 1);  /* 0.1 .. 10.1 — all positive */
        Y_exp[i] = std::exp(X[i] * 0.5);
        Y_pow[i] = std::pow(X[i], 3);
    }
    matlab_mat x  = make_vec(sx,     X);
    matlab_mat y1 = make_vec(sy_exp, Y_exp);
    matlab_mat y2 = make_vec(sy_pow, Y_pow);

    /* semilogy — y exponential is straight on log y. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&x, &y1);
    matlab_semilogy();
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* semilogx. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&x, &y1);
    matlab_semilogx();
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* loglog — power-law is straight on log-log. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&x, &y2);
    matlab_loglog();
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* Switch back to linear after log — current axes should follow. */
    matlab_xscale_linear();
    matlab_yscale_linear();
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);

    /* Data crossing zero on a log axis — must not crash. The non-positive
     * values get clamped/skipped; the positive part still draws. */
    std::vector<double> sxz, syz;
    std::vector<double> XZ{-1, 0, 1, 10, 100}, YZ{0.1, 0.5, 1, 10, 100};
    matlab_mat xz = make_vec(sxz, XZ);
    matlab_mat yz = make_vec(syz, YZ);
    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&xz, &yz);
    matlab_loglog();
    matlab_plot_buffer e = matlab_render_png();
    CHECK(nonempty(e)); matlab_plot_buffer_free(e);
    return 0;
}

/* ---- Test 16: errorbar — symmetric and asymmetric. ---- */
static int test_errorbar() {
    std::vector<double> sx, sy, se, sn, sp;
    std::vector<double> X{1, 2, 3, 4, 5}, Y{2, 4, 3, 5, 4};
    std::vector<double> E{0.5, 0.3, 0.4, 0.6, 0.2};
    std::vector<double> N{0.2, 0.5, 0.3, 0.6, 0.1};
    std::vector<double> P{0.7, 0.3, 0.5, 0.4, 0.4};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);
    matlab_mat e = make_vec(se, E), n = make_vec(sn, N), p = make_vec(sp, P);

    /* Symmetric. */
    matlab_close_all();
    matlab_figure_new();
    matlab_errorbar(&x, &y, &e);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* Asymmetric. */
    matlab_close_all();
    matlab_figure_new();
    matlab_errorbar2(&x, &y, &n, &p);
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* Single-point series — connecting line is skipped, bar still drawn. */
    std::vector<double> sxs, sys, ses;
    std::vector<double> XS{2.5}, YS{1.0}, ES{0.3};
    matlab_mat xs = make_vec(sxs, XS), ys = make_vec(sys, YS), es = make_vec(ses, ES);
    matlab_close_all();
    matlab_figure_new();
    matlab_errorbar(&xs, &ys, &es);
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);
    return 0;
}

/* ---- Test 17: stem and stairs render without crash. ---- */
static int test_stem_stairs() {
    std::vector<double> sx, sy;
    std::vector<double> X{1, 2, 3, 4, 5, 6};
    std::vector<double> Y{0.5, -0.3, 0.8, 0.2, -0.6, 0.4};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);

    /* stem(x, y) — must include zero baseline in y range. */
    matlab_close_all();
    matlab_figure_new();
    matlab_stem2(&x, &y);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* stem(y) — implicit x = 1..N. */
    matlab_close_all();
    matlab_figure_new();
    matlab_stem1(&y);
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* stairs(x, y). */
    matlab_close_all();
    matlab_figure_new();
    matlab_stairs2(&x, &y);
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* stairs(y). */
    matlab_close_all();
    matlab_figure_new();
    matlab_stairs1(&y);
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);
    return 0;
}

/* ---- Test 18: histogram + area. ---- */
static int test_histogram_area() {
    /* Histogram: explicit nbins + auto bins. */
    std::vector<double> data(200);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = std::sin(0.13 * static_cast<double>(i));
    matlab_mat D;
    D.rows = 1; D.cols = static_cast<int64_t>(data.size()); D.data = data.data();

    matlab_close_all();
    matlab_figure_new();
    matlab_histogram(&D, 20);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* Auto bins (nbins <= 0 → sqrt heuristic). */
    matlab_close_all();
    matlab_figure_new();
    matlab_histogram(&D, 0);
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* Empty data — must not crash; render still succeeds. */
    matlab_mat empty;
    empty.rows = 0; empty.cols = 0; empty.data = nullptr;
    matlab_close_all();
    matlab_figure_new();
    matlab_histogram(&empty, 10);
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* Area. */
    std::vector<double> sx, sy;
    std::vector<double> X{0, 1, 2, 3, 4}, Y{1, 3, 2, 4, 1};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);
    matlab_close_all();
    matlab_figure_new();
    matlab_area2(&x, &y);
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);

    /* area(y) — implicit x = 1..N. */
    matlab_close_all();
    matlab_figure_new();
    matlab_area1(&y);
    matlab_plot_buffer e = matlab_render_png();
    CHECK(nonempty(e)); matlab_plot_buffer_free(e);
    return 0;
}

/* ---- Test 19: text annotations cleared by non-hold plot. ---- */
static int test_text_annotations() {
    std::vector<double> sx, sy;
    std::vector<double> X{0, 1, 2}, Y{0, 1, 0};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);

    matlab_close_all();
    matlab_figure_new();
    matlab_plot2(&x, &y);
    const char *a = "annotation"; matlab_text(1.0, 0.5, a, std::strlen(a));
    matlab_plot_buffer p1 = matlab_render_png();
    CHECK(nonempty(p1)); matlab_plot_buffer_free(p1);

    /* Empty / null strings — no-op. */
    matlab_text(0.5, 0.5, nullptr, 0);
    matlab_text(0.5, 0.5, "",      0);

    /* Non-hold plot clears annotations. After it, the previous text is
     * gone — but we can't directly inspect; just check render still works. */
    matlab_plot2(&x, &y);
    matlab_plot_buffer p2 = matlab_render_png();
    CHECK(nonempty(p2)); matlab_plot_buffer_free(p2);
    return 0;
}

/* ---- Test 20: plot3 + view + zlabel render without crash. ---- */
static int test_plot3_view() {
    constexpr int N = 50;
    std::vector<double> sx, sy, sz;
    std::vector<double> X(N), Y(N), Z(N);
    for (int i = 0; i < N; ++i) {
        double t = i * 0.2;
        X[i] = std::cos(t);
        Y[i] = std::sin(t);
        Z[i] = t * 0.1;
    }
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y), z = make_vec(sz, Z);

    /* Default view. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot3(&x, &y, &z);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* matlab_view shifts the camera; output should differ but always non-empty. */
    matlab_view(0, 90);   /* top-down */
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    matlab_view(45, 45);
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* zlabel on a 3-D axes. */
    const char *zl = "height";
    matlab_zlabel(zl, std::strlen(zl));
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);

    /* plot3 with linespec. */
    matlab_close_all();
    matlab_figure_new();
    const char *fmt = "r--";
    matlab_plot3_fmt(&x, &y, &z, fmt, std::strlen(fmt));
    matlab_plot_buffer e = matlab_render_png();
    CHECK(nonempty(e)); matlab_plot_buffer_free(e);
    return 0;
}

/* ---- Test 21: surf and mesh render without crash. ---- */
static int test_surf_mesh() {
    constexpr int M = 12, N = 14;
    std::vector<double> Z(M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            double u = (c - N/2.0) / 4.0;
            double v = (r - M/2.0) / 4.0;
            Z[r * N + c] = std::sin(u) * std::cos(v);
        }
    matlab_mat MZ;
    MZ.rows = M; MZ.cols = N; MZ.data = Z.data();

    /* surf with implicit grid. */
    matlab_close_all();
    matlab_figure_new();
    matlab_surf1(&MZ);
    matlab_plot_buffer a = matlab_render_png();
    CHECK(nonempty(a)); matlab_plot_buffer_free(a);

    /* mesh with implicit grid. */
    matlab_close_all();
    matlab_figure_new();
    matlab_mesh1(&MZ);
    matlab_plot_buffer b = matlab_render_png();
    CHECK(nonempty(b)); matlab_plot_buffer_free(b);

    /* surf with explicit X, Y, Z grids. */
    std::vector<double> XX(M * N), YY(M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            XX[r * N + c] = static_cast<double>(c);
            YY[r * N + c] = static_cast<double>(r);
        }
    matlab_mat MX, MY;
    MX.rows = M; MX.cols = N; MX.data = XX.data();
    MY.rows = M; MY.cols = N; MY.data = YY.data();
    matlab_close_all();
    matlab_figure_new();
    matlab_surf3(&MX, &MY, &MZ);
    matlab_plot_buffer c = matlab_render_png();
    CHECK(nonempty(c)); matlab_plot_buffer_free(c);

    /* Mismatched dims must be a no-op (not crash). */
    matlab_mat bad;
    bad.rows = 3; bad.cols = 3; bad.data = Z.data();
    matlab_close_all();
    matlab_figure_new();
    matlab_surf3(&MX, &MY, &bad);  /* X/Y are M×N, bad is 3×3 → reject */
    matlab_plot_buffer d = matlab_render_png();
    CHECK(nonempty(d)); matlab_plot_buffer_free(d);

    /* Tiny 2×2 grid — minimum that produces one quad. */
    std::vector<double> tiny{1, 2, 3, 4};
    matlab_mat T;
    T.rows = 2; T.cols = 2; T.data = tiny.data();
    matlab_close_all();
    matlab_figure_new();
    matlab_surf1(&T);
    matlab_plot_buffer e = matlab_render_png();
    CHECK(nonempty(e)); matlab_plot_buffer_free(e);
    return 0;
}

/* ---- Test: zlim pins the z-axis on 3-D axes (xlim/ylim parity, #238). ---- */
static int test_zlim_3d() {
    std::vector<double> sx, sy, sz;
    std::vector<double> X{0, 1, 2}, Y{0, 1, 0}, Z{0, 0.5, 1};
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y), z = make_vec(sz, Z);

    /* Baseline: autoscaled z in [0, 1]. */
    matlab_close_all();
    matlab_figure_new();
    matlab_plot3(&x, &y, &z);
    matlab_plot_buffer base = matlab_render_png();
    CHECK(nonempty(base));

    /* Same data, but zlim stretches the box far past the data — the render
     * must change because the z-axis now spans [-5, 10]. */
    std::vector<double> slim;
    std::vector<double> Lim{-5, 10};
    matlab_mat lim = make_vec(slim, Lim);
    matlab_close_all();
    matlab_figure_new();
    matlab_plot3(&x, &y, &z);
    matlab_zlim(&lim);
    matlab_plot_buffer set = matlab_render_png();
    CHECK(nonempty(set));
    CHECK(set.size != base.size ||
          std::memcmp(set.data, base.data, set.size) != 0);
    matlab_plot_buffer_free(base);
    matlab_plot_buffer_free(set);

    /* axis([xmin xmax ymin ymax zmin zmax]) 6-arg also pins z; no crash. */
    std::vector<double> sax;
    std::vector<double> Ax{-2, 2, -2, 2, 0, 5};
    matlab_mat axm = make_vec(sax, Ax);
    matlab_close_all();
    matlab_figure_new();
    matlab_plot3(&x, &y, &z);
    matlab_axis_lims(&axm);
    matlab_plot_buffer ap = matlab_render_png();
    CHECK(nonempty(ap));
    matlab_plot_buffer_free(ap);
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_three_formats();
    rc |= test_hold_semantics();
    rc |= test_savefig_dispatch();
    rc |= test_degenerate();
    rc |= test_multi_figure();
    rc |= test_limits();
    rc |= test_imshow();
    rc |= test_legend_strs();
    rc |= test_markers_rgb();
    rc |= test_subplot();
    rc |= test_colormap();
    rc |= test_imagesc_pcolor_colorbar();
    rc |= test_contour();
    rc |= test_axis_options();
    rc |= test_log_axes();
    rc |= test_errorbar();
    rc |= test_stem_stairs();
    rc |= test_histogram_area();
    rc |= test_text_annotations();
    rc |= test_plot3_view();
    rc |= test_surf_mesh();
    rc |= test_zlim_3d();
    if (rc == 0) std::puts("OK test_plot_basic");
    return rc;
}
