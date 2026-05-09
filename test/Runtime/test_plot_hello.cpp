/* Standalone smoke test for the matlab_plot runtime. Builds a sine wave
 * directly via the C ABI, renders to PNG/SVG/PDF on disk, and reports
 * file sizes. Does NOT exercise the JIT path; just confirms Cairo links
 * and the draw pipeline emits non-empty output.
 *
 * Build: enabled when -DMATLAB_LLVM_WITH_PLOT=ON. Run produces
 *   /tmp/matlab_plot_hello.png|.svg|.pdf */

#include "matlab_plot.h"
#include "runtime_internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

matlab_mat *make_vec(const std::vector<double> &v) {
    auto *m = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
    m->rows = 1;
    m->cols = static_cast<int64_t>(v.size());
    m->data = static_cast<double *>(std::malloc(v.size() * sizeof(double)));
    std::memcpy(m->data, v.data(), v.size() * sizeof(double));
    return m;
}

void free_vec(matlab_mat *m) { std::free(m->data); std::free(m); }

long file_size(const char *p) {
    std::FILE *f = std::fopen(p, "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fclose(f);
    return n;
}

}  // namespace

int main() {
    int rc_total = 0;

    /* ---- 1. sine — line plot, linespec, title, grid. ---- */
    {
        constexpr int N = 200;
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            x[i] = i * 0.05;
            y[i] = std::sin(x[i]);
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y);
        matlab_figure_new();
        const char *fmt = "r--";
        matlab_plot_fmt(X, Y, fmt, std::strlen(fmt));
        const char *t = "sin(x)"; matlab_title (t, std::strlen(t));
        const char *xl = "x";     matlab_xlabel(xl, std::strlen(xl));
        const char *yl = "sin(x)";matlab_ylabel(yl, std::strlen(yl));
        matlab_grid_on();

        const char *paths[] = {
            "/tmp/matlab_plot_hello.png",
            "/tmp/matlab_plot_hello.svg",
            "/tmp/matlab_plot_hello.pdf",
        };
        for (const char *p : paths) {
            int rc = matlab_savefig(p, std::strlen(p));
            long sz = file_size(p);
            std::printf("%-30s rc=%d size=%ld\n", p, rc, sz);
            if (rc != 0 || sz <= 0) rc_total = 1;
        }
        free_vec(X); free_vec(Y);
    }

    /* ---- 2. multi-series via hold on. ---- */
    {
        constexpr int N = 200;
        std::vector<double> x(N), y1(N), y2(N);
        for (int i = 0; i < N; ++i) {
            x[i]  = i * 0.05;
            y1[i] = std::sin(x[i]);
            y2[i] = std::cos(x[i]);
        }
        matlab_mat *X = make_vec(x), *Y1 = make_vec(y1), *Y2 = make_vec(y2);
        matlab_figure_new();
        const char *blue = "b-";    matlab_plot_fmt(X, Y1, blue, std::strlen(blue));
        matlab_hold_on();
        const char *red  = "r--";   matlab_plot_fmt(X, Y2, red,  std::strlen(red));
        const char *t = "sin and cos"; matlab_title(t, std::strlen(t));
        matlab_grid_on();
        const char *p = "/tmp/matlab_plot_multi.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y1); free_vec(Y2);
    }

    /* ---- 3. bar chart. ---- */
    {
        std::vector<double> x{1, 2, 3, 4, 5, 6}, y{3, 7, 2, 8, 5, 4};
        matlab_mat *X = make_vec(x), *Y = make_vec(y);
        matlab_figure_new();
        matlab_bar2(X, Y);
        const char *t = "Quarterly counts"; matlab_title(t, std::strlen(t));
        matlab_grid_on();
        const char *p = "/tmp/matlab_plot_bar.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y);
    }

    /* ---- 4. scatter. ---- */
    {
        constexpr int N = 80;
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            x[i] = 10.0 * i / (N - 1);
            /* deterministic pseudo-noise so the test is reproducible */
            double n = 0.3 * std::sin(13.7 * x[i] + 2.1);
            y[i] = 0.5 * x[i] + n;
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y);
        matlab_figure_new();
        matlab_scatter(X, Y);
        const char *t = "Noisy linear scatter"; matlab_title(t, std::strlen(t));
        matlab_grid_on();
        const char *p = "/tmp/matlab_plot_scatter.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y);
    }

    /* ---- 5. multi-series with legend. ---- */
    {
        constexpr int N = 200;
        std::vector<double> x(N), y1(N), y2(N);
        for (int i = 0; i < N; ++i) {
            x[i]  = i * 0.05;
            y1[i] = std::sin(x[i]);
            y2[i] = std::cos(x[i]);
        }
        matlab_mat *X = make_vec(x), *Y1 = make_vec(y1), *Y2 = make_vec(y2);
        matlab_figure_new();
        const char *blue = "b-";  matlab_plot_fmt(X, Y1, blue, std::strlen(blue));
        matlab_hold_on();
        const char *red  = "r--"; matlab_plot_fmt(X, Y2, red,  std::strlen(red));
        const char *t = "Legend test"; matlab_title(t, std::strlen(t));
        matlab_grid_on();

        const char *labels[] = {"sin(x)", "cos(x)"};
        const int64_t lens[] = {6, 6};
        matlab_legend_strs(labels, lens, 2);

        const char *p = "/tmp/matlab_plot_legend.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y1); free_vec(Y2);
    }

    /* ---- markers + RGB color triplet ---- */
    {
        std::vector<double> x{1, 2, 3, 4, 5, 6};
        std::vector<double> y1{1, 4, 2, 5, 3, 6};
        std::vector<double> y2{6, 3, 5, 2, 4, 1};
        std::vector<double> y3{2, 3, 4, 5, 6, 7};
        matlab_mat *X = make_vec(x);
        matlab_mat *Y1 = make_vec(y1), *Y2 = make_vec(y2), *Y3 = make_vec(y3);
        matlab_figure_new();
        const char *fmt1 = "ko-";   matlab_plot_fmt(X, Y1, fmt1, std::strlen(fmt1));
        matlab_hold_on();
        const char *fmt2 = "rs--";  matlab_plot_fmt(X, Y2, fmt2, std::strlen(fmt2));
        const char *fmt3 = "d";     matlab_plot_fmt(X, Y3, fmt3, std::strlen(fmt3));
        matlab_set_series_color(0.49, 0.18, 0.56);
        const char *t = "Markers + RGB"; matlab_title(t, std::strlen(t));
        matlab_grid_on();
        const char *labels[] = {"line+circle", "dashed+square", "diamonds"};
        const int64_t lens[] = {11, 13, 8};
        matlab_legend_strs(labels, lens, 3);
        const char *p = "/tmp/matlab_plot_markers.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y1); free_vec(Y2); free_vec(Y3);
    }

    /* ---- 6. imshow — 2-D Gaussian bump, autoscaled grayscale. ---- */
    {
        constexpr int R = 64, C = 64;
        std::vector<double> img(R * C);
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                double dx = (c - C / 2.0) / (C / 4.0);
                double dy = (r - R / 2.0) / (R / 4.0);
                img[r * C + c] = std::exp(-(dx * dx + dy * dy));
            }
        }
        auto *M = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
        M->rows = R; M->cols = C;
        M->data = static_cast<double *>(std::malloc(R * C * sizeof(double)));
        std::memcpy(M->data, img.data(), R * C * sizeof(double));

        matlab_figure_new();
        matlab_imshow(M);
        const char *t = "imshow: Gaussian bump"; matlab_title(t, std::strlen(t));
        const char *p = "/tmp/matlab_plot_imshow.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        std::free(M->data); std::free(M);
    }

    /* ---- 7. subplot 2x2 — line, scatter, bar, image. ---- */
    {
        constexpr int N = 100;
        std::vector<double> x(N), s1(N), s2(N);
        for (int i = 0; i < N; ++i) {
            x[i]  = i * 0.1;
            s1[i] = std::sin(x[i]);
            s2[i] = std::cos(x[i]);
        }
        std::vector<double> bx{1, 2, 3, 4, 5}, by{4, 7, 2, 8, 5};
        constexpr int R = 32;
        std::vector<double> img(R * R);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < R; ++c) {
                double dx = (c - R/2.0)/(R/4.0), dy = (r - R/2.0)/(R/4.0);
                img[r * R + c] = std::exp(-(dx*dx + dy*dy));
            }
        matlab_mat *X = make_vec(x), *S1 = make_vec(s1), *S2 = make_vec(s2);
        matlab_mat *BX = make_vec(bx), *BY = make_vec(by);
        auto *M = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
        M->rows = R; M->cols = R;
        M->data = static_cast<double *>(std::malloc(R*R*sizeof(double)));
        std::memcpy(M->data, img.data(), R*R*sizeof(double));

        matlab_figure_new();
        matlab_subplot(2, 2, 1);
        const char *blue = "b-"; matlab_plot_fmt(X, S1, blue, std::strlen(blue));
        const char *t1 = "sin"; matlab_title(t1, std::strlen(t1));

        matlab_subplot(2, 2, 2);
        matlab_scatter(X, S2);
        const char *t2 = "scatter cos"; matlab_title(t2, std::strlen(t2));

        matlab_subplot(2, 2, 3);
        matlab_bar2(BX, BY);
        const char *t3 = "bars"; matlab_title(t3, std::strlen(t3));

        matlab_subplot(2, 2, 4);
        matlab_imshow(M);
        const char *t4 = "imshow"; matlab_title(t4, std::strlen(t4));

        const char *p = "/tmp/matlab_plot_subplot.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(S1); free_vec(S2);
        free_vec(BX); free_vec(BY);
        std::free(M->data); std::free(M);
    }

    /* ---- 8. colormaps — same Gaussian, four maps. ---- */
    {
        constexpr int R = 64;
        std::vector<double> img(R * R);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < R; ++c) {
                double dx = (c - R/2.0)/(R/4.0), dy = (r - R/2.0)/(R/4.0);
                img[r * R + c] = std::exp(-(dx*dx + dy*dy));
            }
        auto build_M = [&]() {
            auto *M = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
            M->rows = R; M->cols = R;
            M->data = static_cast<double *>(std::malloc(R*R*sizeof(double)));
            std::memcpy(M->data, img.data(), R*R*sizeof(double));
            return M;
        };
        auto free_M = [](matlab_mat *M) { std::free(M->data); std::free(M); };

        matlab_figure_new();
        struct { int slot; const char *name; } cells[] = {
            {1, "parula"}, {2, "viridis"}, {3, "jet"}, {4, "hot"}
        };
        for (auto cell : cells) {
            matlab_subplot(2, 2, cell.slot);
            matlab_colormap(cell.name, std::strlen(cell.name));
            matlab_mat *M = build_M();
            matlab_imshow(M);
            matlab_title(cell.name, std::strlen(cell.name));
            free_M(M);
        }
        const char *p = "/tmp/matlab_plot_colormaps.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
    }

    /* ---- 9. imagesc + colorbar — heatmap with axes and color scale. ---- */
    {
        constexpr int R = 40, C = 60;
        std::vector<double> img(R * C);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                double dx = (c - C/2.0)/(C/4.0), dy = (r - R/2.0)/(R/4.0);
                img[r * C + c] = std::sin(3 * dx) * std::cos(2 * dy);
            }
        auto *M = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
        M->rows = R; M->cols = C;
        M->data = static_cast<double *>(std::malloc(R*C*sizeof(double)));
        std::memcpy(M->data, img.data(), R*C*sizeof(double));

        matlab_figure_new();
        const char *cm = "viridis"; matlab_colormap(cm, std::strlen(cm));
        matlab_imagesc(M);
        matlab_colorbar();
        const char *t = "imagesc + colorbar"; matlab_title(t, std::strlen(t));
        const char *xl = "column"; matlab_xlabel(xl, std::strlen(xl));
        const char *yl = "row";    matlab_ylabel(yl, std::strlen(yl));

        const char *p = "/tmp/matlab_plot_imagesc.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        std::free(M->data); std::free(M);
    }

    /* ---- 10. contour — marching squares of a peaks-like surface. ---- */
    {
        constexpr int R = 50, C = 50;
        std::vector<double> Z(R * C);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                double x = -3.0 + 6.0 * c / (C - 1);
                double y = -3.0 + 6.0 * r / (R - 1);
                Z[r * C + c] = 3 * (1 - x) * (1 - x) * std::exp(-x*x - (y+1)*(y+1))
                             - 10 * (x/5 - x*x*x - y*y*y*y*y) * std::exp(-x*x - y*y)
                             - (1.0/3) * std::exp(-(x+1)*(x+1) - y*y);
            }
        auto *M = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
        M->rows = R; M->cols = C;
        M->data = static_cast<double *>(std::malloc(R*C*sizeof(double)));
        std::memcpy(M->data, Z.data(), R*C*sizeof(double));

        matlab_figure_new();
        matlab_contour(M);
        const char *t = "contour(peaks)"; matlab_title(t, std::strlen(t));
        matlab_grid_on();
        const char *p = "/tmp/matlab_plot_contour.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        std::free(M->data); std::free(M);
    }

    /* ---- 11. axis options — equal aspect on a circle. ---- */
    {
        constexpr int N = 200;
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            double t = 2 * M_PI * i / (N - 1);
            x[i] = std::cos(t);
            y[i] = std::sin(t);
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y);
        matlab_figure_new();

        matlab_subplot(1, 2, 1);
        const char *fmt1 = "b-"; matlab_plot_fmt(X, Y, fmt1, std::strlen(fmt1));
        const char *t1 = "default aspect"; matlab_title(t1, std::strlen(t1));
        matlab_grid_on();

        matlab_subplot(1, 2, 2);
        matlab_plot_fmt(X, Y, fmt1, std::strlen(fmt1));
        const char *eq = "equal"; matlab_axis(eq, std::strlen(eq));
        const char *t2 = "axis equal"; matlab_title(t2, std::strlen(t2));
        matlab_grid_on();

        const char *p = "/tmp/matlab_plot_axis.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y);
    }

    /* ---- 12. log axes — exponential decay (linear vs semilogy) and
     *           power-law (loglog). ---- */
    {
        constexpr int N = 80;
        std::vector<double> x(N), y_exp(N), y_pow(N);
        for (int i = 0; i < N; ++i) {
            x[i]     = 0.1 + 100.0 * i / (N - 1);  /* 0.1 .. 100.1 */
            y_exp[i] = std::exp(-x[i] * 0.05) * 1000;
            y_pow[i] = std::pow(x[i], 2.5);
        }
        matlab_mat *X = make_vec(x), *YE = make_vec(y_exp), *YP = make_vec(y_pow);

        matlab_figure_new();
        matlab_subplot(2, 2, 1);
        const char *fmt = "b-"; matlab_plot_fmt(X, YE, fmt, std::strlen(fmt));
        const char *t1 = "linear: exp decay"; matlab_title(t1, std::strlen(t1));
        matlab_grid_on();

        matlab_subplot(2, 2, 2);
        matlab_plot_fmt(X, YE, fmt, std::strlen(fmt));
        matlab_semilogy();
        const char *t2 = "semilogy: exp decay"; matlab_title(t2, std::strlen(t2));
        matlab_grid_on();

        matlab_subplot(2, 2, 3);
        matlab_plot_fmt(X, YP, fmt, std::strlen(fmt));
        const char *t3 = "linear: power law"; matlab_title(t3, std::strlen(t3));
        matlab_grid_on();

        matlab_subplot(2, 2, 4);
        matlab_plot_fmt(X, YP, fmt, std::strlen(fmt));
        matlab_loglog();
        const char *t4 = "loglog: power law"; matlab_title(t4, std::strlen(t4));
        matlab_grid_on();

        const char *p = "/tmp/matlab_plot_log.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(YE); free_vec(YP);
    }

    /* ---- 13. errorbar — symmetric noise on a sin wave. ---- */
    {
        constexpr int N = 20;
        std::vector<double> x(N), y(N), e(N);
        for (int i = 0; i < N; ++i) {
            x[i] = 0.5 + 9.5 * i / (N - 1);
            y[i] = std::sin(x[i]);
            /* deterministic pseudo-noise for reproducibility */
            e[i] = 0.05 + 0.15 * std::abs(std::sin(7.3 * x[i] + 1.1));
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y), *E = make_vec(e);
        matlab_figure_new();
        matlab_errorbar(X, Y, E);
        const char *t = "errorbar(sin) ± noise"; matlab_title(t, std::strlen(t));
        const char *xl = "x"; matlab_xlabel(xl, std::strlen(xl));
        const char *yl = "y"; matlab_ylabel(yl, std::strlen(yl));
        matlab_grid_on();
        const char *p = "/tmp/matlab_plot_errorbar.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y); free_vec(E);
    }

    /* ---- 14. stem + stairs — discrete signal display. ---- */
    {
        constexpr int N = 30;
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            x[i] = i;
            y[i] = std::sin(0.4 * i) * std::exp(-0.05 * i);
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y);
        matlab_figure_new();

        matlab_subplot(2, 1, 1);
        matlab_stem2(X, Y);
        const char *t1 = "stem(x, y)"; matlab_title(t1, std::strlen(t1));
        matlab_grid_on();

        matlab_subplot(2, 1, 2);
        matlab_stairs2(X, Y);
        const char *t2 = "stairs(x, y)"; matlab_title(t2, std::strlen(t2));
        matlab_grid_on();

        const char *p = "/tmp/matlab_plot_stem_stairs.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y);
    }

    /* ---- 15. histogram + area. ---- */
    {
        constexpr int N = 1000;
        std::vector<double> samples(N);
        /* Pseudo-Gaussian via Box-Muller-ish deterministic mixing. */
        for (int i = 0; i < N; ++i) {
            double u1 = (i * 0.001 + 0.0001);
            double u2 = std::fmod(i * 0.7919 + 0.31, 1.0) + 1e-9;
            samples[i] = std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2);
        }
        matlab_mat *D = make_vec(samples);

        std::vector<double> x(60), y(60);
        for (int i = 0; i < 60; ++i) {
            x[i] = i * 0.1;
            y[i] = std::sin(x[i]) * std::exp(-0.05 * x[i]);
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y);

        matlab_figure_new();
        matlab_subplot(1, 2, 1);
        matlab_histogram(D, 30);
        const char *t1 = "histogram(N=1000, 30 bins)"; matlab_title(t1, std::strlen(t1));
        matlab_grid_on();

        matlab_subplot(1, 2, 2);
        matlab_area2(X, Y);
        const char *t2 = "area(damped sin)"; matlab_title(t2, std::strlen(t2));
        matlab_grid_on();

        const char *p = "/tmp/matlab_plot_hist_area.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(D); free_vec(X); free_vec(Y);
    }

    /* ---- 16. text annotations on a sin curve. ---- */
    {
        constexpr int N = 200;
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            x[i] = i * 0.05;
            y[i] = std::sin(x[i]);
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y);
        matlab_figure_new();
        const char *fmt = "b-"; matlab_plot_fmt(X, Y, fmt, std::strlen(fmt));
        const char *t = "text annotations"; matlab_title(t, std::strlen(t));
        const char *a1 = "first peak";   matlab_text(M_PI/2,           0.85, a1, std::strlen(a1));
        const char *a2 = "first trough"; matlab_text(3 * M_PI/2 - 0.4, -0.85, a2, std::strlen(a2));
        const char *a3 = "second peak";  matlab_text(5 * M_PI/2 - 0.4, 0.85, a3, std::strlen(a3));
        matlab_grid_on();
        const char *p = "/tmp/matlab_plot_text.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y);
    }

    /* ---- 17. plot3 — 3-D helix with the default view. ---- */
    {
        constexpr int N = 200;
        std::vector<double> x(N), y(N), z(N);
        for (int i = 0; i < N; ++i) {
            double t = i * 0.1;
            x[i] = std::cos(t);
            y[i] = std::sin(t);
            z[i] = t * 0.2;
        }
        matlab_mat *X = make_vec(x), *Y = make_vec(y), *Z = make_vec(z);
        matlab_figure_new();
        matlab_plot3(X, Y, Z);
        const char *t = "plot3: helix"; matlab_title(t, std::strlen(t));
        const char *p = "/tmp/matlab_plot_3d_helix.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        free_vec(X); free_vec(Y); free_vec(Z);
    }

    /* ---- 18. mesh + surf — peaks-like surface side by side. ---- */
    {
        constexpr int M = 30, N = 30;
        std::vector<double> Z(M * N);
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c) {
                double x = -3.0 + 6.0 * c / (N - 1);
                double y = -3.0 + 6.0 * r / (M - 1);
                Z[r * N + c] = 3 * (1 - x) * (1 - x) * std::exp(-x*x - (y+1)*(y+1))
                             - 10 * (x/5 - x*x*x - y*y*y*y*y) * std::exp(-x*x - y*y)
                             - (1.0/3) * std::exp(-(x+1)*(x+1) - y*y);
            }
        auto *MZ = static_cast<matlab_mat *>(std::malloc(sizeof(matlab_mat)));
        MZ->rows = M; MZ->cols = N;
        MZ->data = static_cast<double *>(std::malloc(M*N*sizeof(double)));
        std::memcpy(MZ->data, Z.data(), M*N*sizeof(double));

        matlab_figure_new();
        matlab_subplot(1, 2, 1);
        matlab_mesh1(MZ);
        const char *t1 = "mesh(peaks)"; matlab_title(t1, std::strlen(t1));

        matlab_subplot(1, 2, 2);
        const char *cm = "viridis"; matlab_colormap(cm, std::strlen(cm));
        matlab_surf1(MZ);
        const char *t2 = "surf(peaks)"; matlab_title(t2, std::strlen(t2));

        const char *p = "/tmp/matlab_plot_3d_surface.png";
        int rc = matlab_savefig(p, std::strlen(p));
        std::printf("%-30s rc=%d size=%ld\n", p, rc, file_size(p));
        if (rc != 0) rc_total = 1;
        std::free(MZ->data); std::free(MZ);
    }

    matlab_plot_buffer png = matlab_render_png();
    std::printf("in-memory png      bytes=%zu\n", png.size);
    if (png.size == 0) rc_total = 1;
    matlab_plot_buffer_free(png);

    matlab_close_all();
    return rc_total;
}
