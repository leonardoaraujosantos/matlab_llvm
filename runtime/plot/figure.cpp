#include "figure.h"

#include <memory>

namespace matlab_plot {

namespace {

/* Per-thread figure registry. MATLAB's gcf is global, but threading the
 * runtime is in scope, so we keep the current-figure slot per-thread. The
 * figures themselves live in a thread-local owning vector. */
struct ThreadFigures {
    std::vector<std::unique_ptr<Figure>> all;
    Figure *current = nullptr;
    int     next_id = 1;
};

thread_local ThreadFigures tls;

}  // namespace

Figure *current_figure() {
    if (!tls.current) figure_new();
    return tls.current;
}

/* New Figure starts as a 1×1 grid containing one empty Axes. */
static std::unique_ptr<Figure> make_default_figure(int id) {
    auto fig = std::make_unique<Figure>();
    fig->id = id;
    fig->axes_list.push_back(std::make_unique<Axes>());
    return fig;
}

Figure *figure_new() {
    auto fig = make_default_figure(tls.next_id++);
    Figure *raw = fig.get();
    tls.all.push_back(std::move(fig));
    tls.current = raw;
    return raw;
}

Figure *figure_by_id(int id) {
    for (auto &f : tls.all) {
        if (f->id == id) {
            tls.current = f.get();
            return tls.current;
        }
    }
    auto fig = make_default_figure(id);
    if (id >= tls.next_id) tls.next_id = id + 1;
    Figure *raw = fig.get();
    tls.all.push_back(std::move(fig));
    tls.current = raw;
    return raw;
}

void close_all() {
    tls.all.clear();
    tls.current = nullptr;
    tls.next_id = 1;
}

Axes &current_axes() {
    return current_figure()->axes();
}

void subplot(int rows, int cols, int idx_1based) {
    if (rows < 1 || cols < 1) return;
    if (idx_1based < 1 || idx_1based > rows * cols) return;
    Figure *f = current_figure();
    if (f->rows != rows || f->cols != cols ||
        static_cast<int>(f->axes_list.size()) != rows * cols) {
        f->rows = rows;
        f->cols = cols;
        f->axes_list.clear();
        for (int i = 0; i < rows * cols; ++i)
            f->axes_list.push_back(std::make_unique<Axes>());
    }
    f->active = idx_1based - 1;
}

}  // namespace matlab_plot
