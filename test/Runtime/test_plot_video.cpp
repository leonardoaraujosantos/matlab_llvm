// test/Runtime/test_plot_video.cpp — direct C-ABI tests for getframe and
// VideoWriter (docs/plotting.md §4 Tier A/B). No JIT involved; exercises the
// same runtime entry points matlabc lowers to.
//
// Frame capture (getframe / render_raw) is always tested. Real video encoding
// is only asserted when this build links libav (MATLAB_LLVM_WITH_PLOT_FFMPEG);
// otherwise the test asserts the writer reports the disabled state cleanly.

#include "matlab_plot.h"
#include "runtime_internal.h"

#include <cmath>
#include <cstdint>
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

long file_size(const char *p) {
    std::FILE *f = std::fopen(p, "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fclose(f);
    return n;
}

/* Build a simple plot on the current figure for frame k. */
void draw_frame(int k) {
    std::vector<double> sx, sy;
    std::vector<double> X, Y;
    for (int i = 0; i <= 60; ++i) {
        double x = i * 0.1;
        X.push_back(x);
        Y.push_back(std::sin(x + k * 0.3));
    }
    matlab_mat x = make_vec(sx, X), y = make_vec(sy, Y);
    matlab_plot2(&x, &y);
}

}  // namespace

/* ---- Test 1: getframe captures the current figure. ---- */
static int test_getframe() {
    matlab_close_all();
    matlab_figure_new();
    draw_frame(0);
    matlab_frame *f = matlab_getframe();
    CHECK(f != nullptr);
    /* A second capture is independent and also valid. */
    draw_frame(1);
    matlab_frame *f2 = matlab_getframe();
    CHECK(f2 != nullptr);
    CHECK(f2 != f);
    matlab_close_all();   // frees the per-thread frame registry
    return 0;
}

/* ---- Test 2: VideoWriter rejects null / unopened misuse cleanly. ---- */
static int test_videowriter_guards() {
    matlab_close_all();
    CHECK(matlab_videowriter_new(nullptr, 0) == nullptr);

    const char *path = "/tmp/mlv_guard.mp4";
    matlab_videowriter *v = matlab_videowriter_new(path, (int64_t)std::strlen(path));
    CHECK(v != nullptr);
    /* writeVideo before open() must fail, not crash. */
    matlab_figure_new();
    draw_frame(0);
    matlab_frame *frame = matlab_getframe();
    CHECK(frame != nullptr);
    CHECK(matlab_videowriter_write(v, frame) != 0);   // not opened yet
    matlab_videowriter_close(v);
    matlab_close_all();
    return 0;
}

/* ---- Test 3: full lifecycle. ---- */
static int test_videowriter_lifecycle() {
    matlab_close_all();
    const char *path = "/tmp/mlv_lifecycle.mp4";
    std::remove(path);

    matlab_videowriter *v =
        matlab_videowriter_new_profile(path, (int64_t)std::strlen(path),
                                       "MPEG-4", 6);
    CHECK(v != nullptr);
    matlab_videowriter_set_framerate(v, 24.0);
    matlab_videowriter_set_quality(v, 90.0);
    CHECK(matlab_videowriter_open(v) == 0);

    matlab_figure_new();
    int write_rc = 0;
    for (int k = 0; k < 8; ++k) {
        draw_frame(k);
        matlab_frame *frame = matlab_getframe();
        CHECK(frame != nullptr);
        write_rc |= matlab_videowriter_write(v, frame);
    }
    int close_rc = matlab_videowriter_close(v);

#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG
    /* With libav linked, every frame encodes and the file is finalised. */
    CHECK(write_rc == 0);
    CHECK(close_rc == 0);
    long sz = file_size(path);
    CHECK(sz > 0);
    /* MP4 starts with an ftyp box: bytes 4..7 == "ftyp". */
    std::FILE *fp = std::fopen(path, "rb");
    CHECK(fp != nullptr);
    unsigned char hdr[12] = {};
    size_t got = std::fread(hdr, 1, sizeof hdr, fp);
    std::fclose(fp);
    CHECK(got == sizeof hdr);
    CHECK(std::memcmp(hdr + 4, "ftyp", 4) == 0);
#else
    /* Without libav, the writer reports the disabled state rather than
     * silently producing a bogus file. */
    CHECK(write_rc != 0);
    CHECK(close_rc != 0);
    (void)file_size;
#endif
    matlab_close_all();
    return 0;
}

int main() {
    if (test_getframe())              return 1;
    if (test_videowriter_guards())    return 1;
    if (test_videowriter_lifecycle()) return 1;
    std::printf("test_plot_video: all passed\n");
    return 0;
}
