/* ============================================================================
 * runtime_images.cpp — Image Processing Toolbox runtime (Tiers 1-2)
 * ----------------------------------------------------------------------------
 * Tier-1: image file I/O (PGM/PPM/BMP) + a synthetic generator, type
 * conversions, image arithmetic, histogram/intensity statistics.
 * Tier-2: spatial filtering (fspecial / imgaussfilt / imboxfilt / medfilt2 /
 * ordfilt2 / stdfilt / rangefilt) and enhancement (histeq / adapthisteq /
 * imsharpen / imhistmatch / imnoise).
 *
 * Representation (documented simplification): images are plain `double`
 * matrices — M×N grayscale (`matlab_mat`) or M×N×3 truecolor
 * (`matlab_mat3`, slice-major), values in [0,255] for uint8-class images
 * or [0,1] for double-class.  This reuses the entire shipped double kernel
 * (conv2 / imfilter / padarray); the uint8 *class* tag is approximated by
 * value range + saturation in the arithmetic ops.  No external dependency.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

/* shipped helpers reused. */
extern "C" matlab_mat *matlab_imfilter(matlab_mat *A, matlab_mat *h);
extern "C" matlab_mat *matlab_conv2(matlab_mat *A, matlab_mat *h);
extern "C" matlab_mat *matlab_rand(double m, double n);
extern "C" matlab_mat *matlab_randn(double m, double n);

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
struct img_string_s { char *data; int64_t len; };

/* ===== file-scope helpers ================================================ */

static std::string img_sstr(const void *s) {
    if (!s) return std::string();
    const img_string_s *p = reinterpret_cast<const img_string_s *>(s);
    if (!p->data || p->len <= 0) return std::string();
    return std::string(p->data, p->data + p->len);
}
static double img_sc(const matlab_mat *m, double dflt) {
    return (m && m->data && m->rows * m->cols > 0) ? m->data[0] : dflt;
}
static double img_clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
/* Does the data look like a uint8-range image (max > 1)? */
static bool img_is_u8range(const double *d, int64_t n) {
    for (int64_t i = 0; i < n; ++i) if (d[i] > 1.0) return true;
    return false;
}
static std::string img_ext_lower(const std::string &path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return std::string();
    std::string e = path.substr(dot + 1);
    for (char &c : e) c = static_cast<char>(tolower(c));
    return e;
}

/* ----- PNM (PGM/PPM) parsing ----------------------------------------------*/
static int pnm_next(FILE *f) {           /* next token int, skipping ws/comments */
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while ((c = fgetc(f)) != '\n' && c != EOF) {} continue; }
        if (c == EOF) return -1;
        if (!isspace(c)) break;
    }
    int v = 0;
    while (c != EOF && !isspace(c)) { v = v * 10 + (c - '0'); c = fgetc(f); }
    return v;
}

/* image arithmetic binop (saturating); file scope — templates
 * can't live inside extern "C". */
template <class F>
static matlab_mat *img_binop(matlab_mat *A, matlab_mat *B, F f) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    bool bscalar = (B && B->rows * B->cols == 1);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) {
        double b = B ? (bscalar ? B->data[0] : B->data[i]) : 0.0;
        R->data[i] = img_clamp(f(A->data[i], b), 0, 255);
    }
    return R;
}

/* ===== Tier-3 geometric helpers (file scope) ============================ */

/* obj accessors for the affine2d / projective2d transform classdefs. */
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" int         matlab_obj_is_known(const void *p);

/* method code from a string arg (0=nearest, 1=bilinear, 2=bicubic). */
static int img_method(const void *s, int dflt) {
    std::string t = img_sstr(s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    if (t.find("nearest") != std::string::npos) return 0;
    if (t.find("bilinear") != std::string::npos || t.find("linear") != std::string::npos) return 1;
    if (t.find("bicubic") != std::string::npos || t.find("cubic") != std::string::npos) return 2;
    return dflt;
}
static double img_cubic(double t) {                 /* cubic-convolution kernel, a=-0.5 */
    double a = -0.5, x = fabs(t);
    if (x <= 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    if (x < 2.0)  return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    return 0.0;
}
/* sample plane d (H×W row-major) at fractional (fy,fx); 0 outside. */
static double img_sample_plane(const double *d, int64_t H, int64_t W,
                              double fy, double fx, int method) {
    if (method == 0) {
        int64_t iy = static_cast<int64_t>(floor(fy + 0.5)), ix = static_cast<int64_t>(floor(fx + 0.5));
        return (iy >= 0 && iy < H && ix >= 0 && ix < W) ? d[iy * W + ix] : 0.0;
    }
    int64_t y0 = static_cast<int64_t>(floor(fy)), x0 = static_cast<int64_t>(floor(fx));
    auto at = [&](int64_t yy, int64_t xx) { return (yy >= 0 && yy < H && xx >= 0 && xx < W) ? d[yy * W + xx] : 0.0; };
    if (method == 2) {
        double sum = 0.0;
        for (int m = -1; m <= 2; ++m) for (int nn = -1; nn <= 2; ++nn)
            sum += at(y0 + m, x0 + nn) * img_cubic(fy - (y0 + m)) * img_cubic(fx - (x0 + nn));
        return sum;
    }
    double dy = fy - y0, dx = fx - x0;
    return at(y0, x0) * (1 - dy) * (1 - dx) + at(y0, x0 + 1) * (1 - dy) * dx +
           at(y0 + 1, x0) * dy * (1 - dx) + at(y0 + 1, x0 + 1) * dy * dx;
}

/* 3×3 inverse (row-major in/out); returns identity on singular. */
static void img_inv3x3(const double *M, double *out) {
    double det = M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]);
    if (fabs(det) < 1e-300) { for (int i = 0; i < 9; ++i) out[i] = (i % 4 == 0) ? 1.0 : 0.0; return; }
    double id = 1.0 / det;
    out[0] =  (M[4]*M[8]-M[5]*M[7]) * id; out[1] = -(M[1]*M[8]-M[2]*M[7]) * id; out[2] =  (M[1]*M[5]-M[2]*M[4]) * id;
    out[3] = -(M[3]*M[8]-M[5]*M[6]) * id; out[4] =  (M[0]*M[8]-M[2]*M[6]) * id; out[5] = -(M[0]*M[5]-M[2]*M[3]) * id;
    out[6] =  (M[3]*M[7]-M[4]*M[6]) * id; out[7] = -(M[0]*M[7]-M[1]*M[6]) * id; out[8] =  (M[0]*M[4]-M[1]*M[3]) * id;
}

/* solve A x = b (A n×n row-major) by Gaussian elimination w/ partial pivot. */
static std::vector<double> img_gauss(std::vector<double> A, std::vector<double> b, int n) {
    for (int col = 0; col < n; ++col) {
        int piv = col; double best = fabs(A[static_cast<size_t>(col * n + col)]);
        for (int r = col + 1; r < n; ++r) { double v = fabs(A[static_cast<size_t>(r * n + col)]); if (v > best) { best = v; piv = r; } }
        if (best < 1e-300) continue;
        if (piv != col) { for (int j = 0; j < n; ++j) std::swap(A[static_cast<size_t>(col*n+j)], A[static_cast<size_t>(piv*n+j)]); std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(piv)]); }
        double d = A[static_cast<size_t>(col*n+col)];
        for (int j = 0; j < n; ++j) A[static_cast<size_t>(col*n+j)] /= d; b[static_cast<size_t>(col)] /= d;
        for (int r = 0; r < n; ++r) { if (r == col) continue; double f = A[static_cast<size_t>(r*n+col)];
            for (int j = 0; j < n; ++j) A[static_cast<size_t>(r*n+j)] -= f * A[static_cast<size_t>(col*n+j)]; b[static_cast<size_t>(r)] -= f * b[static_cast<size_t>(col)]; }
    }
    return b;
}

/* Apply an inverse-coordinate map to produce an OH×OW output (grayscale or
 * per-channel RGB), sampling the input with `method`.  invmap(oy,ox) ->
 * (iy,ix) in input pixel coordinates. */
template <class Map>
static matlab_mat *img_geo_apply(void *A, int64_t OH, int64_t OW, int method, Map invmap) {
    if (mat_is_3d(A)) {
        matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
        int64_t H = m->rows, W = m->cols, ip = H * W, op = OH * OW;
        matlab_mat3 *R = mat3_alloc(OH, OW, m->depth);
        for (int64_t oy = 0; oy < OH; ++oy) for (int64_t ox = 0; ox < OW; ++ox) {
            double iy, ix; invmap(oy, ox, iy, ix);
            for (int64_t c = 0; c < m->depth; ++c)
                R->data[c * op + oy * OW + ox] = img_sample_plane(m->data + c * ip, H, W, iy, ix, method);
        }
        return reinterpret_cast<matlab_mat *>(R);
    }
    matlab_mat *m = reinterpret_cast<matlab_mat *>(A);
    int64_t H = m->rows, W = m->cols;
    matlab_mat *R = mat_alloc(OH, OW);
    for (int64_t oy = 0; oy < OH; ++oy) for (int64_t ox = 0; ox < OW; ++ox) {
        double iy, ix; invmap(oy, ox, iy, ix);
        R->data[oy * OW + ox] = img_sample_plane(m->data, H, W, iy, ix, method);
    }
    return R;
}
static void img_dims(void *A, int64_t &H, int64_t &W) {
    if (mat_is_3d(A)) { matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A); H = m->rows; W = m->cols; }
    else { matlab_mat *m = reinterpret_cast<matlab_mat *>(A); H = m->rows; W = m->cols; }
}

extern "C" {

/* ===== Tier-1 — file I/O ================================================= */

matlab_mat *matlab_image_imread(void *path_s) {
    std::string path = img_sstr(path_s);
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return mat_alloc(0, 0);
    char magic[2] = {0, 0};
    if (fread(magic, 1, 2, f) != 2) { fclose(f); return mat_alloc(0, 0); }
    matlab_mat *out = nullptr;

    if (magic[0] == 'P' && (magic[1] == '5' || magic[1] == '2')) {   /* PGM */
        int w = pnm_next(f), h = pnm_next(f), mx = pnm_next(f);
        if (w <= 0 || h <= 0) { fclose(f); return mat_alloc(0, 0); }
        out = mat_alloc(h, w);
        if (magic[1] == '5') {
            std::vector<unsigned char> buf(static_cast<size_t>(w * h));
            fread(buf.data(), 1, buf.size(), f);
            for (int64_t i = 0; i < static_cast<int64_t>(w) * h; ++i) out->data[i] = buf[static_cast<size_t>(i)];
        } else {
            for (int64_t i = 0; i < static_cast<int64_t>(w) * h; ++i) out->data[i] = pnm_next(f);
        }
        (void)mx;
    } else if (magic[0] == 'P' && (magic[1] == '6' || magic[1] == '3')) { /* PPM RGB */
        int w = pnm_next(f), h = pnm_next(f), mx = pnm_next(f);
        if (w <= 0 || h <= 0) { fclose(f); return mat_alloc(0, 0); }
        matlab_mat3 *rgb = mat3_alloc(h, w, 3);
        int64_t plane = static_cast<int64_t>(w) * h;
        if (magic[1] == '6') {
            std::vector<unsigned char> buf(static_cast<size_t>(w * h * 3));
            fread(buf.data(), 1, buf.size(), f);
            for (int64_t i = 0; i < plane; ++i)
                for (int c = 0; c < 3; ++c)
                    rgb->data[c * plane + i] = buf[static_cast<size_t>(i * 3 + c)];
        } else {
            for (int64_t i = 0; i < plane; ++i)
                for (int c = 0; c < 3; ++c)
                    rgb->data[c * plane + i] = pnm_next(f);
        }
        (void)mx; fclose(f);
        return reinterpret_cast<matlab_mat *>(rgb);
    } else if (magic[0] == 'B' && magic[1] == 'M') {                 /* BMP */
        unsigned char hdr[52];
        if (fread(hdr, 1, 52, f) != 52) { fclose(f); return mat_alloc(0, 0); }
        uint32_t off = hdr[8] | (hdr[9] << 8) | (hdr[10] << 16) | (hdr[11] << 24);
        int32_t w = hdr[16] | (hdr[17] << 8) | (hdr[18] << 16) | (hdr[19] << 24);
        int32_t h = hdr[20] | (hdr[21] << 8) | (hdr[22] << 16) | (hdr[23] << 24);
        uint16_t bpp = hdr[26] | (hdr[27] << 8);
        bool topdown = h < 0; if (topdown) h = -h;
        if (w <= 0 || h <= 0) { fclose(f); return mat_alloc(0, 0); }
        fseek(f, off, SEEK_SET);
        int channels = (bpp == 24) ? 3 : 1;
        int rowbytes = ((w * (bpp / 8) + 3) / 4) * 4;
        std::vector<unsigned char> row(static_cast<size_t>(rowbytes));
        if (channels == 3) {
            matlab_mat3 *rgb = mat3_alloc(h, w, 3);
            int64_t plane = static_cast<int64_t>(w) * h;
            for (int r = 0; r < h; ++r) {
                fread(row.data(), 1, row.size(), f);
                int dr = topdown ? r : (h - 1 - r);
                for (int x = 0; x < w; ++x) {
                    rgb->data[2 * plane + dr * w + x] = row[static_cast<size_t>(x * 3 + 0)]; /* B */
                    rgb->data[1 * plane + dr * w + x] = row[static_cast<size_t>(x * 3 + 1)]; /* G */
                    rgb->data[0 * plane + dr * w + x] = row[static_cast<size_t>(x * 3 + 2)]; /* R */
                }
            }
            fclose(f);
            return reinterpret_cast<matlab_mat *>(rgb);
        }
        out = mat_alloc(h, w);
        for (int r = 0; r < h; ++r) {
            fread(row.data(), 1, row.size(), f);
            int dr = topdown ? r : (h - 1 - r);
            for (int x = 0; x < w; ++x) out->data[dr * w + x] = row[static_cast<size_t>(x)];
        }
    } else {
        fclose(f);
        return mat_alloc(0, 0);   /* PNG/JPEG/TIFF: documented format follow-on */
    }
    fclose(f);
    return out ? out : mat_alloc(0, 0);
}

double matlab_image_imwrite(void *img, void *path_s) {
    std::string path = img_sstr(path_s);
    std::string ext = img_ext_lower(path);
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return 0.0;
    bool rgb = mat_is_3d(img);
    int64_t H, W; const double *d; int64_t plane = 0;
    if (rgb) { matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(img); H = m->rows; W = m->cols; d = m->data; plane = H * W; }
    else     { matlab_mat *m = reinterpret_cast<matlab_mat *>(img);  H = m->rows; W = m->cols; d = m->data; }
    auto px = [&](int64_t i, int c) { return static_cast<unsigned char>(img_clamp(rgb ? d[c * plane + i] : d[i], 0, 255) + 0.5); };

    if (ext == "ppm" || (ext != "pgm" && ext != "bmp" && rgb)) {     /* PPM (P6) */
        fprintf(f, "P6\n%lld %lld\n255\n", static_cast<long long>(W), static_cast<long long>(H));
        for (int64_t i = 0; i < H * W; ++i)
            for (int c = 0; c < 3; ++c) { unsigned char b = px(i, rgb ? c : 0); fwrite(&b, 1, 1, f); }
    } else if (ext == "pgm" || (!rgb && ext != "bmp")) {             /* PGM (P5) */
        fprintf(f, "P5\n%lld %lld\n255\n", static_cast<long long>(W), static_cast<long long>(H));
        for (int64_t i = 0; i < H * W; ++i) { unsigned char b = px(i, 0); fwrite(&b, 1, 1, f); }
    } else {                                                          /* BMP 24-bit */
        int rowbytes = ((W * 3 + 3) / 4) * 4;
        uint32_t dataSize = static_cast<uint32_t>(rowbytes * H);
        uint32_t fileSize = 54 + dataSize;
        unsigned char hdr[54] = {0};
        hdr[0] = 'B'; hdr[1] = 'M';
        hdr[2] = fileSize & 0xff; hdr[3] = (fileSize >> 8) & 0xff; hdr[4] = (fileSize >> 16) & 0xff; hdr[5] = (fileSize >> 24) & 0xff;
        hdr[10] = 54; hdr[14] = 40;
        hdr[18] = W & 0xff; hdr[19] = (W >> 8) & 0xff; hdr[20] = (W >> 16) & 0xff; hdr[21] = (W >> 24) & 0xff;
        hdr[22] = H & 0xff; hdr[23] = (H >> 8) & 0xff; hdr[24] = (H >> 16) & 0xff; hdr[25] = (H >> 24) & 0xff;
        hdr[26] = 1; hdr[28] = 24;
        fwrite(hdr, 1, 54, f);
        std::vector<unsigned char> row(static_cast<size_t>(rowbytes), 0);
        for (int64_t r = H - 1; r >= 0; --r) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t i = r * W + x;
                row[static_cast<size_t>(x * 3 + 0)] = px(i, rgb ? 2 : 0);  /* B */
                row[static_cast<size_t>(x * 3 + 1)] = px(i, rgb ? 1 : 0);  /* G */
                row[static_cast<size_t>(x * 3 + 2)] = px(i, rgb ? 0 : 0);  /* R */
            }
            fwrite(row.data(), 1, row.size(), f);
        }
    }
    fclose(f);
    return 1.0;
}

/* checkerboard(n[, p, q]) — p×q tiles of 2n×2n; values in [0,1] (double). */
matlab_mat *matlab_image_checkerboard(matlab_mat *nm, matlab_mat *pm, matlab_mat *qm) {
    int n = static_cast<int>(img_sc(nm, 10));
    int p = static_cast<int>(img_sc(pm, 4));
    int q = static_cast<int>(img_sc(qm, p));
    if (n < 1) n = 10; if (p < 1) p = 4; if (q < 1) q = p;
    int H = 2 * n * p, W = 2 * n * q;
    matlab_mat *R = mat_alloc(H, W);
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
        int bi = (i / n) & 1, bj = (j / n) & 1;
        R->data[i * W + j] = (bi == bj) ? 0.0 : 1.0;
    }
    return R;
}

/* ===== Tier-1 — type conversions ======================================== */

matlab_mat *matlab_image_im2double(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    if (mat_is_3d(A)) {
        matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
        int64_t n = m->rows * m->cols * m->depth;
        bool u8 = img_is_u8range(m->data, n);
        matlab_mat3 *R = mat3_alloc(m->rows, m->cols, m->depth);
        for (int64_t i = 0; i < n; ++i) R->data[i] = u8 ? m->data[i] / 255.0 : m->data[i];
        return reinterpret_cast<matlab_mat *>(R);
    }
    int64_t n = A->rows * A->cols;
    bool u8 = img_is_u8range(A->data, n);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = u8 ? A->data[i] / 255.0 : A->data[i];
    return R;
}
matlab_mat *matlab_image_im2single(matlab_mat *A) { return matlab_image_im2double(A); }

matlab_mat *matlab_image_im2uint8(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);   /* values in [0,1] -> scale by 255 */
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i)
        R->data[i] = img_clamp(floor((norm ? 255.0 * A->data[i] : A->data[i]) + 0.5), 0, 255);
    return R;
}

/* rgb2gray / im2gray: 0.2989R + 0.5870G + 0.1140B; passthrough if 2-D. */
matlab_mat *matlab_image_rgb2gray(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    if (!mat_is_3d(A)) {
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        memcpy(R->data, A->data, sizeof(double) * static_cast<size_t>(A->rows * A->cols));
        return R;
    }
    matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
    int64_t plane = m->rows * m->cols;
    matlab_mat *R = mat_alloc(m->rows, m->cols);
    for (int64_t i = 0; i < plane; ++i)
        R->data[i] = 0.2989 * m->data[i] + 0.5870 * m->data[plane + i] + 0.1140 * m->data[2 * plane + i];
    return R;
}

matlab_mat *matlab_image_mat2gray(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    double lo = A->data[0], hi = A->data[0];
    for (int64_t i = 1; i < n; ++i) { if (A->data[i] < lo) lo = A->data[i]; if (A->data[i] > hi) hi = A->data[i]; }
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    double rng = hi - lo;
    for (int64_t i = 0; i < n; ++i) R->data[i] = (rng > 0) ? (A->data[i] - lo) / rng : 0.0;
    return R;
}

/* ===== Tier-1 — image arithmetic (saturating to [0,255]) ================ */

matlab_mat *matlab_image_imadd(matlab_mat *A, matlab_mat *B)      { return img_binop(A, B, [](double a, double b){ return a + b; }); }
matlab_mat *matlab_image_imsubtract(matlab_mat *A, matlab_mat *B) { return img_binop(A, B, [](double a, double b){ return a - b; }); }
matlab_mat *matlab_image_immultiply(matlab_mat *A, matlab_mat *B) { return img_binop(A, B, [](double a, double b){ return a * b; }); }
matlab_mat *matlab_image_imdivide(matlab_mat *A, matlab_mat *B)   { return img_binop(A, B, [](double a, double b){ return b != 0 ? a / b : 0.0; }); }
matlab_mat *matlab_image_imabsdiff(matlab_mat *A, matlab_mat *B)  { return img_binop(A, B, [](double a, double b){ return fabs(a - b); }); }
matlab_mat *matlab_image_imcomplement(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    double top = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = top - A->data[i];
    return R;
}
/* imlincomb(k1, A1, k2, A2) = k1*A1 + k2*A2 (saturated). */
matlab_mat *matlab_image_imlincomb(matlab_mat *k1, matlab_mat *A1, matlab_mat *k2, matlab_mat *A2) {
    if (!A1 || !A1->data) return mat_alloc(0, 0);
    int64_t n = A1->rows * A1->cols;
    double a = img_sc(k1, 1.0), b = img_sc(k2, 0.0);
    matlab_mat *R = mat_alloc(A1->rows, A1->cols);
    for (int64_t i = 0; i < n; ++i)
        R->data[i] = img_clamp(a * A1->data[i] + (A2 ? b * A2->data[i] : 0.0), 0, 255);
    return R;
}

/* ===== Tier-1 — histogram + intensity stats ============================= */

matlab_mat *matlab_image_imhist(matlab_mat *A) {
    matlab_mat *R = mat_alloc(256, 1);
    if (!A || !A->data) return R;
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    for (int64_t i = 0; i < n; ++i) {
        double v = norm ? 255.0 * A->data[i] : A->data[i];
        int b = static_cast<int>(img_clamp(floor(v + 0.5), 0, 255));
        R->data[b] += 1.0;
    }
    return R;
}
matlab_mat *matlab_image_stretchlim(matlab_mat *A) {
    matlab_mat *R = mat_alloc(2, 1);
    if (!A || !A->data) { R->data[0] = 0; R->data[1] = 1; return R; }
    int64_t n = A->rows * A->cols;
    std::vector<double> v(A->data, A->data + n);
    std::sort(v.begin(), v.end());
    double scale = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    int64_t lo = static_cast<int64_t>(0.01 * n), hi = static_cast<int64_t>(0.99 * n);
    if (hi >= n) hi = n - 1;
    R->data[0] = v[static_cast<size_t>(lo)] / scale;
    R->data[1] = v[static_cast<size_t>(hi)] / scale;
    return R;
}
/* imadjust(A) auto-stretch; imadjust(A,[lo hi],[bot top][,gamma]). */
static matlab_mat *img_adjust(matlab_mat *A, double lin, double hin, double lout, double hout, double gamma) {
    int64_t n = A->rows * A->cols;
    double scale = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    double span = (hin - lin);
    for (int64_t i = 0; i < n; ++i) {
        double x = A->data[i] / scale;
        double t = (span > 0) ? img_clamp((x - lin) / span, 0, 1) : 0.0;
        double y = pow(t, gamma) * (hout - lout) + lout;
        R->data[i] = y * scale;
    }
    return R;
}
matlab_mat *matlab_image_imadjust1(matlab_mat *A) {
    matlab_mat *sl = matlab_image_stretchlim(A);
    return img_adjust(A, sl->data[0], sl->data[1], 0.0, 1.0, 1.0);
}
matlab_mat *matlab_image_imadjust(matlab_mat *A, matlab_mat *in, matlab_mat *out) {
    double lin = (in && in->rows * in->cols >= 2) ? in->data[0] : 0.0;
    double hin = (in && in->rows * in->cols >= 2) ? in->data[1] : 1.0;
    double lout = (out && out->rows * out->cols >= 2) ? out->data[0] : 0.0;
    double hout = (out && out->rows * out->cols >= 2) ? out->data[1] : 1.0;
    return img_adjust(A, lin, hin, lout, hout, 1.0);
}
matlab_mat *matlab_image_imadjustg(matlab_mat *A, matlab_mat *in, matlab_mat *out, matlab_mat *g) {
    double lin = (in && in->rows * in->cols >= 2) ? in->data[0] : 0.0;
    double hin = (in && in->rows * in->cols >= 2) ? in->data[1] : 1.0;
    double lout = (out && out->rows * out->cols >= 2) ? out->data[0] : 0.0;
    double hout = (out && out->rows * out->cols >= 2) ? out->data[1] : 1.0;
    return img_adjust(A, lin, hin, lout, hout, img_sc(g, 1.0));
}
double matlab_image_mean2(matlab_mat *A) {
    if (!A || !A->data) return 0.0;
    int64_t n = A->rows * A->cols; double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += A->data[i];
    return n ? s / n : 0.0;
}
double matlab_image_std2(matlab_mat *A) {
    if (!A || !A->data) return 0.0;
    int64_t n = A->rows * A->cols; double mu = matlab_image_mean2(A), s = 0.0;
    for (int64_t i = 0; i < n; ++i) { double d = A->data[i] - mu; s += d * d; }
    return n ? sqrt(s / n) : 0.0;
}

/* ===== Tier-2 — fspecial kernels ======================================== */
/* type codes: 1=gaussian 2=average 3=laplacian 4=log 5=sobel 6=prewitt
 * 7=disk 8=motion 9=unsharp. */
matlab_mat *matlab_image_fspecial(void *type_s, matlab_mat *p1m, matlab_mat *p2m) {
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    if (t == "sobel")   { matlab_mat *K = mat_alloc(3, 3); double s[9] = {1,2,1, 0,0,0, -1,-2,-1}; for (int i=0;i<9;++i) K->data[i]=s[i]; return K; }
    if (t == "prewitt") { matlab_mat *K = mat_alloc(3, 3); double s[9] = {1,1,1, 0,0,0, -1,-1,-1}; for (int i=0;i<9;++i) K->data[i]=s[i]; return K; }
    if (t == "laplacian") {
        double a = img_sc(p1m, 0.2);
        matlab_mat *K = mat_alloc(3, 3);
        double c = 4.0 / (a + 1.0), e = a / (a + 1.0), mid = -4.0 / (a + 1.0);
        double s[9] = {e, c/4*0+ (1-a)/(a+1), e,  (1-a)/(a+1), -4.0/(a+1.0), (1-a)/(a+1),  e, (1-a)/(a+1), e};
        (void)c; (void)mid; for (int i=0;i<9;++i) K->data[i]=s[i]; return K;
    }
    if (t == "average") {
        int sz = static_cast<int>(img_sc(p1m, 3));
        matlab_mat *K = mat_alloc(sz, sz);
        for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] = 1.0 / (sz * sz);
        return K;
    }
    if (t == "disk") {
        int r = static_cast<int>(img_sc(p1m, 5));
        int sz = 2 * r + 1; matlab_mat *K = mat_alloc(sz, sz); double sum = 0.0;
        for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
            double dy = i - r, dx = j - r; double w = (dx*dx + dy*dy <= static_cast<double>(r) * r) ? 1.0 : 0.0;
            K->data[i*sz+j] = w; sum += w;
        }
        if (sum > 0) for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] /= sum;
        return K;
    }
    if (t == "log") {
        int sz = static_cast<int>(img_sc(p1m, 5)); double sg = img_sc(p2m, 0.5);
        int r = sz / 2; matlab_mat *K = mat_alloc(sz, sz); double s2 = sg*sg, sum = 0.0;
        for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
            double x = j - r, y = i - r, h = (x*x + y*y - 2*s2) / (s2*s2) * exp(-(x*x+y*y)/(2*s2));
            K->data[i*sz+j] = h; sum += h;
        }
        double mean = sum / (sz*sz); for (int64_t i=0;i<static_cast<int64_t>(sz) * sz;++i) K->data[i] -= mean;
        return K;
    }
    if (t == "motion") {
        int len = static_cast<int>(img_sc(p1m, 9));
        matlab_mat *K = mat_alloc(1, len);
        for (int j = 0; j < len; ++j) K->data[j] = 1.0 / len;
        return K;
    }
    /* gaussian (default) */
    int sz = static_cast<int>(img_sc(p1m, 3)); double sg = img_sc(p2m, 0.5);
    int r = sz / 2; matlab_mat *K = mat_alloc(sz, sz); double sum = 0.0;
    for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
        double x = j - r, y = i - r, h = exp(-(x*x + y*y) / (2*sg*sg));
        K->data[i*sz+j] = h; sum += h;
    }
    if (sum > 0) for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] /= sum;
    return K;
}

/* imgaussfilt(A, sigma): separable Gaussian via fspecial+imfilter. */
matlab_mat *matlab_image_imgaussfilt(matlab_mat *A, matlab_mat *sigm) {
    double sg = img_sc(sigm, 0.5);
    int r = static_cast<int>(ceil(2.0 * sg)); if (r < 1) r = 1;
    int sz = 2 * r + 1;
    matlab_mat *K = mat_alloc(sz, sz); double sum = 0.0;
    for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
        double x = j - r, y = i - r, h = exp(-(x*x + y*y) / (2*sg*sg));
        K->data[i*sz+j] = h; sum += h;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] /= sum;
    return matlab_imfilter(A, K);
}
matlab_mat *matlab_image_imboxfilt(matlab_mat *A, matlab_mat *szm) {
    int sz = static_cast<int>(img_sc(szm, 3));
    matlab_mat *K = mat_alloc(sz, sz);
    for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] = 1.0 / (sz*sz);
    return matlab_imfilter(A, K);
}

/* ----- rank/order filters ------------------------------------------------ */
static matlab_mat *img_rankfilt(matlab_mat *A, int mh, int mw, int order /* 1-based; -1 = mean of window stat */, int mode) {
    /* mode 0 = order statistic; mode 1 = std; mode 2 = range. */
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    int rr = mh / 2, rc = mw / 2;
    matlab_mat *R = mat_alloc(H, W);
    std::vector<double> win;
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        win.clear();
        for (int di = -rr; di <= rr; ++di) for (int dj = -rc; dj <= rc; ++dj) {
            int64_t y = i + di, x = j + dj;
            double v = (y >= 0 && y < H && x >= 0 && x < W) ? A->data[y * W + x] : 0.0;
            win.push_back(v);
        }
        if (mode == 0) {
            std::sort(win.begin(), win.end());
            int k = order; if (k < 1) k = static_cast<int>(win.size()) / 2 + 1; if (k > static_cast<int>(win.size())) k = static_cast<int>(win.size());
            R->data[i * W + j] = win[static_cast<size_t>(k - 1)];
        } else {
            double mu = 0.0; for (double v : win) mu += v; mu /= win.size();
            if (mode == 1) { double s = 0.0; for (double v : win) s += (v-mu)*(v-mu); R->data[i*W+j] = sqrt(s/win.size()); }
            else { double lo = win[0], hi = win[0]; for (double v : win) { if (v<lo) lo=v; if (v>hi) hi=v; } R->data[i*W+j] = hi - lo; }
        }
    }
    return R;
}
matlab_mat *matlab_image_medfilt2(matlab_mat *A, matlab_mat *szm) {
    int m = 3, n = 3;
    if (szm && szm->rows * szm->cols >= 2) { m = static_cast<int>(szm->data[0]); n = static_cast<int>(szm->data[1]); }
    else if (szm && szm->rows * szm->cols == 1) { m = n = static_cast<int>(szm->data[0]); }
    return img_rankfilt(A, m, n, (m*n)/2 + 1, 0);
}
matlab_mat *matlab_image_ordfilt2(matlab_mat *A, matlab_mat *ordm, matlab_mat *domm) {
    int order = static_cast<int>(img_sc(ordm, 1));
    int sz = 3; if (domm && domm->rows * domm->cols > 1) sz = static_cast<int>(domm->rows);
    return img_rankfilt(A, sz, sz, order, 0);
}
matlab_mat *matlab_image_stdfilt(matlab_mat *A)   { return img_rankfilt(A, 3, 3, 0, 1); }
matlab_mat *matlab_image_rangefilt(matlab_mat *A) { return img_rankfilt(A, 3, 3, 0, 2); }

/* ===== Tier-2 — enhancement ============================================= */

/* histeq(A): global histogram equalisation to [0,255] (uint8-class). */
matlab_mat *matlab_image_histeq(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    double hist[256] = {0};
    for (int64_t i = 0; i < n; ++i) {
        double v = norm ? 255.0 * A->data[i] : A->data[i];
        hist[static_cast<int>(img_clamp(floor(v + 0.5), 0, 255))] += 1.0;
    }
    double cdf[256]; double acc = 0.0;
    for (int k = 0; k < 256; ++k) { acc += hist[k]; cdf[k] = acc; }
    double total = cdf[255], cdfmin = 0.0;
    for (int k = 0; k < 256; ++k) if (cdf[k] > 0) { cdfmin = cdf[k]; break; }
    double map[256];
    for (int k = 0; k < 256; ++k)
        map[k] = (total > cdfmin) ? (cdf[k] - cdfmin) / (total - cdfmin) * 255.0 : k;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) {
        double v = norm ? 255.0 * A->data[i] : A->data[i];
        int b = static_cast<int>(img_clamp(floor(v + 0.5), 0, 255));
        R->data[i] = norm ? map[b] / 255.0 : map[b];
    }
    return R;
}

/* adapthisteq(A): tiled adaptive equalisation (8×8 tiles, clip 0.01),
 * bilinear interpolation between tile mappings (a compact CLAHE). */
matlab_mat *matlab_image_adapthisteq(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    bool norm = !img_is_u8range(A->data, H * W);
    int nt = 8;
    int th = static_cast<int>((H + nt - 1) / nt), tw = static_cast<int>((W + nt - 1) / nt);
    std::vector<std::vector<double>> maps(static_cast<size_t>(nt * nt), std::vector<double>(256, 0.0));
    auto pix = [&](int64_t i, int64_t j) { double v = A->data[i*W+j]; return img_clamp(floor((norm?255.0*v:v)+0.5),0,255); };
    for (int ty = 0; ty < nt; ++ty) for (int tx = 0; tx < nt; ++tx) {
        double hist[256] = {0}; int64_t cnt = 0;
        for (int64_t i = ty*th; i < std::min<int64_t>((ty+1)*th, H); ++i)
            for (int64_t j = tx*tw; j < std::min<int64_t>((tx+1)*tw, W); ++j) { hist[static_cast<int>(pix(i,j))]++; cnt++; }
        double clip = 0.01 * cnt, excess = 0.0;
        for (int k = 0; k < 256; ++k) if (hist[k] > clip) { excess += hist[k] - clip; hist[k] = clip; }
        double add = excess / 256.0; for (int k = 0; k < 256; ++k) hist[k] += add;
        double acc = 0.0; auto &mp = maps[static_cast<size_t>(ty*nt+tx)];
        for (int k = 0; k < 256; ++k) { acc += hist[k]; mp[static_cast<size_t>(k)] = (cnt>0) ? acc/cnt*255.0 : k; }
    }
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        int b = static_cast<int>(pix(i, j));
        double gy = (i + 0.5) / th - 0.5, gx = (j + 0.5) / tw - 0.5;
        int ty0 = static_cast<int>(floor(gy)), tx0 = static_cast<int>(floor(gx));
        double fy = gy - ty0, fx = gx - tx0;
        auto M = [&](int ty, int tx) { ty = std::max(0, std::min(nt-1, ty)); tx = std::max(0, std::min(nt-1, tx)); return maps[static_cast<size_t>(ty*nt+tx)][static_cast<size_t>(b)]; };
        double v = (1-fy)*(1-fx)*M(ty0,tx0) + (1-fy)*fx*M(ty0,tx0+1) + fy*(1-fx)*M(ty0+1,tx0) + fy*fx*M(ty0+1,tx0+1);
        R->data[i*W+j] = norm ? v/255.0 : v;
    }
    return R;
}

/* imsharpen(A): unsharp masking, amount 0.8. */
matlab_mat *matlab_image_imsharpen(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    double sgv = 1.0; matlab_mat sg; sg.data = &sgv; sg.rows = 1; sg.cols = 1;
    matlab_mat *blur = matlab_image_imgaussfilt(A, &sg);
    int64_t n = A->rows * A->cols; double amount = 0.8;
    double top = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = img_clamp(A->data[i] + amount * (A->data[i] - blur->data[i]), 0, top);
    return R;
}

/* imhistmatch(A, ref): match A's histogram to ref's via CDF mapping. */
matlab_mat *matlab_image_imhistmatch(matlab_mat *A, matlab_mat *ref) {
    if (!A || !A->data || !ref || !ref->data) return mat_alloc(0, 0);
    auto cdf256 = [](matlab_mat *X, double *cdf) {
        int64_t n = X->rows * X->cols; bool norm = !img_is_u8range(X->data, n);
        double h[256] = {0};
        for (int64_t i = 0; i < n; ++i) h[static_cast<int>(img_clamp(floor((norm?255*X->data[i]:X->data[i])+0.5),0,255))]++;
        double acc = 0; for (int k = 0; k < 256; ++k) { acc += h[k]; cdf[k] = acc / n; }
    };
    double ca[256], cr[256]; cdf256(A, ca); cdf256(ref, cr);
    int map[256];
    for (int k = 0; k < 256; ++k) { int j = 0; while (j < 255 && cr[j] < ca[k]) j++; map[k] = j; }
    int64_t n = A->rows * A->cols; bool norm = !img_is_u8range(A->data, n);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) {
        int b = static_cast<int>(img_clamp(floor((norm?255*A->data[i]:A->data[i])+0.5),0,255));
        R->data[i] = norm ? map[b] / 255.0 : map[b];
    }
    return R;
}

/* imnoise(A, type, p): gaussian / salt & pepper / speckle. */
matlab_mat *matlab_image_imnoise(matlab_mat *A, void *type_s, matlab_mat *pm) {
    if (!A || !A->data) return mat_alloc(0, 0);
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    double scale = norm ? 1.0 : 255.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    if (t.find("salt") != std::string::npos) {
        double d = img_sc(pm, 0.05);
        matlab_mat *u = matlab_rand(A->rows, A->cols);
        for (int64_t i = 0; i < n; ++i) {
            if (u->data[i] < d / 2) R->data[i] = 0.0;
            else if (u->data[i] > 1 - d / 2) R->data[i] = scale;
            else R->data[i] = A->data[i];
        }
    } else if (t.find("speckle") != std::string::npos) {
        double v = img_sc(pm, 0.04);
        matlab_mat *g = matlab_randn(A->rows, A->cols);
        for (int64_t i = 0; i < n; ++i) R->data[i] = img_clamp(A->data[i] + A->data[i] * sqrt(v) * g->data[i], 0, scale);
    } else {                                  /* gaussian */
        double var = img_sc(pm, 0.01);
        matlab_mat *g = matlab_randn(A->rows, A->cols);
        for (int64_t i = 0; i < n; ++i) R->data[i] = img_clamp(A->data[i] + sqrt(var) * scale * g->data[i], 0, scale);
    }
    return R;
}

/* ===== Tier-3 — geometric transformations =============================== */

/* imresize(A, scale|[rows cols][, method]).  scale scalar = uniform; a
 * 1×2 vector gives the explicit output size.  Default method bicubic. */
matlab_mat *matlab_image_imresize(matlab_mat *A, matlab_mat *scalem, void *method_s) {
    if (!A) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    int64_t OH, OW;
    if (scalem && scalem->rows * scalem->cols >= 2) { OH = static_cast<int64_t>(scalem->data[0]); OW = static_cast<int64_t>(scalem->data[1]); }
    else { double s = img_sc(scalem, 1.0); OH = static_cast<int64_t>(floor(H * s + 0.5)); OW = static_cast<int64_t>(floor(W * s + 0.5)); }
    if (OH < 1) OH = 1; if (OW < 1) OW = 1;
    int method = img_method(method_s, 2);
    double sh = static_cast<double>(OH) / H, sw = static_cast<double>(OW) / W;
    return img_geo_apply(A, OH, OW, method, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        iy = (oy + 0.5) / sh - 0.5; ix = (ox + 0.5) / sw - 0.5;
    });
}
matlab_mat *matlab_image_imresize2(matlab_mat *A, matlab_mat *scalem) { return matlab_image_imresize(A, scalem, nullptr); }

/* imrotate(A, angle[, method][, bbox]).  bbox 'crop' keeps the input size;
 * default 'loose' fits the whole rotated image.  Default method nearest. */
matlab_mat *matlab_image_imrotate(matlab_mat *A, matlab_mat *anglem, void *method_s, void *bbox_s) {
    if (!A) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    double ang = img_sc(anglem, 0.0) * M_PI / 180.0;
    double ca = cos(ang), sa = sin(ang);
    int method = img_method(method_s, 0);
    std::string bb = img_sstr(bbox_s);
    bool crop = (bb.find("crop") != std::string::npos);
    int64_t OH, OW;
    if (crop) { OH = H; OW = W; }
    else {
        double cw = fabs(W * ca) + fabs(H * sa), ch = fabs(W * sa) + fabs(H * ca);
        OW = static_cast<int64_t>(ceil(cw)); OH = static_cast<int64_t>(ceil(ch));
    }
    double cy = (H - 1) / 2.0, cx = (W - 1) / 2.0;
    double ocy = (OH - 1) / 2.0, ocx = (OW - 1) / 2.0;
    return img_geo_apply(A, OH, OW, method, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        double dx = ox - ocx, dy = oy - ocy;
        ix = cx + ( ca * dx + sa * dy);
        iy = cy + (-sa * dx + ca * dy);
    });
}
matlab_mat *matlab_image_imrotate2(matlab_mat *A, matlab_mat *ang) { return matlab_image_imrotate(A, ang, nullptr, nullptr); }
matlab_mat *matlab_image_imrotate3(matlab_mat *A, matlab_mat *ang, void *m) { return matlab_image_imrotate(A, ang, m, nullptr); }

/* imcrop(A, [xmin ymin width height]) — 1-based rect, height+1 × width+1. */
matlab_mat *matlab_image_imcrop(matlab_mat *A, matlab_mat *rect) {
    if (!A || !rect || rect->rows * rect->cols < 4) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    int64_t x0 = static_cast<int64_t>(floor(rect->data[0] + 0.5)) - 1;
    int64_t y0 = static_cast<int64_t>(floor(rect->data[1] + 0.5)) - 1;
    int64_t ow = static_cast<int64_t>(floor(rect->data[2] + 0.5)) + 1;
    int64_t oh = static_cast<int64_t>(floor(rect->data[3] + 0.5)) + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x0 + ow > W) ow = W - x0; if (y0 + oh > H) oh = H - y0;
    if (ow < 1 || oh < 1) return mat_alloc(0, 0);
    return img_geo_apply(A, oh, ow, 0, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        iy = static_cast<double>(y0 + oy); ix = static_cast<double>(x0 + ox);
    });
}

/* imtranslate(A, [tx ty]) — same-size shift, bilinear, zero fill. */
matlab_mat *matlab_image_imtranslate(matlab_mat *A, matlab_mat *t) {
    if (!A) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    double tx = (t && t->rows * t->cols >= 1) ? t->data[0] : 0.0;
    double ty = (t && t->rows * t->cols >= 2) ? t->data[1] : 0.0;
    return img_geo_apply(A, H, W, 1, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        ix = ox - tx; iy = oy - ty;
    });
}

/* imwarp(A, tform) — affine (Kind 1) / projective (Kind 2).  Output is the
 * bounding box of the forward-transformed input; bilinear resampling. */
matlab_mat *matlab_image_imwarp(matlab_mat *A, matlab_obj *tform) {
    if (!A || !tform || !matlab_obj_is_known(tform)) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    matlab_mat *Tm = matlab_obj_get_mat(tform, "T", 1);
    if (!Tm || Tm->rows * Tm->cols < 9) return mat_alloc(0, 0);
    double T[9]; for (int i = 0; i < 9; ++i) T[i] = Tm->data[i];
    int kind = static_cast<int>(matlab_obj_get_f64(tform, "Kind", 4));
    /* forward: [x y 1] * T (row-major).  x=col, y=row. */
    auto fwd = [&](double x, double y, double &xp, double &yp) {
        double u = T[0]*x + T[3]*y + T[6];
        double v = T[1]*x + T[4]*y + T[7];
        double w = T[2]*x + T[5]*y + T[8];
        if (kind == 2 && fabs(w) > 1e-300) { xp = u / w; yp = v / w; } else { xp = u; yp = v; }
    };
    double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
    double cx[4] = {0, static_cast<double>(W - 1), 0, static_cast<double>(W - 1)};
    double cy[4] = {0, 0, static_cast<double>(H - 1), static_cast<double>(H - 1)};
    for (int k = 0; k < 4; ++k) { double xp, yp; fwd(cx[k], cy[k], xp, yp);
        minx = std::min(minx, xp); maxx = std::max(maxx, xp); miny = std::min(miny, yp); maxy = std::max(maxy, yp); }
    int64_t OW = static_cast<int64_t>(floor(maxx - minx + 0.5)) + 1;
    int64_t OH = static_cast<int64_t>(floor(maxy - miny + 0.5)) + 1;
    if (OW < 1) OW = 1; if (OH < 1) OH = 1;
    double Tinv[9]; img_inv3x3(T, Tinv);
    return img_geo_apply(A, OH, OW, 1, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        double wx = ox + minx, wy = oy + miny;
        double u = wx*Tinv[0] + wy*Tinv[3] + Tinv[6];
        double v = wx*Tinv[1] + wy*Tinv[4] + Tinv[7];
        double w = wx*Tinv[2] + wy*Tinv[5] + Tinv[8];
        if (kind == 2 && fabs(w) > 1e-300) { ix = u / w; iy = v / w; } else { ix = u; iy = v; }
    });
}

/* fitgeotform2d(moving, fixed, type) — least-squares affine / similarity;
 * populates an already-allocated affine2d (alloc-then-populate). */
matlab_mat *matlab_image_fitgeo_init(matlab_obj *obj, matlab_mat *moving, matlab_mat *fixed, void *type_s) {
    if (!obj || !moving || !fixed) return mat_alloc(0, 0);
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    int64_t N = moving->rows;
    double T[9] = {1,0,0, 0,1,0, 0,0,1};
    if (t.find("similarity") != std::string::npos || t.find("rigid") != std::string::npos) {
        /* unknowns [a b tx ty]: fx = a*mx - b*my + tx ; fy = b*mx + a*my + ty. */
        std::vector<double> AtA(16, 0.0), Atb(4, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            double mx = moving->data[i*2+0], my = moving->data[i*2+1];
            double fx = fixed->data[i*2+0],  fy = fixed->data[i*2+1];
            double r1[4] = {mx, -my, 1, 0};   /* = fx */
            double r2[4] = {my,  mx, 0, 1};   /* = fy */
            for (int a = 0; a < 4; ++a) { for (int b = 0; b < 4; ++b) AtA[static_cast<size_t>(a*4+b)] += r1[a]*r1[b] + r2[a]*r2[b];
                Atb[static_cast<size_t>(a)] += r1[a]*fx + r2[a]*fy; }
        }
        std::vector<double> p = img_gauss(AtA, Atb, 4);
        T[0] = p[0]; T[1] = p[1]; T[3] = -p[1]; T[4] = p[0]; T[6] = p[2]; T[7] = p[3];
    } else {
        /* affine: [a c e] from (mx,my,1)->fx ; [b d f]->fy. */
        std::vector<double> AtA(9, 0.0), Atfx(3, 0.0), Atfy(3, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            double mx = moving->data[i*2+0], my = moving->data[i*2+1];
            double fx = fixed->data[i*2+0],  fy = fixed->data[i*2+1];
            double r[3] = {mx, my, 1};
            for (int a = 0; a < 3; ++a) { for (int b = 0; b < 3; ++b) AtA[static_cast<size_t>(a*3+b)] += r[a]*r[b];
                Atfx[static_cast<size_t>(a)] += r[a]*fx; Atfy[static_cast<size_t>(a)] += r[a]*fy; }
        }
        std::vector<double> px = img_gauss(AtA, Atfx, 3), py = img_gauss(AtA, Atfy, 3);
        T[0] = px[0]; T[3] = px[1]; T[6] = px[2];     /* a c e */
        T[1] = py[0]; T[4] = py[1]; T[7] = py[2];     /* b d f */
    }
    matlab_mat *Tm = mat_alloc(3, 3);
    for (int i = 0; i < 9; ++i) Tm->data[i] = T[i];
    matlab_obj_set_mat(obj, "T", 1, Tm);
    matlab_obj_set_f64(obj, "Kind", 4, 1.0);
    return mat_alloc(0, 0);
}

/* ----- multi-arity wrappers (pde_table matches one arity per entry) ------ */
matlab_mat *matlab_image_fspecial1(void *t)            { return matlab_image_fspecial(t, nullptr, nullptr); }
matlab_mat *matlab_image_fspecial2(void *t, matlab_mat *p1) { return matlab_image_fspecial(t, p1, nullptr); }
matlab_mat *matlab_image_checkerboard1(matlab_mat *n)  { return matlab_image_checkerboard(n, nullptr, nullptr); }
matlab_mat *matlab_image_checkerboard2(matlab_mat *n, matlab_mat *p) { return matlab_image_checkerboard(n, p, nullptr); }
matlab_mat *matlab_image_imnoise1(matlab_mat *A)       { return matlab_image_imnoise(A, nullptr, nullptr); }
matlab_mat *matlab_image_imnoise2(matlab_mat *A, void *t) { return matlab_image_imnoise(A, t, nullptr); }
matlab_mat *matlab_image_imgaussfilt1(matlab_mat *A)   { double v = 0.5; matlab_mat s; s.data = &v; s.rows = 1; s.cols = 1; return matlab_image_imgaussfilt(A, &s); }
matlab_mat *matlab_image_medfilt2_1(matlab_mat *A)     { return matlab_image_medfilt2(A, nullptr); }
matlab_mat *matlab_image_imboxfilt1(matlab_mat *A)     { return matlab_image_imboxfilt(A, nullptr); }

}  /* extern "C" */
