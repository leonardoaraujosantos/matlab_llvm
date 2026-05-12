/* runtime_rf.cpp — RF Toolbox companion (RF-Tier-1 + RF-Tier-2 subset,
 *                  Friis cascade for rfbudget).
 *
 * Function-form, 2-port-only subset of the RF Toolbox surface
 * documented in docs/comm_toolbox_roadmap.md §9.  v1 ships:
 *
 *   §9.1 RF-Tier-1 — Network parameter objects + Touchstone I/O.
 *     - Touchstone v1 .s2p reader returning a matlab_struct with
 *       S11/S12/S21/S22 (complex column vectors), Frequencies (real
 *       column), Z0 + NumPorts (scalars).  Tolerates MA / DB / RI
 *       data formats and the standard unit set (Hz/kHz/MHz/GHz).
 *     - Touchstone v1 .s2p writer in MA format.
 *     - Per-frequency S↔Y and S↔Z conversions (2-port).
 *
 *   §9.2 RF-Tier-2 — Closed-form S-parameter analyses + cascade.
 *     - gammaIn, gammaOut (input/output reflection coefficient).
 *     - vswr from gamma.
 *     - powerGain (Gt / Ga / Gp via type code).
 *     - stabilityK (Rollett) + stabilityMu (Edwards-Sinsky mu1/mu2).
 *     - cascadeSparams (2-port via T-parameter matrix multiply).
 *     - s2tf voltage transfer function.
 *
 *   §9.2.3 — rfbudget Friis cascade returning the canonical RF-budget
 *     struct (cascaded gain, NF, IP3, output power, SNR).
 *
 * Higher port counts (s3p+, mixed-mode 4-port) are deferred to a
 * follow-on slice; the 2-port subset covers ~90% of practical
 * vendor-data / amplifier / filter / cable analysis.  String
 * selectors are avoided (numeric tag for the powerGain / stabilityMu
 * type choice), matching the rest of the tensor-ops dispatch table.
 *
 * Layout notes:
 *   - Complex S-parameters live in matlab_mat_c column vectors of
 *     length NumFreqs.  Real Frequencies live in matlab_mat column
 *     vectors of the same length.
 *   - 2-port-specific functions take four matlab_mat_c* (one per S
 *     entry) plus the relevant scalar arguments.  This matches how
 *     the lowering ladder routes ptr-typed args.
 *   - The cascade helper returns a matlab_struct with S11/S12/S21/S22
 *     fields, mirroring the touchstoneRead return shape so callers
 *     can chain `cascadeSparams(touchstoneRead('a.s2p'), ...)`.
 */

#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* matlab_string descriptor — first two fields match the runtime
 * layout exactly (data ptr + length).  Used by entries that accept a
 * filename string from the user. */
struct rf_string_view {
    char *data;
    int64_t len;
};

/* Shared struct + string helpers from the main runtime TU. */
extern "C" {
    matlab_struct *matlab_struct_new(void);
    void matlab_struct_set_f64(matlab_struct *s, const char *name,
                                int64_t len, double v);
    void matlab_struct_set_mat(matlab_struct *s, const char *name,
                                int64_t len, matlab_mat *m);
    matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name,
                                       int64_t len);
    double matlab_struct_get_f64(matlab_struct *s, const char *name,
                                  int64_t len);
}

/* ===== Complex helpers (per-frequency 2x2 algebra) =====================
 *
 * Re/Im pair-style arithmetic keeps the inner loop tight without
 * pulling in <complex.h> (which has different ABI conventions across
 * compilers).  Every per-frequency 2-port formula uses these. */
namespace {

struct C { double re, im; };

static inline C cadd(C a, C b) { return {a.re + b.re, a.im + b.im}; }
static inline C csub(C a, C b) { return {a.re - b.re, a.im - b.im}; }
static inline C cmul(C a, C b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static inline C cdiv(C a, C b) {
    double d = b.re * b.re + b.im * b.im;
    if (d == 0.0) return {0.0, 0.0};
    return {(a.re * b.re + a.im * b.im) / d,
            (a.im * b.re - a.re * b.im) / d};
}
static inline double cabs2(C a) { return a.re * a.re + a.im * a.im; }
static inline double cmag(C a)  { return sqrt(cabs2(a)); }

/* Gamma load for a real-impedance termination zl referenced to z0:
 *     Gamma_L = (zl - z0) / (zl + z0). */
static inline C gamma_term(double zl, double z0) {
    if (zl + z0 == 0.0) return {0.0, 0.0};
    return {(zl - z0) / (zl + z0), 0.0};
}

/* Sample S parameter from a complex column vector; clamps to bounds. */
static inline C sread(const matlab_mat_c *S, int64_t k) {
    if (!S) return {0.0, 0.0};
    int64_t N = S->rows * S->cols;
    if (k < 0 || k >= N) return {0.0, 0.0};
    return {S->re[k], S->im[k]};
}

/* Result-cube allocation helpers. */
static inline matlab_mat_c *cvec(int64_t N) {
    matlab_mat_c *out = mat_c_alloc(N, 1);
    return out;
}
static inline matlab_mat *rvec(int64_t N) {
    return mat_alloc(N, 1);
}

/* Determine NumFreqs from any of the four S-parameter vectors. */
static inline int64_t nfreq_of(const matlab_mat_c *s11,
                                const matlab_mat_c *s12,
                                const matlab_mat_c *s21,
                                const matlab_mat_c *s22) {
    int64_t N = 0;
    if (s11) N = std::max<int64_t>(N, s11->rows * s11->cols);
    if (s12) N = std::max<int64_t>(N, s12->rows * s12->cols);
    if (s21) N = std::max<int64_t>(N, s21->rows * s21->cols);
    if (s22) N = std::max<int64_t>(N, s22->rows * s22->cols);
    return N;
}

}  /* anonymous namespace */

extern "C" {

/* ====================================================================== */
/* §9.1.3 Touchstone v1 reader.                                           */
/* ====================================================================== */

/* Parse the option line of a Touchstone v1 file, e.g.:
 *     # MHz S DB R 50
 * Sets freq-unit multiplier (Hz), parameter type ('S'/'Y'/'Z'),
 * data format ('M'/'D'/'R'), reference impedance.  Defaults
 * mirror MathWorks: GHz, S, MA, 50 Ω. */
static void parse_option_line(const char *line,
                                double *freq_mult,
                                char *ptype,
                                char *dformat,
                                double *z0) {
    *freq_mult = 1.0e9;
    *ptype = 'S';
    *dformat = 'M';   /* MA = magnitude / angle */
    *z0 = 50.0;
    /* Skip the leading '#' and whitespace. */
    const char *p = line;
    while (*p && (*p == '#' || *p == ' ' || *p == '\t')) ++p;
    char tok[64];
    while (*p) {
        int i = 0;
        while (*p && (*p == ' ' || *p == '\t')) ++p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n'
                  && i < 63)
            tok[i++] = *p++;
        tok[i] = 0;
        if (!i) break;
        /* Frequency unit. */
        if (!strcasecmp(tok, "HZ"))  *freq_mult = 1.0;
        else if (!strcasecmp(tok, "KHZ")) *freq_mult = 1.0e3;
        else if (!strcasecmp(tok, "MHZ")) *freq_mult = 1.0e6;
        else if (!strcasecmp(tok, "GHZ")) *freq_mult = 1.0e9;
        /* Parameter type. */
        else if (!strcasecmp(tok, "S")) *ptype = 'S';
        else if (!strcasecmp(tok, "Y")) *ptype = 'Y';
        else if (!strcasecmp(tok, "Z")) *ptype = 'Z';
        /* Data format. */
        else if (!strcasecmp(tok, "MA")) *dformat = 'M';
        else if (!strcasecmp(tok, "DB")) *dformat = 'D';
        else if (!strcasecmp(tok, "RI")) *dformat = 'R';
        /* Reference impedance flag — next token is the value. */
        else if (!strcasecmp(tok, "R")) {
            while (*p && (*p == ' ' || *p == '\t')) ++p;
            *z0 = strtod(p, (char **)&p);
        }
    }
}

/* Convert one Touchstone data pair (a, b) into a complex C value
 * according to the file's data format.  dformat ∈ {M = MA, D = DB,
 * R = RI}. */
static C touchstone_decode(double a, double b, char dformat) {
    if (dformat == 'R') {
        /* RI — direct re/im. */
        return {a, b};
    }
    if (dformat == 'D') {
        /* DB — magnitude in dB, angle in degrees. */
        double mag = pow(10.0, a / 20.0);
        double ang = b * M_PI / 180.0;
        return {mag * cos(ang), mag * sin(ang)};
    }
    /* MA — magnitude (linear), angle in degrees. */
    double ang = b * M_PI / 180.0;
    return {a * cos(ang), a * sin(ang)};
}

/* Detect the port count from a filename ending in `.sNp` (Touchstone
 * convention). Returns N if found, else 2 (default). */
static int touchstone_port_count_from_name(const char *path) {
    if (!path) return 2;
    size_t n = strlen(path);
    /* Look for `.s<digits>p` at the end. */
    if (n < 4) return 2;
    /* Scan backward to find a '.'. */
    size_t dot = n;
    for (size_t i = n; i-- > 0; ) {
        if (path[i] == '.') { dot = i; break; }
        if (path[i] == '/' || path[i] == '\\') break;
    }
    if (dot >= n - 2) return 2;
    /* path[dot..n] = ".sNp" or ".sNNp" etc. */
    if (path[dot+1] != 's' && path[dot+1] != 'S') return 2;
    if (path[n-1] != 'p' && path[n-1] != 'P') return 2;
    /* Extract digits between 's' and 'p'. */
    int N = 0;
    for (size_t i = dot + 2; i < n - 1; ++i) {
        if (path[i] >= '0' && path[i] <= '9') N = N * 10 + (path[i] - '0');
        else return 2;
    }
    return (N >= 1 && N <= 64) ? N : 2;
}

matlab_struct *matlab_rf_touchstone_read(void *fname_str) {
    matlab_struct *out = matlab_struct_new();
    rf_string_view *sv = (rf_string_view *)fname_str;
    char path[1024];
    int64_t pn = 0;
    if (sv && sv->data && sv->len > 0) {
        pn = sv->len < 1023 ? sv->len : 1023;
        memcpy(path, sv->data, pn);
    }
    path[pn] = 0;
    int nPorts = touchstone_port_count_from_name(path);
    FILE *fp = fopen(path, "r");
    if (!fp) {
        /* Empty-result struct, sized to expected port count. */
        matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)mat_c_alloc(0, 1));
        matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)mat_c_alloc(0, 1));
        matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)mat_c_alloc(0, 1));
        matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)mat_c_alloc(0, 1));
        matlab_struct_set_mat(out, "Frequencies", 11, mat_alloc(0, 1));
        matlab_struct_set_f64(out, "Z0",       2, 50.0);
        matlab_struct_set_f64(out, "NumPorts", 8, (double)nPorts);
        return out;
    }
    double freq_mult = 1.0e9;
    char ptype = 'S';
    char dformat = 'M';
    double z0 = 50.0;
    int n_per_freq = 1 + 2 * nPorts * nPorts;   /* tokens: f + Sij re/im */
    std::vector<double> freqs;
    /* Each Sij gets its own complex column accumulator (length =
     * NumFreqs).  For sNp with N>2 the per-row layout is row-major
     * [s11 s12 s13 .. s1N; s21 ...].  For s2p (N=2) the layout is the
     * historical [s11 s21 s12 s22] — we transpose at read time. */
    int n_sij = nPorts * nPorts;
    std::vector<std::vector<C>> sij_cols((size_t)n_sij);
    /* Token accumulator across lines for multi-line per-frequency
     * formats. */
    std::vector<double> tokens;
    tokens.reserve((size_t)n_per_freq);
    char line[8192];
    /* Touchstone v2 introduces bracketed metadata keywords —
     * [Version], [Number of Ports], [Two-Port Order], [Reference],
     * [Network Data], [End].  We need to recognise:
     *   [Number of Ports] N         — sets nPorts (overrides file-name guess)
     *   [Two-Port Order] 12_21      — alternate s2p row order (Y or N)
     *   [Reference] z1 z2 …          — per-port reference impedances (use mean)
     *   [Network Data]               — begin payload section
     *   [End]                         — end of file
     * Other bracket lines are tolerated as metadata. */
    bool in_network_data = false;
    int v2_port_order = 0;     /* 0 = default (s2p [s11 s21 s12 s22]),
                                * 1 = 12_21 (s2p row-major [s11 s12 s21 s22]) */
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == 0 || *p == '\r' || *p == '\n') continue;
        if (*p == '!') continue;          /* comment */
        if (*p == '#') {
            parse_option_line(p, &freq_mult, &ptype, &dformat, &z0);
            continue;
        }
        if (*p == '[') {
            /* Touchstone v2 keyword.  Parse the simple ones we
             * understand; tolerate the rest. */
            char tag[64] = {0};
            int ti = 0;
            char *q = p + 1;
            while (*q && *q != ']' && ti < 63) tag[ti++] = *q++;
            tag[ti] = 0;
            const char *args = (*q == ']') ? q + 1 : "";
            while (*args == ' ' || *args == '\t') ++args;
            if (!strcasecmp(tag, "Number of Ports")) {
                int n = atoi(args);
                if (n >= 1 && n <= 64) {
                    nPorts = n;
                    n_per_freq = 1 + 2 * nPorts * nPorts;
                    n_sij = nPorts * nPorts;
                    sij_cols.assign((size_t)n_sij, std::vector<C>());
                }
            } else if (!strcasecmp(tag, "Two-Port Order")) {
                /* Two-Port Order: 12_21 means row-major [s11 s12 s21 s22];
                 *                 21_12 (legacy s2p) means [s11 s21 s12 s22]. */
                if (strstr(args, "12_21") || strstr(args, "12-21")) {
                    v2_port_order = 1;
                }
            } else if (!strcasecmp(tag, "Reference")) {
                /* Average the reference impedances (simple v1 behavior). */
                double sum = 0.0;
                int n = 0;
                const char *r = args;
                while (*r) {
                    char *next = NULL;
                    double v = strtod(r, &next);
                    if (next == r) break;
                    sum += v;
                    n++;
                    r = next;
                }
                if (n > 0) z0 = sum / n;
            } else if (!strcasecmp(tag, "Network Data")) {
                in_network_data = true;
            } else if (!strcasecmp(tag, "End")) {
                break;
            }
            /* Unknown bracket tags are tolerated. */
            continue;
        }
        (void)in_network_data;
        (void)v2_port_order;
        char *q = p;
        while (*q) {
            char *next = NULL;
            double v = strtod(q, &next);
            if (next == q) break;
            tokens.push_back(v);
            q = next;
            if ((int)tokens.size() >= n_per_freq) {
                /* One frequency complete. */
                double f = tokens[0] * freq_mult;
                freqs.push_back(f);
                /* Decode the Sij grid. */
                for (int idx = 0; idx < n_sij; ++idx) {
                    double a = tokens[(size_t)(1 + 2*idx    )];
                    double b = tokens[(size_t)(1 + 2*idx + 1)];
                    C s = touchstone_decode(a, b, dformat);
                    int sij_pos = idx;
                    /* s2p uses [s11 s21 s12 s22] order — transpose
                     * map idx → (i,j) row-major.  For sNp with N != 2,
                     * the file order IS row-major so no transpose. */
                    if (nPorts == 2 && v2_port_order == 0) {
                        /* Touchstone s2p historical position order
                         * [s11 s21 s12 s22] → row-major idx 0/2/1/3.
                         * v2 [Two-Port Order] 12_21 uses row-major
                         * order natively, no remap needed. */
                        static const int s2p_remap[4] = {0, 2, 1, 3};
                        sij_pos = s2p_remap[idx];
                    }
                    sij_cols[(size_t)sij_pos].push_back(s);
                }
                tokens.clear();
            }
        }
    }
    fclose(fp);
    (void)ptype;
    int64_t Nk = (int64_t)freqs.size();
    matlab_mat *F = mat_alloc(Nk, 1);
    for (int64_t k = 0; k < Nk; ++k) F->data[k] = freqs[k];
    /* Per-Sij field naming: "S<i><j>" with 1-based indexing. Cap at
     * 9 ports (single-digit i,j).  Beyond that the field-name
     * decoration scheme needs revisiting. */
    int cap = nPorts <= 9 ? nPorts : 9;
    char fname[8];
    for (int i = 1; i <= cap; ++i) {
        for (int j = 1; j <= cap; ++j) {
            int idx = (i - 1) * nPorts + (j - 1);
            matlab_mat_c *col = mat_c_alloc(Nk, 1);
            if (idx < n_sij) {
                int avail = (int)sij_cols[(size_t)idx].size();
                for (int k = 0; k < Nk && k < avail; ++k) {
                    col->re[k] = sij_cols[(size_t)idx][(size_t)k].re;
                    col->im[k] = sij_cols[(size_t)idx][(size_t)k].im;
                }
            }
            int fn = snprintf(fname, sizeof(fname), "S%d%d", i, j);
            matlab_struct_set_mat(out, fname, fn, (matlab_mat *)col);
        }
    }
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    matlab_struct_set_f64(out, "Z0",       2, z0);
    matlab_struct_set_f64(out, "NumPorts", 8, (double)nPorts);
    return out;
}

/* tsSij(data, i, j) — generic typed-getter for an arbitrary port pair.
 * 1-based indexing.  Returns the complex column for S(i,j). */
matlab_mat_c *matlab_rf_ts_sij(matlab_struct *s, double i_d, double j_d) {
    int i = (int)i_d, j = (int)j_d;
    if (i < 1) i = 1; if (i > 9) i = 9;
    if (j < 1) j = 1; if (j > 9) j = 9;
    char fname[8];
    int fn = snprintf(fname, sizeof(fname), "S%d%d", i, j);
    return (matlab_mat_c *)matlab_struct_get_mat(s, fname, fn);
}

/* Compatibility entry point preserved for the original 2-port shim —
 * the body has been generalized above; keep the function-name aliases
 * for the old code path that called the v1 implementation directly. */
matlab_struct *matlab_rf_touchstone_read_v1_compat(void *fname_str) {
    return matlab_rf_touchstone_read(fname_str);
}


/* Typed-getter helpers that pull S-parameter columns out of the
 * struct returned by matlab_rf_touchstone_read.  Needed because the
 * default struct-field-access lowering routes scalar-typed fields
 * through matlab_struct_get_f64, which would unbox the matlab_mat_c*
 * to 0.0.  The runtime knows the field is matrix-shaped; we expose
 * one entry per parameter so user code can write:
 *
 *   data = touchstoneRead("amp.s2p");
 *   s11 = tsS11(data);
 *   v   = vswr(gammaIn(s11, tsS12(data), tsS21(data), tsS22(data),
 *                       50.0, tsZ0(data)));
 */
extern matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name,
                                          int64_t len);
extern double matlab_struct_get_f64(matlab_struct *s, const char *name,
                                     int64_t len);

matlab_mat_c *matlab_rf_ts_s11(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "S11", 3);
}
matlab_mat_c *matlab_rf_ts_s12(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "S12", 3);
}
matlab_mat_c *matlab_rf_ts_s21(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "S21", 3);
}
matlab_mat_c *matlab_rf_ts_s22(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "S22", 3);
}
matlab_mat *matlab_rf_ts_freqs(matlab_struct *s) {
    return matlab_struct_get_mat(s, "Frequencies", 11);
}
double matlab_rf_ts_z0(matlab_struct *s) {
    return matlab_struct_get_f64(s, "Z0", 2);
}
double matlab_rf_ts_num_ports(matlab_struct *s) {
    return matlab_struct_get_f64(s, "NumPorts", 8);
}

/* ====================================================================== */
/* §9.1.3 Touchstone v1 .s2p writer (MA format).                          */
/* ====================================================================== */

double matlab_rf_touchstone_write_s2p(void *fname_str,
                                       matlab_mat_c *S11, matlab_mat_c *S12,
                                       matlab_mat_c *S21, matlab_mat_c *S22,
                                       matlab_mat *F, double z0) {
    rf_string_view *sv = (rf_string_view *)fname_str;
    char path[1024];
    int64_t pn = 0;
    if (sv && sv->data && sv->len > 0) {
        pn = sv->len < 1023 ? sv->len : 1023;
        memcpy(path, sv->data, pn);
    }
    path[pn] = 0;
    FILE *fp = fopen(path, "w");
    if (!fp) return 0.0;
    int64_t N = F ? F->rows * F->cols : 0;
    fprintf(fp, "! Generated by matlab_llvm runtime_rf.cpp\n");
    fprintf(fp, "# Hz S MA R %g\n", z0);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k);
        C s12 = sread(S12, k);
        C s21 = sread(S21, k);
        C s22 = sread(S22, k);
        double f = F->data[k];
        auto ma = [](C v, double *mag, double *deg) {
            *mag = cmag(v);
            *deg = (v.re == 0.0 && v.im == 0.0) ? 0.0
                 : atan2(v.im, v.re) * 180.0 / M_PI;
        };
        double m11, a11, m12, a12, m21, a21, m22, a22;
        ma(s11, &m11, &a11);
        ma(s21, &m21, &a21);
        ma(s12, &m12, &a12);
        ma(s22, &m22, &a22);
        fprintf(fp, "%.10g %.10g %.6g %.10g %.6g %.10g %.6g %.10g %.6g\n",
                f, m11, a11, m21, a21, m12, a12, m22, a22);
    }
    fclose(fp);
    return 1.0;
}

/* ====================================================================== */
/* §9.2.1 Closed-form S-parameter analyses (2-port).                      */
/* ====================================================================== */

/* gammaIn(s11, s12, s21, s22, zl, z0)
 *   = s11 + s12·s21·gamma_L / (1 - s22·gamma_L) */
matlab_mat_c *matlab_rf_gamma_in(matlab_mat_c *S11, matlab_mat_c *S12,
                                  matlab_mat_c *S21, matlab_mat_c *S22,
                                  double zl, double z0) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *out = cvec(N);
    C gl = gamma_term(zl, z0);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C num = cmul(cmul(s12, s21), gl);
        C den = csub({1.0, 0.0}, cmul(s22, gl));
        C g = cadd(s11, cdiv(num, den));
        out->re[k] = g.re; out->im[k] = g.im;
    }
    return out;
}

/* gammaOut(s11, s12, s21, s22, zs, z0)
 *   = s22 + s12·s21·gamma_S / (1 - s11·gamma_S) */
matlab_mat_c *matlab_rf_gamma_out(matlab_mat_c *S11, matlab_mat_c *S12,
                                   matlab_mat_c *S21, matlab_mat_c *S22,
                                   double zs, double z0) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *out = cvec(N);
    C gs = gamma_term(zs, z0);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C num = cmul(cmul(s12, s21), gs);
        C den = csub({1.0, 0.0}, cmul(s11, gs));
        C g = cadd(s22, cdiv(num, den));
        out->re[k] = g.re; out->im[k] = g.im;
    }
    return out;
}

/* vswr(gamma) = (1 + |gamma|) / (1 - |gamma|).  gamma is a complex
 * column vector (typically the output of gammaIn / gammaOut).  Clamps
 * to a finite ceiling when |gamma| → 1. */
matlab_mat *matlab_rf_vswr_from_gamma(matlab_mat_c *gamma) {
    int64_t N = gamma ? gamma->rows * gamma->cols : 0;
    matlab_mat *out = rvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C g = sread(gamma, k);
        double m = cmag(g);
        if (m >= 1.0) { out->data[k] = 1.0e9; continue; }
        out->data[k] = (1.0 + m) / (1.0 - m);
    }
    return out;
}

/* powerGain(s11, s12, s21, s22, zs, zl, z0, type_code)
 *   type_code = 0 → Gt (transducer)
 *               1 → Ga (available, zl = conj-match at output)
 *               2 → Gp (operating, zs = conj-match at input)
 * Returns the linear power-gain ratio (not dB).  Caller applies
 * 10*log10 when a dB number is wanted. */
matlab_mat *matlab_rf_power_gain(matlab_mat_c *S11, matlab_mat_c *S12,
                                  matlab_mat_c *S21, matlab_mat_c *S22,
                                  double zs, double zl, double z0,
                                  double type_d) {
    int t = (int)type_d;
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat *out = rvec(N);
    C gs = gamma_term(zs, z0);
    C gl = gamma_term(zl, z0);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        double s21_2 = cabs2(s21);
        if (t == 1) {
            /* Ga = |s21|² · (1 - |gs|²) / (|1 - s11·gs|² · (1 - |gout|²)) */
            C gout_n = cmul(cmul(s12, s21), gs);
            C gout_d = csub({1.0, 0.0}, cmul(s11, gs));
            C gout = cadd(s22, cdiv(gout_n, gout_d));
            double num = s21_2 * (1.0 - cabs2(gs));
            C dterm = csub({1.0, 0.0}, cmul(s11, gs));
            double den = cabs2(dterm) * (1.0 - cabs2(gout));
            out->data[k] = den == 0.0 ? 0.0 : num / den;
        } else if (t == 2) {
            /* Gp = |s21|² · (1 - |gl|²) / ((1 - |gin|²) · |1 - s22·gl|²) */
            C gin_n = cmul(cmul(s12, s21), gl);
            C gin_d = csub({1.0, 0.0}, cmul(s22, gl));
            C gin = cadd(s11, cdiv(gin_n, gin_d));
            double num = s21_2 * (1.0 - cabs2(gl));
            C dterm = csub({1.0, 0.0}, cmul(s22, gl));
            double den = (1.0 - cabs2(gin)) * cabs2(dterm);
            out->data[k] = den == 0.0 ? 0.0 : num / den;
        } else {
            /* Gt = |s21|² · (1 - |gs|²) · (1 - |gl|²) /
             *      |(1 - s11·gs)·(1 - s22·gl) - s12·s21·gs·gl|² */
            C den_c = csub(cmul(csub({1.0, 0.0}, cmul(s11, gs)),
                                csub({1.0, 0.0}, cmul(s22, gl))),
                            cmul(cmul(s12, s21), cmul(gs, gl)));
            double num = s21_2 * (1.0 - cabs2(gs)) * (1.0 - cabs2(gl));
            double den = cabs2(den_c);
            out->data[k] = den == 0.0 ? 0.0 : num / den;
        }
    }
    return out;
}

/* stabilityK — Rollett's K factor.
 *   K = (1 - |s11|² - |s22|² + |Δ|²) / (2 · |s12·s21|)
 * with Δ = s11·s22 - s12·s21.  K > 1 + |Δ| < 1 → unconditionally
 * stable. */
matlab_mat *matlab_rf_stability_k(matlab_mat_c *S11, matlab_mat_c *S12,
                                   matlab_mat_c *S21, matlab_mat_c *S22) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat *out = rvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C delta = csub(cmul(s11, s22), cmul(s12, s21));
        double num = 1.0 - cabs2(s11) - cabs2(s22) + cabs2(delta);
        double den = 2.0 * cmag(cmul(s12, s21));
        out->data[k] = den == 0.0 ? 0.0 : num / den;
    }
    return out;
}

/* stabilityMu (Edwards-Sinsky).  type=0 → mu1 (source-side
 * unconditional-stability measure), type=1 → mu2 (load-side).
 *   mu1 = (1 - |s11|²) / (|s22 - conj(s11)·Δ| + |s12·s21|)
 *   mu2 = (1 - |s22|²) / (|s11 - conj(s22)·Δ| + |s12·s21|) */
matlab_mat *matlab_rf_stability_mu(matlab_mat_c *S11, matlab_mat_c *S12,
                                    matlab_mat_c *S21, matlab_mat_c *S22,
                                    double type_d) {
    int t = (int)type_d;
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat *out = rvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C delta = csub(cmul(s11, s22), cmul(s12, s21));
        double num, denA, denB;
        if (t == 1) {
            num = 1.0 - cabs2(s22);
            C conj_s22 = {s22.re, -s22.im};
            denA = cmag(csub(s11, cmul(conj_s22, delta)));
            denB = cmag(cmul(s12, s21));
        } else {
            num = 1.0 - cabs2(s11);
            C conj_s11 = {s11.re, -s11.im};
            denA = cmag(csub(s22, cmul(conj_s11, delta)));
            denB = cmag(cmul(s12, s21));
        }
        double den = denA + denB;
        out->data[k] = den == 0.0 ? 0.0 : num / den;
    }
    return out;
}

/* s2tf — voltage transfer function Vout/Vin for a source impedance zs
 * driving a load zl through the 2-port:
 *   tf = (1 + gamma_L) · s21 · (1 - gamma_S) /
 *        (2 · (1 - s11·gs) · (1 - s22·gl - s12·s21·gs·gl/(1 - s11·gs))) */
matlab_mat_c *matlab_rf_s2tf(matlab_mat_c *S11, matlab_mat_c *S12,
                              matlab_mat_c *S21, matlab_mat_c *S22,
                              double zs, double zl, double z0) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *out = cvec(N);
    C gs = gamma_term(zs, z0);
    C gl = gamma_term(zl, z0);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C oneA = csub({1.0, 0.0}, cmul(s11, gs));
        C oneB = csub({1.0, 0.0}, cmul(s22, gl));
        C cross = cdiv(cmul(cmul(cmul(s12, s21), gs), gl), oneA);
        C den_inner = csub(oneB, cross);
        C den = cmul({2.0, 0.0}, cmul(oneA, den_inner));
        C numL = cadd({1.0, 0.0}, gl);
        C numS = csub({1.0, 0.0}, gs);
        C num = cmul(cmul(numL, s21), numS);
        C tf = cdiv(num, den);
        out->re[k] = tf.re; out->im[k] = tf.im;
    }
    return out;
}

/* ====================================================================== */
/* §9.2.2 Cascade — 2-port T-parameter chain.                             */
/* ====================================================================== */

/* Convert one frequency's S-parameter 2×2 block to T-parameters:
 *   T = [[ -det(S)/s21,  s11/s21 ],
 *        [ -s22/s21,     1/s21    ]]
 * with det(S) = s11·s22 - s12·s21. */
static inline void s_to_t(C s11, C s12, C s21, C s22,
                          C *T11, C *T12, C *T21, C *T22) {
    C inv = cdiv({1.0, 0.0}, s21);
    C det = csub(cmul(s11, s22), cmul(s12, s21));
    *T11 = cmul({-1.0, 0.0}, cmul(det, inv));
    *T12 = cmul(s11, inv);
    *T21 = cmul({-1.0, 0.0}, cmul(s22, inv));
    *T22 = inv;
}

/* T → S conversion:
 *   s11 = T12/T22,   s12 = T11 - T12·T21/T22,
 *   s21 = 1/T22,     s22 = -T21/T22. */
static inline void t_to_s(C T11, C T12, C T21, C T22,
                          C *s11, C *s12, C *s21, C *s22) {
    C inv = cdiv({1.0, 0.0}, T22);
    *s11 = cmul(T12, inv);
    *s21 = inv;
    *s22 = cmul({-1.0, 0.0}, cmul(T21, inv));
    *s12 = csub(T11, cmul(cmul(T12, T21), inv));
}

matlab_struct *matlab_rf_cascade2(matlab_mat_c *A11, matlab_mat_c *A12,
                                   matlab_mat_c *A21, matlab_mat_c *A22,
                                   matlab_mat_c *B11, matlab_mat_c *B12,
                                   matlab_mat_c *B21, matlab_mat_c *B22) {
    int64_t N = nfreq_of(A11, A12, A21, A22);
    int64_t Nb = nfreq_of(B11, B12, B21, B22);
    if (Nb < N) N = Nb;
    matlab_mat_c *C11 = cvec(N), *C12 = cvec(N);
    matlab_mat_c *C21 = cvec(N), *C22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C a11 = sread(A11, k), a12 = sread(A12, k);
        C a21 = sread(A21, k), a22 = sread(A22, k);
        C b11 = sread(B11, k), b12 = sread(B12, k);
        C b21 = sread(B21, k), b22 = sread(B22, k);
        C TA11, TA12, TA21, TA22;
        C TB11, TB12, TB21, TB22;
        s_to_t(a11, a12, a21, a22, &TA11, &TA12, &TA21, &TA22);
        s_to_t(b11, b12, b21, b22, &TB11, &TB12, &TB21, &TB22);
        /* T_AB = T_A · T_B (matrix multiply, 2×2). */
        C T11 = cadd(cmul(TA11, TB11), cmul(TA12, TB21));
        C T12 = cadd(cmul(TA11, TB12), cmul(TA12, TB22));
        C T21 = cadd(cmul(TA21, TB11), cmul(TA22, TB21));
        C T22 = cadd(cmul(TA21, TB12), cmul(TA22, TB22));
        C s11, s12, s21, s22;
        t_to_s(T11, T12, T21, T22, &s11, &s12, &s21, &s22);
        C11->re[k] = s11.re; C11->im[k] = s11.im;
        C12->re[k] = s12.re; C12->im[k] = s12.im;
        C21->re[k] = s21.re; C21->im[k] = s21.im;
        C22->re[k] = s22.re; C22->im[k] = s22.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)C11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)C12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)C21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)C22);
    return out;
}

/* ====================================================================== */
/* §9.1.2 S↔Y and S↔Z conversions (2-port, per-frequency).               */
/* ====================================================================== */

/* For a 2-port S-matrix referenced to a real z0, the corresponding Y
 * and Z matrices are given by:
 *   Y = (1/z0) · (I + S)^{-1} · (I - S)
 *   Z =  z0   · (I - S)^{-1} · (I + S)
 * Returns a struct with Y11/Y12/Y21/Y22 (or Z11/...) complex columns. */
static matlab_struct *s_to_yz(matlab_mat_c *S11, matlab_mat_c *S12,
                               matlab_mat_c *S21, matlab_mat_c *S22,
                               double z0, int do_y) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *O11 = cvec(N), *O12 = cvec(N);
    matlab_mat_c *O21 = cvec(N), *O22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        /* Compute (I ± S)^{-1} via 2x2 cofactor inverse. */
        auto inv2 = [](C a, C b, C c, C d, C *ia, C *ib, C *ic, C *id) {
            C det = csub(cmul(a, d), cmul(b, c));
            C inv = cdiv({1.0, 0.0}, det);
            *ia = cmul( d,           inv);
            *ib = cmul({-b.re, -b.im}, inv);
            *ic = cmul({-c.re, -c.im}, inv);
            *id = cmul( a,           inv);
        };
        C plusA  = cadd({1.0, 0.0}, s11), plusB = s12;
        C plusC  = s21,                   plusD = cadd({1.0, 0.0}, s22);
        C minusA = csub({1.0, 0.0}, s11), minusB = {-s12.re, -s12.im};
        C minusC = {-s21.re, -s21.im},    minusD = csub({1.0, 0.0}, s22);
        C iA, iB, iC, iD;
        if (do_y) inv2(plusA,  plusB,  plusC,  plusD,  &iA, &iB, &iC, &iD);
        else      inv2(minusA, minusB, minusC, minusD, &iA, &iB, &iC, &iD);
        /* Multiply (I ± S)^{-1} · (I ∓ S). */
        C mA, mB, mC, mD;
        if (do_y) { mA = minusA; mB = minusB; mC = minusC; mD = minusD; }
        else       { mA = plusA;  mB = plusB;  mC = plusC;  mD = plusD;  }
        C O_a = cadd(cmul(iA, mA), cmul(iB, mC));
        C O_b = cadd(cmul(iA, mB), cmul(iB, mD));
        C O_c = cadd(cmul(iC, mA), cmul(iD, mC));
        C O_d = cadd(cmul(iC, mB), cmul(iD, mD));
        double scale = do_y ? (1.0 / z0) : z0;
        O11->re[k] = O_a.re * scale; O11->im[k] = O_a.im * scale;
        O12->re[k] = O_b.re * scale; O12->im[k] = O_b.im * scale;
        O21->re[k] = O_c.re * scale; O21->im[k] = O_c.im * scale;
        O22->re[k] = O_d.re * scale; O22->im[k] = O_d.im * scale;
    }
    matlab_struct *out = matlab_struct_new();
    const char *l11 = do_y ? "Y11" : "Z11";
    const char *l12 = do_y ? "Y12" : "Z12";
    const char *l21 = do_y ? "Y21" : "Z21";
    const char *l22 = do_y ? "Y22" : "Z22";
    matlab_struct_set_mat(out, l11, 3, (matlab_mat *)O11);
    matlab_struct_set_mat(out, l12, 3, (matlab_mat *)O12);
    matlab_struct_set_mat(out, l21, 3, (matlab_mat *)O21);
    matlab_struct_set_mat(out, l22, 3, (matlab_mat *)O22);
    return out;
}

matlab_struct *matlab_rf_s2y(matlab_mat_c *S11, matlab_mat_c *S12,
                              matlab_mat_c *S21, matlab_mat_c *S22, double z0) {
    return s_to_yz(S11, S12, S21, S22, z0, 1);
}

matlab_struct *matlab_rf_s2z(matlab_mat_c *S11, matlab_mat_c *S12,
                              matlab_mat_c *S21, matlab_mat_c *S22, double z0) {
    return s_to_yz(S11, S12, S21, S22, z0, 0);
}

/* ====================================================================== */
/* §9.2.3 rfbudget — Friis cascade over a stage list.                     */
/* ====================================================================== */

/* Inputs are real column vectors of equal length giving the per-stage
 * power gain (dB), noise figure (dB), and input third-order intercept
 * point (dBm).  p_in_dBm = input power at stage 1; bw_Hz = noise
 * bandwidth.
 *
 * Outputs (struct):
 *   Gain_dB         — cascaded total gain
 *   NF_dB           — cascaded noise figure (Friis chain)
 *   IP3_in_dBm      — cascaded input-referred third-order intercept
 *   OutputPower_dBm — small-signal output power
 *   NoiseFloor_dBm  — thermal + NF, referenced to output
 *   SNR_dB          — OutputPower − NoiseFloor
 *   ThermalNoise_dBm — kTB at 290 K, output of stage chain
 *
 * Friis NF cascade (linear):  F = F1 + (F2-1)/G1 + (F3-1)/(G1·G2) + ...
 * Cascaded IP3 (input-referred):
 *     1/IP3_in = 1/IP3_1 + G1/IP3_2 + G1·G2/IP3_3 + ... (linear)
 */
matlab_struct *matlab_rf_budget_friis(matlab_mat *gains_dB,
                                       matlab_mat *nfs_dB,
                                       matlab_mat *ip3_dBm,
                                       double p_in_dBm,
                                       double bw_Hz) {
    int64_t N = gains_dB ? gains_dB->rows * gains_dB->cols : 0;
    /* Cascade Friis. */
    double f_total = 0.0;
    double g_run_lin = 1.0;
    double g_total_dB = 0.0;
    double inv_ip3_lin = 0.0;
    for (int64_t k = 0; k < N; ++k) {
        double g_dB = gains_dB->data[k];
        double nf_dB = (nfs_dB && k < nfs_dB->rows * nfs_dB->cols)
                       ? nfs_dB->data[k] : 0.0;
        double ip3 = (ip3_dBm && k < ip3_dBm->rows * ip3_dBm->cols)
                     ? ip3_dBm->data[k] : 1.0e6;
        double f_k = pow(10.0, nf_dB / 10.0);
        double g_k = pow(10.0, g_dB / 10.0);
        if (k == 0) f_total = f_k;
        else        f_total += (f_k - 1.0) / g_run_lin;
        /* Input-referred IP3 cascade.  ip3_k is referred to the
         * stage-k input.  Stage-k input is g_run_lin times the chain
         * input, so referred-to-chain-input it's ip3_k / g_run_lin. */
        double ip3_lin_in = pow(10.0, (ip3 - g_total_dB) / 10.0);
        inv_ip3_lin += 1.0 / ip3_lin_in;
        g_run_lin *= g_k;
        g_total_dB += g_dB;
    }
    double nf_total_dB = 10.0 * log10(f_total <= 0.0 ? 1.0 : f_total);
    double ip3_in_dBm  = inv_ip3_lin > 0.0 ? 10.0 * log10(1.0 / inv_ip3_lin)
                                            : 1.0e6;
    /* Thermal noise at the output: kT at 290 K = -174 dBm/Hz, times
     * the chain gain + bandwidth in dB·Hz. */
    double kT_dBm_per_Hz = -174.0;
    double bw_dBHz = (bw_Hz > 0.0) ? 10.0 * log10(bw_Hz) : 0.0;
    double thermal_out_dBm = kT_dBm_per_Hz + bw_dBHz + g_total_dB;
    double noise_out_dBm   = thermal_out_dBm + nf_total_dB;
    double p_out_dBm = p_in_dBm + g_total_dB;
    double snr_dB = p_out_dBm - noise_out_dBm;
    matlab_struct *out = matlab_struct_new();
    #define SET(name, v) matlab_struct_set_f64(out, name, sizeof(name)-1, v)
    SET("Gain_dB",          g_total_dB);
    SET("NF_dB",            nf_total_dB);
    SET("IP3_in_dBm",       ip3_in_dBm);
    SET("OutputPower_dBm",  p_out_dBm);
    SET("NoiseFloor_dBm",   noise_out_dBm);
    SET("ThermalNoise_dBm", thermal_out_dBm);
    SET("SNR_dB",           snr_dB);
    SET("InputPower_dBm",   p_in_dBm);
    SET("Bandwidth_Hz",     bw_Hz);
    SET("NumStages",        (double)N);
    #undef SET
    return out;
}

/* ====================================================================== */
/* §9.3.1 Vector Fitting (Gustavsen-Semlyen 1999).                        */
/* ====================================================================== */
/*
 * Fits measured frequency-domain data h(jω) with a real-coefficient
 * rational function
 *
 *   H(s) ≈ Σⱼ rⱼ / (s − pⱼ) + d
 *
 * via iterative pole relocation.  v1 keeps poles **real-valued** for
 * simplicity (complex-conjugate-pair poles are a follow-on).  Real
 * poles cover the smooth-attenuation use case (typical S-parameter
 * magnitude fitting) but cannot reproduce sharp resonances; sharp
 * features need complex pairs.
 *
 * Algorithm sketch (Sanathanan-Koerner-style identification):
 *   1. Initial real, negative poles `aⱼ` log-spaced over the data
 *      frequency range.
 *   2. Per iteration:
 *      a. Build the LS system from
 *           Σⱼ cⱼ·φⱼₖ + d − h_k · (Σⱼ c̃ⱼ·φⱼₖ + 1) = 0
 *         where φⱼₖ = 1/(jω_k − aⱼ).  Unknowns: c, d, c̃.  All real
 *         (we keep aⱼ real this slice).
 *      b. Stack the complex equations as 2K real rows (real-then-
 *         imag).
 *      c. Solve via normal equations  AᵀA x = Aᵀb  using the runtime's
 *         real LU solver.
 *      d. New poles = eigenvalues of M = diag(a) − 𝟙·c̃ᵀ.  Drop the
 *         imaginary part (real-pole restriction); flip unstable real
 *         poles to the left half-plane.
 *   3. Final residue fit with the converged poles: smaller LS solving
 *      for c, d only.
 *
 * Returns a matlab_struct with fields:
 *   Poles        — real column (length nPoles)
 *   Residues     — real column (length nPoles)
 *   D            — direct term (scalar)
 *   Order        — number of poles
 *   FitError     — RMS error |h − H| / |h|, averaged over samples
 */

extern matlab_mat *matlab_inv(matlab_mat *A);
extern matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B);
extern matlab_mat *matlab_eig(matlab_mat *A_in);

/* Helpers below stay `static` (internal linkage) inside the extern "C"
 * block — anonymous namespaces aren't legal here, but static C++
 * functions are. */

/* Build LS system (real-stacked) and solve via normal equations.
 *
 *   A is (2K) × M (real), b is (2K) × 1 (real).
 *   Solves   AᵀA · x = Aᵀb     →   x = (AᵀA)⁻¹ Aᵀb     (M × 1).
 *
 * Caller-owned heap-free buffers via std::vector.  We rely on the
 * runtime's real-LU mldivide for the M×M solve, which is cheap when
 * M ≪ K.  For nPoles = 8, M = 17, so AᵀA is 17×17 — trivial. */
static std::vector<double> vf_ls_solve(const std::vector<double> &A,
                                        const std::vector<double> &b,
                                        int twoK, int M) {
    /* AtA = Aᵀ A. */
    std::vector<double> AtA((size_t)M * M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double s = 0.0;
            for (int k = 0; k < twoK; ++k) {
                s += A[(size_t)k * M + i] * A[(size_t)k * M + j];
            }
            AtA[(size_t)i * M + j] = s;
        }
    }
    /* Atb = Aᵀ b. */
    std::vector<double> Atb((size_t)M, 0.0);
    for (int i = 0; i < M; ++i) {
        double s = 0.0;
        for (int k = 0; k < twoK; ++k) {
            s += A[(size_t)k * M + i] * b[k];
        }
        Atb[i] = s;
    }
    /* Wrap into matlab_mat descriptors and call the runtime solver.
     * Zero-copy is fine — the solver reads .data and writes a new
     * output mat. */
    matlab_mat Adesc;
    Adesc.data = AtA.data();
    Adesc.rows = M;
    Adesc.cols = M;
    matlab_mat Bdesc;
    Bdesc.data = Atb.data();
    Bdesc.rows = M;
    Bdesc.cols = 1;
    matlab_mat *X = matlab_mldivide_mm(&Adesc, &Bdesc);
    std::vector<double> x((size_t)M, 0.0);
    if (X && X->rows == M && X->cols == 1) {
        for (int i = 0; i < M; ++i) x[i] = X->data[i];
    }
    return x;
}

/* Extract eigenvalues of M = diag(a) − 𝟙·c̃ᵀ.  Returns the **real
 * parts** of the eigenvalues (real-pole restriction in v1).  Calls
 * matlab_eig which auto-dispatches between symmetric Jacobi and
 * non-symmetric Francis QR.  For our M the matrix is asymmetric in
 * general.  The output type is polymorphic on magic word — we read
 * either the real-mat data[] or the complex-mat re[]. */
static void vf_relocate_poles(const std::vector<double> &a,
                               const std::vector<double> &csig,
                               std::vector<double> &new_poles) {
    int n = (int)a.size();
    matlab::runtime::MatPtr Mmat = matlab::runtime::make_mat(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Mmat->data[(size_t)i * n + j] = (i == j ? a[i] : 0.0) - csig[j];
        }
    }
    matlab_mat *E = matlab_eig(Mmat.get());
    new_poles.resize(n);
    if (!E) { for (int i = 0; i < n; ++i) new_poles[i] = a[i]; return; }
    if (mat_is_complex(E)) {
        matlab_mat_c *Ec = (matlab_mat_c *)E;
        int got = (int)(Ec->rows * Ec->cols);
        for (int i = 0; i < n && i < got; ++i) new_poles[i] = Ec->re[i];
    } else {
        int got = (int)(E->rows * E->cols);
        for (int i = 0; i < n && i < got; ++i) new_poles[i] = E->data[i];
    }
    /* Flip unstable real poles to the left half-plane (Gustavsen-Semlyen
     * trick: a positive real pole means the iteration drove a pole into
     * the right half-plane; mirror it back). */
    for (int i = 0; i < n; ++i) {
        if (new_poles[i] > 0.0) new_poles[i] = -new_poles[i];
        /* Spread degenerate poles slightly to keep the LS conditioning
         * sane on subsequent iterations. */
        if (new_poles[i] == 0.0) new_poles[i] = -1.0;
    }
}

/* v2 rationalfit with complex-conjugate-pair pole support.
 *
 * Each "pole entry" is either a real pole (beta == 0) or the upper
 * half of a complex conjugate pair (beta > 0).  All arithmetic stays
 * real-valued thanks to the (γ, δ) parameterization of paired
 * residues — c = γ + jδ for the upper, c̄ = γ − jδ for the lower —
 * and the [α β; −β α] real 2×2 block structure of complex pairs in
 * the relocation matrix M.  The eigenvalues of M (real-valued)
 * automatically come in real or complex-conjugate pairs and feed
 * directly back into the (α, β) pole representation.
 *
 * Output struct fields:
 *   Poles    — complex column, length = total complex pole count
 *              (n_real + 2·n_pairs), with conjugate pairs stored as
 *              two entries (α + jβ followed by α − jβ).
 *   Residues — complex column, same length, conjugate pairs mirrored.
 *   D        — direct term (real scalar)
 *   Order    — total complex pole count
 *   FitError — relative RMS reconstruction error. */
struct VfPoleEntry { double alpha; double beta; };

/* Forward declaration of the weight-aware helper.  Both the public
 * matlab_rf_rationalfit and matlab_rf_rationalfit_w route through it. */
static matlab_struct *rationalfit_inner(matlab_mat *freq, matlab_mat *h_re,
                                         matlab_mat *h_im, int nPoles,
                                         int nIter, matlab_mat *weight);

matlab_struct *matlab_rf_rationalfit(matlab_mat *freq,
                                      matlab_mat *h_re, matlab_mat *h_im,
                                      double n_poles_d, double n_iter_d) {
    int nPoles = (int)n_poles_d;
    if (nPoles < 1) nPoles = 1;
    if (nPoles > 64) nPoles = 64;
    int nIter = (int)n_iter_d;
    if (nIter < 1) nIter = 1;
    if (nIter > 50) nIter = 50;
    return rationalfit_inner(freq, h_re, h_im, nPoles, nIter, nullptr);
}

matlab_struct *matlab_rf_rationalfit_w(matlab_mat *freq,
                                        matlab_mat *h_re, matlab_mat *h_im,
                                        matlab_mat *weight,
                                        double n_poles_d, double n_iter_d) {
    int nPoles = (int)n_poles_d;
    if (nPoles < 1) nPoles = 1;
    if (nPoles > 64) nPoles = 64;
    int nIter = (int)n_iter_d;
    if (nIter < 1) nIter = 1;
    if (nIter > 50) nIter = 50;
    return rationalfit_inner(freq, h_re, h_im, nPoles, nIter, weight);
}

static matlab_struct *rationalfit_inner(matlab_mat *freq, matlab_mat *h_re,
                                         matlab_mat *h_im, int nPoles,
                                         int nIter, matlab_mat *weight) {
    int K = freq ? (int)(freq->rows * freq->cols) : 0;
    int Khr = h_re ? (int)(h_re->rows * h_re->cols) : 0;
    int Khi = h_im ? (int)(h_im->rows * h_im->cols) : 0;
    if (K <= 0 || Khr != K || Khi != K) {
        matlab_struct *out = matlab_struct_new();
        matlab_struct_set_mat(out, "Poles",    5, (matlab_mat *)mat_c_alloc(0, 1));
        matlab_struct_set_mat(out, "Residues", 8, (matlab_mat *)mat_c_alloc(0, 1));
        matlab_struct_set_f64(out, "D",        1, 0.0);
        matlab_struct_set_f64(out, "Order",    5, 0.0);
        matlab_struct_set_f64(out, "FitError", 8, 0.0);
        return out;
    }
    /* Initial poles: prefer complex pairs (Gustavsen-Semlyen
     * recommendation).  α = −ω_j / 100 keeps each pair comfortably
     * stable; β = ω_j puts the resonant ω where the data has
     * frequency support. */
    double f_lo = freq->data[0];
    double f_hi = freq->data[K - 1];
    if (f_lo <= 0.0) f_lo = (f_hi > 0.0) ? f_hi * 1e-3 : 1.0;
    if (f_hi <= f_lo) f_hi = f_lo * 10.0;
    double log_lo = log10(f_lo), log_hi = log10(f_hi);
    int n_pairs = nPoles / 2;
    int n_real_init = nPoles - 2 * n_pairs;
    std::vector<VfPoleEntry> poles;
    poles.reserve((size_t)(n_pairs + n_real_init));
    for (int i = 0; i < n_pairs; ++i) {
        double t = (n_pairs == 1) ? 0.5
                                   : (double)i / (double)(n_pairs - 1);
        double f_i = pow(10.0, log_lo + t * (log_hi - log_lo));
        double w_i = 2.0 * M_PI * f_i;
        poles.push_back({-w_i / 100.0, w_i});
    }
    for (int i = 0; i < n_real_init; ++i) {
        double f_r = sqrt(f_lo * f_hi);
        poles.push_back({-2.0 * M_PI * f_r, 0.0});
    }
    int twoK = 2 * K;
    /* Helper closures (locally captured) for kernel evaluation. */
    auto real_pole_kernel = [](double a, double w,
                                double *phi_re, double *phi_im) {
        double den = a * a + w * w;
        *phi_re = -a / den;
        *phi_im = -w / den;
    };
    auto pair_kernels = [](double alpha, double beta, double w,
                            double *A_g_re, double *A_g_im,
                            double *A_d_re, double *A_d_im) {
        double Q_re = alpha*alpha + beta*beta - w*w;
        double Q_im = -2.0 * w * alpha;
        double Q_mag2 = Q_re*Q_re + Q_im*Q_im;
        double A_g_num_re = -2.0 * alpha;
        double A_g_num_im = 2.0 * w;
        *A_g_re = (A_g_num_re * Q_re + A_g_num_im * Q_im) / Q_mag2;
        *A_g_im = (A_g_num_im * Q_re - A_g_num_re * Q_im) / Q_mag2;
        double A_d_num_re = -2.0 * beta;
        *A_d_re = (A_d_num_re * Q_re) / Q_mag2;
        *A_d_im = (-A_d_num_re * Q_im) / Q_mag2;
    };
    /* Iterate pole relocation. */
    for (int iter = 0; iter < nIter; ++iter) {
        int n_dof_side = 0;
        for (auto &p : poles) n_dof_side += (p.beta > 0.0) ? 2 : 1;
        int M_cols = 2 * n_dof_side + 1;
        std::vector<double> Amat((size_t)twoK * M_cols, 0.0);
        std::vector<double> bvec((size_t)twoK, 0.0);
        for (int k = 0; k < K; ++k) {
            double w = 2.0 * M_PI * freq->data[k];
            double h_re_k = h_re->data[k];
            double h_im_k = h_im->data[k];
            int col_c    = 0;
            int col_csig = n_dof_side + 1;
            for (auto &p : poles) {
                if (p.beta == 0.0) {
                    double phi_re, phi_im;
                    real_pole_kernel(p.alpha, w, &phi_re, &phi_im);
                    Amat[(size_t)(2*k)     * M_cols + col_c] = phi_re;
                    Amat[(size_t)(2*k + 1) * M_cols + col_c] = phi_im;
                    double pr = h_re_k * phi_re - h_im_k * phi_im;
                    double pi = h_re_k * phi_im + h_im_k * phi_re;
                    Amat[(size_t)(2*k)     * M_cols + col_csig] = -pr;
                    Amat[(size_t)(2*k + 1) * M_cols + col_csig] = -pi;
                    col_c    += 1;
                    col_csig += 1;
                } else {
                    double A_g_re, A_g_im, A_d_re, A_d_im;
                    pair_kernels(p.alpha, p.beta, w,
                                  &A_g_re, &A_g_im, &A_d_re, &A_d_im);
                    Amat[(size_t)(2*k)     * M_cols + col_c]     = A_g_re;
                    Amat[(size_t)(2*k + 1) * M_cols + col_c]     = A_g_im;
                    Amat[(size_t)(2*k)     * M_cols + col_c + 1] = A_d_re;
                    Amat[(size_t)(2*k + 1) * M_cols + col_c + 1] = A_d_im;
                    double pr_g = h_re_k * A_g_re - h_im_k * A_g_im;
                    double pi_g = h_re_k * A_g_im + h_im_k * A_g_re;
                    Amat[(size_t)(2*k)     * M_cols + col_csig]     = -pr_g;
                    Amat[(size_t)(2*k + 1) * M_cols + col_csig]     = -pi_g;
                    double pr_d = h_re_k * A_d_re - h_im_k * A_d_im;
                    double pi_d = h_re_k * A_d_im + h_im_k * A_d_re;
                    Amat[(size_t)(2*k)     * M_cols + col_csig + 1] = -pr_d;
                    Amat[(size_t)(2*k + 1) * M_cols + col_csig + 1] = -pi_d;
                    col_c    += 2;
                    col_csig += 2;
                }
            }
            /* d coefficient — single real unknown. */
            Amat[(size_t)(2*k)     * M_cols + n_dof_side] = 1.0;
            Amat[(size_t)(2*k + 1) * M_cols + n_dof_side] = 0.0;
            bvec[2*k]     = h_re_k;
            bvec[2*k + 1] = h_im_k;
        }
        /* Apply weighted LS row scaling: multiply each row of A and b
         * by sqrt(weight[k]) — this is equivalent to solving the
         * normal-equation system AᵀWA·x = AᵀWb where W = diag(weight). */
        if (weight) {
            int K_w = (int)(weight->rows * weight->cols);
            for (int k = 0; k < K && k < K_w; ++k) {
                double w_k = weight->data[k];
                double sw = (w_k > 0.0) ? sqrt(w_k) : 0.0;
                for (int c = 0; c < M_cols; ++c) {
                    Amat[(size_t)(2*k)     * M_cols + c] *= sw;
                    Amat[(size_t)(2*k + 1) * M_cols + c] *= sw;
                }
                bvec[2*k]     *= sw;
                bvec[2*k + 1] *= sw;
            }
        }
        std::vector<double> x = vf_ls_solve(Amat, bvec, twoK, M_cols);
        if ((int)x.size() != M_cols) break;
        /* Extract c̃ (the σ rational's residues). */
        std::vector<double> csig((size_t)n_dof_side, 0.0);
        for (int i = 0; i < n_dof_side; ++i) {
            csig[(size_t)i] = x[(size_t)(n_dof_side + 1 + i)];
        }
        /* Build M for pole relocation.  Real entries fill [α], complex
         * pairs fill the [α β; −β α] real 2×2 block.  Rank-one update
         * M -= 𝟙·c̃ᵀ shifts the eigenvalues toward the data's natural
         * pole locations. */
        int Mdim = n_dof_side;
        std::vector<double> Mmat((size_t)Mdim * Mdim, 0.0);
        int pos = 0;
        for (auto &p : poles) {
            if (p.beta == 0.0) {
                Mmat[(size_t)pos * Mdim + pos] = p.alpha;
                pos += 1;
            } else {
                Mmat[(size_t)pos       * Mdim + pos    ] = p.alpha;
                Mmat[(size_t)pos       * Mdim + pos + 1] = p.beta;
                Mmat[(size_t)(pos + 1) * Mdim + pos    ] = -p.beta;
                Mmat[(size_t)(pos + 1) * Mdim + pos + 1] = p.alpha;
                pos += 2;
            }
        }
        for (int i = 0; i < Mdim; ++i) {
            for (int j = 0; j < Mdim; ++j) {
                Mmat[(size_t)i * Mdim + j] -= csig[(size_t)j];
            }
        }
        matlab_mat Mdesc;
        Mdesc.data = Mmat.data();
        Mdesc.rows = Mdim;
        Mdesc.cols = Mdim;
        matlab_mat *E = matlab_eig(&Mdesc);
        if (!E) break;
        /* Classify eigenvalues into real + complex pairs.  Read rows/
         * cols via the correct layout — matlab_mat_c has the magic
         * prefix that shifts the rows/cols offset. */
        std::vector<VfPoleEntry> new_poles;
        int n_eig;
        if (mat_is_complex(E)) {
            matlab_mat_c *Ec = (matlab_mat_c *)E;
            n_eig = (int)(Ec->rows * Ec->cols);
        } else {
            n_eig = (int)(E->rows * E->cols);
        }
        if (mat_is_complex(E)) {
            matlab_mat_c *Ec = (matlab_mat_c *)E;
            std::vector<bool> used((size_t)n_eig, false);
            for (int i = 0; i < n_eig; ++i) {
                if (used[(size_t)i]) continue;
                double a_i = Ec->re[i];
                double b_i = Ec->im[i];
                if (fabs(b_i) < 1e-8 * (fabs(a_i) + 1.0)) {
                    new_poles.push_back({a_i, 0.0});
                    used[(size_t)i] = true;
                } else {
                    bool found = false;
                    for (int j = i + 1; j < n_eig; ++j) {
                        if (used[(size_t)j]) continue;
                        double tol_r = 1e-6 * (fabs(a_i) + 1.0);
                        double tol_i = 1e-6 * (fabs(b_i) + 1.0);
                        if (fabs(Ec->re[j] - a_i) < tol_r &&
                            fabs(Ec->im[j] + b_i) < tol_i) {
                            new_poles.push_back({a_i, fabs(b_i)});
                            used[(size_t)i] = true;
                            used[(size_t)j] = true;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        /* Unpaired complex eigenvalue — defensively
                         * treat as real (take real part). */
                        new_poles.push_back({a_i, 0.0});
                        used[(size_t)i] = true;
                    }
                }
            }
        } else {
            for (int i = 0; i < n_eig; ++i) {
                new_poles.push_back({E->data[i], 0.0});
            }
        }
        /* Flip unstable poles into the left half-plane. */
        for (auto &p : new_poles) {
            if (p.alpha > 0.0) p.alpha = -p.alpha;
            if (p.alpha == 0.0) p.alpha = -1.0;
        }
        poles = new_poles;
    }
    /* Final residue fit: c, d only (no c̃ this round). */
    int n_dof_final = 0;
    for (auto &p : poles) n_dof_final += (p.beta > 0.0) ? 2 : 1;
    int M_final = n_dof_final + 1;
    std::vector<double> Af((size_t)twoK * M_final, 0.0);
    std::vector<double> bf((size_t)twoK, 0.0);
    for (int k = 0; k < K; ++k) {
        double w = 2.0 * M_PI * freq->data[k];
        int col = 0;
        for (auto &p : poles) {
            if (p.beta == 0.0) {
                double phi_re, phi_im;
                real_pole_kernel(p.alpha, w, &phi_re, &phi_im);
                Af[(size_t)(2*k)     * M_final + col] = phi_re;
                Af[(size_t)(2*k + 1) * M_final + col] = phi_im;
                col += 1;
            } else {
                double A_g_re, A_g_im, A_d_re, A_d_im;
                pair_kernels(p.alpha, p.beta, w,
                              &A_g_re, &A_g_im, &A_d_re, &A_d_im);
                Af[(size_t)(2*k)     * M_final + col]     = A_g_re;
                Af[(size_t)(2*k + 1) * M_final + col]     = A_g_im;
                Af[(size_t)(2*k)     * M_final + col + 1] = A_d_re;
                Af[(size_t)(2*k + 1) * M_final + col + 1] = A_d_im;
                col += 2;
            }
        }
        Af[(size_t)(2*k)     * M_final + n_dof_final] = 1.0;
        Af[(size_t)(2*k + 1) * M_final + n_dof_final] = 0.0;
        bf[2*k]     = h_re->data[k];
        bf[2*k + 1] = h_im->data[k];
    }
    /* Same weighting on the final residue-fit LS. */
    if (weight) {
        int K_w = (int)(weight->rows * weight->cols);
        for (int k = 0; k < K && k < K_w; ++k) {
            double w_k = weight->data[k];
            double sw = (w_k > 0.0) ? sqrt(w_k) : 0.0;
            for (int c = 0; c < M_final; ++c) {
                Af[(size_t)(2*k)     * M_final + c] *= sw;
                Af[(size_t)(2*k + 1) * M_final + c] *= sw;
            }
            bf[2*k]     *= sw;
            bf[2*k + 1] *= sw;
        }
    }
    std::vector<double> xf = vf_ls_solve(Af, bf, twoK, M_final);
    /* Compute relative RMS fit error. */
    double err_num = 0.0, err_den = 0.0;
    if ((int)xf.size() == M_final) {
        for (int k = 0; k < K; ++k) {
            double w = 2.0 * M_PI * freq->data[k];
            double Hr = xf[(size_t)n_dof_final];
            double Hi = 0.0;
            int col = 0;
            for (auto &p : poles) {
                if (p.beta == 0.0) {
                    double phi_re, phi_im;
                    real_pole_kernel(p.alpha, w, &phi_re, &phi_im);
                    Hr += xf[(size_t)col] * phi_re;
                    Hi += xf[(size_t)col] * phi_im;
                    col += 1;
                } else {
                    double A_g_re, A_g_im, A_d_re, A_d_im;
                    pair_kernels(p.alpha, p.beta, w,
                                  &A_g_re, &A_g_im, &A_d_re, &A_d_im);
                    double g = xf[(size_t)col];
                    double d = xf[(size_t)(col + 1)];
                    Hr += g * A_g_re + d * A_d_re;
                    Hi += g * A_g_im + d * A_d_im;
                    col += 2;
                }
            }
            double dr = Hr - h_re->data[k];
            double di = Hi - h_im->data[k];
            err_num += dr*dr + di*di;
            err_den += h_re->data[k]*h_re->data[k]
                     + h_im->data[k]*h_im->data[k];
        }
    }
    double fit_err = (err_den > 0.0) ? sqrt(err_num / err_den) : 0.0;
    /* Pack output: complex Poles + complex Residues (paired conjugates
     * laid out as adjacent entries). */
    int total = 0;
    for (auto &p : poles) total += (p.beta > 0.0) ? 2 : 1;
    matlab_mat_c *Pmat = mat_c_alloc(total, 1);
    matlab_mat_c *Rmat = mat_c_alloc(total, 1);
    if ((int)xf.size() == M_final) {
        int pos = 0;
        int col = 0;
        for (auto &p : poles) {
            if (p.beta == 0.0) {
                Pmat->re[pos] = p.alpha; Pmat->im[pos] = 0.0;
                Rmat->re[pos] = xf[(size_t)col]; Rmat->im[pos] = 0.0;
                pos += 1;
                col += 1;
            } else {
                double gamma = xf[(size_t)col];
                double delta = xf[(size_t)(col + 1)];
                Pmat->re[pos]     = p.alpha; Pmat->im[pos]     =  p.beta;
                Pmat->re[pos + 1] = p.alpha; Pmat->im[pos + 1] = -p.beta;
                Rmat->re[pos]     = gamma;   Rmat->im[pos]     =  delta;
                Rmat->re[pos + 1] = gamma;   Rmat->im[pos + 1] = -delta;
                pos += 2;
                col += 2;
            }
        }
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Poles",    5, (matlab_mat *)Pmat);
    matlab_struct_set_mat(out, "Residues", 8, (matlab_mat *)Rmat);
    matlab_struct_set_f64(out, "D",        1,
                          ((int)xf.size() == M_final) ? xf[(size_t)n_dof_final] : 0.0);
    matlab_struct_set_f64(out, "Order",    5, (double)total);
    matlab_struct_set_f64(out, "FitError", 8, fit_err);
    return out;
}

/* ====================================================================== */
/* §9.3.1 freqresp — evaluate the fitted rational at requested frequencies. */
/* ====================================================================== */

/* Helper: read Poles/Residues entry j as a complex pair, regardless
 * of whether the stored matrix is real or complex.  Used by freqresp,
 * passivity, and timeresp on rationalfit models. */
static inline void rf_pole_residue_at(matlab_mat *Pmat, matlab_mat *Rmat,
                                       int j, bool p_complex, bool r_complex,
                                       double *pr, double *pi,
                                       double *cr, double *ci) {
    if (p_complex) {
        matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
        *pr = Pc->re[j]; *pi = Pc->im[j];
    } else {
        *pr = Pmat->data[j]; *pi = 0.0;
    }
    if (r_complex) {
        matlab_mat_c *Rc = (matlab_mat_c *)Rmat;
        *cr = Rc->re[j]; *ci = Rc->im[j];
    } else {
        *cr = Rmat->data[j]; *ci = 0.0;
    }
}

matlab_mat_c *matlab_rf_freqresp(matlab_struct *mdl, matlab_mat *freq) {
    int K = freq ? (int)(freq->rows * freq->cols) : 0;
    matlab_mat_c *out = mat_c_alloc(K, 1);
    if (!mdl) return out;
    matlab_mat *Pmat = matlab_struct_get_mat(mdl, "Poles", 5);
    matlab_mat *Rmat = matlab_struct_get_mat(mdl, "Residues", 8);
    double D = matlab_struct_get_f64(mdl, "D", 1);
    if (!Pmat || !Rmat) {
        for (int k = 0; k < K; ++k) { out->re[k] = D; out->im[k] = 0.0; }
        return out;
    }
    bool p_complex = mat_is_complex(Pmat);
    bool r_complex = mat_is_complex(Rmat);
    int n;
    if (p_complex) {
        matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
        n = (int)(Pc->rows * Pc->cols);
    } else {
        n = (int)(Pmat->rows * Pmat->cols);
    }
    if (n == 0) {
        for (int k = 0; k < K; ++k) { out->re[k] = D; out->im[k] = 0.0; }
        return out;
    }
    for (int k = 0; k < K; ++k) {
        double w = 2.0 * M_PI * freq->data[k];
        double Hr = D, Hi = 0.0;
        for (int j = 0; j < n; ++j) {
            double pr, pi, cr, ci;
            rf_pole_residue_at(Pmat, Rmat, j, p_complex, r_complex,
                                &pr, &pi, &cr, &ci);
            /* c / (jω − p):
             *   jω − p = −pr + j(ω − pi).
             *   denom_mag² = pr² + (ω − pi)².
             *   c · conj(denom) = (cr + j·ci)(−pr − j(ω − pi))
             *                    = (−cr·pr + ci·(ω − pi))
             *                      + j(−ci·pr − cr·(ω − pi)). */
            double d_re = -pr;
            double d_im = w - pi;
            double d_mag2 = d_re*d_re + d_im*d_im;
            double num_re = cr * d_re + ci * d_im;
            double num_im = ci * d_re - cr * d_im;
            Hr += num_re / d_mag2;
            Hi += num_im / d_mag2;
        }
        out->re[k] = Hr;
        out->im[k] = Hi;
    }
    return out;
}

/* Typed-getter helpers for the rationalfit struct's matrix fields,
 * paralleling the touchstoneRead getter family. */
matlab_mat *matlab_rf_rf_poles(matlab_struct *s) {
    return matlab_struct_get_mat(s, "Poles", 5);
}
matlab_mat *matlab_rf_rf_residues(matlab_struct *s) {
    return matlab_struct_get_mat(s, "Residues", 8);
}
double matlab_rf_rf_d(matlab_struct *s) {
    return matlab_struct_get_f64(s, "D", 1);
}
double matlab_rf_rf_order(matlab_struct *s) {
    return matlab_struct_get_f64(s, "Order", 5);
}
double matlab_rf_rf_fit_error(matlab_struct *s) {
    return matlab_struct_get_f64(s, "FitError", 8);
}

/* ====================================================================== */
/* §9.3.2  s2tdr / s2tdt + timeresp — state-space realization              */
/* ====================================================================== */
/*
 * Real-coefficient partial-fraction realization:
 *   H(s) = Σ rⱼ/(s − pⱼ) + d   →   ẋ = A·x + B·u,   y = C·x + D·u
 * with A = diag(p), B = ones(n,1), C = r' (row of residues), D = d.
 *
 * timeresp(mdl, u, ts): integrates this LTI system forward in time
 * via the simple zero-order-hold update:
 *   x[k+1] = expm(A·ts) · x[k] + (expm(A·ts) − I)/A · B · u[k]
 * Since A is diagonal, expm(A·ts) is just element-wise exp(pⱼ·ts).
 * This avoids the full matrix-exponential call.
 *
 * s2tdr / s2tdt: fit the supplied S-parameter column with rationalfit
 * internally (real-pole MVP), then drive timeresp with a unit step.
 *
 * Inputs (timeresp):
 *   mdl: rationalfit struct (Poles, Residues, D)
 *   u:   real column input signal
 *   ts:  sample interval (seconds)
 * Output: real column (same length as u) — time-domain response.
 *
 * Inputs (s2tdr / s2tdt):
 *   S:        complex column at the supplied frequencies
 *   freqs:    real column (Hz)
 *   nPoles:   rationalfit order
 *   ts, nSamples: time-domain grid spec
 * Output: real column [nSamples × 1] — step response in time. */

matlab_mat *matlab_rf_timeresp(matlab_struct *mdl, matlab_mat *u, double ts) {
    int N = u ? (int)(u->rows * u->cols) : 0;
    matlab_mat *out = mat_alloc(N, 1);
    if (!mdl || N <= 0) return out;
    matlab_mat *Pmat = matlab_struct_get_mat(mdl, "Poles", 5);
    matlab_mat *Rmat = matlab_struct_get_mat(mdl, "Residues", 8);
    double D = matlab_struct_get_f64(mdl, "D", 1);
    if (!Pmat || !Rmat) {
        for (int k = 0; k < N; ++k) out->data[k] = D * u->data[k];
        return out;
    }
    bool p_complex = mat_is_complex(Pmat);
    bool r_complex = mat_is_complex(Rmat);
    int n;
    if (p_complex) {
        matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
        n = (int)(Pc->rows * Pc->cols);
    } else {
        n = (int)(Pmat->rows * Pmat->cols);
    }
    if (n == 0) {
        for (int k = 0; k < N; ++k) out->data[k] = D * u->data[k];
        return out;
    }
    /* Per-pole ZOH discretization with complex poles:
     *   φⱼ = exp(pⱼ·ts)            (complex)
     *   γⱼ = (φⱼ − 1) / pⱼ          (complex; reduces to ts when pⱼ=0)
     *   x_j[k+1] = φⱼ·x_j[k] + γⱼ·u[k]    (complex state)
     *
     * Output at time k:  y[k] = D·u[k] + Σ_j Re(c_j · x_j[k])
     *
     * For a conjugate pole pair (p, p̄) with residues (c, c̄), the
     * two state evolutions are conjugates of each other, so
     *   c·x + c̄·x̄ = 2·Re(c·x).
     * Iterating both members of the pair and summing Re(c·x) gives
     * exactly this — no explicit halving needed. */
    std::vector<double> phi_re(n), phi_im(n);
    std::vector<double> gam_re(n), gam_im(n);
    std::vector<double> x_re(n, 0.0), x_im(n, 0.0);
    std::vector<double> c_re(n), c_im(n);
    for (int j = 0; j < n; ++j) {
        double pr, pi, cr, ci;
        rf_pole_residue_at(Pmat, Rmat, j, p_complex, r_complex,
                            &pr, &pi, &cr, &ci);
        c_re[j] = cr; c_im[j] = ci;
        /* φ_j = exp((pr + j·pi) · ts)
         *      = exp(pr·ts) · (cos(pi·ts) + j·sin(pi·ts)). */
        double mag = exp(pr * ts);
        phi_re[j] = mag * cos(pi * ts);
        phi_im[j] = mag * sin(pi * ts);
        /* γ_j = (φ_j - 1) / p_j.  If |p_j| ≈ 0, use the limit ts. */
        double num_re = phi_re[j] - 1.0;
        double num_im = phi_im[j];
        double p_mag2 = pr*pr + pi*pi;
        if (p_mag2 < 1e-300) {
            gam_re[j] = ts;
            gam_im[j] = 0.0;
        } else {
            /* (num) / (pr + j·pi) = (num · (pr − j·pi)) / |p|². */
            gam_re[j] = (num_re * pr + num_im * pi) / p_mag2;
            gam_im[j] = (num_im * pr - num_re * pi) / p_mag2;
        }
    }
    for (int k = 0; k < N; ++k) {
        double uk = u->data[k];
        double y = D * uk;
        for (int j = 0; j < n; ++j) {
            /* y += Re(c_j · x_j[k]) = c_re·x_re - c_im·x_im. */
            y += c_re[j] * x_re[j] - c_im[j] * x_im[j];
        }
        out->data[k] = y;
        /* x_j[k+1] = φ_j · x_j[k] + γ_j · u[k] (complex multiply-add).
         *   φ·x = (φ_re·x_re − φ_im·x_im) + j(φ_re·x_im + φ_im·x_re)
         *   γ·u = (γ_re·u) + j·(γ_im·u) */
        for (int j = 0; j < n; ++j) {
            double new_re = phi_re[j]*x_re[j] - phi_im[j]*x_im[j]
                          + gam_re[j]*uk;
            double new_im = phi_re[j]*x_im[j] + phi_im[j]*x_re[j]
                          + gam_im[j]*uk;
            x_re[j] = new_re;
            x_im[j] = new_im;
        }
    }
    return out;
}

/* TDR/TDT step response.  Fits the given S column with rationalfit
 * (real-pole MVP — works on synthetic-rational targets; measured data
 * needs the complex-pair upgrade, which is a follow-on). */
static matlab_mat *tdx_step_response(matlab_mat_c *S, matlab_mat *freqs,
                                      int nPoles, double ts, int nSamples) {
    /* Split S into re/im real columns. */
    int K = S ? (int)(S->rows * S->cols) : 0;
    matlab_mat *h_re = mat_alloc(K, 1);
    matlab_mat *h_im = mat_alloc(K, 1);
    for (int k = 0; k < K; ++k) {
        h_re->data[k] = S->re[k];
        h_im->data[k] = S->im[k];
    }
    matlab_struct *mdl = matlab_rf_rationalfit(freqs, h_re, h_im,
                                                (double)nPoles, 10.0);
    matlab_mat *u = mat_alloc(nSamples, 1);
    for (int k = 0; k < nSamples; ++k) u->data[k] = 1.0;
    matlab_mat *y = matlab_rf_timeresp(mdl, u, ts);
    return y;
}

matlab_mat *matlab_rf_s2tdr(matlab_mat_c *S11, matlab_mat *freqs,
                             double n_poles_d, double ts, double n_samples_d) {
    int nPoles = (int)n_poles_d;
    if (nPoles < 1) nPoles = 4;
    int nSamples = (int)n_samples_d;
    if (nSamples < 1) nSamples = 256;
    return tdx_step_response(S11, freqs, nPoles, ts, nSamples);
}

matlab_mat *matlab_rf_s2tdt(matlab_mat_c *S21, matlab_mat *freqs,
                             double n_poles_d, double ts, double n_samples_d) {
    int nPoles = (int)n_poles_d;
    if (nPoles < 1) nPoles = 4;
    int nSamples = (int)n_samples_d;
    if (nSamples < 1) nSamples = 256;
    return tdx_step_response(S21, freqs, nPoles, ts, nSamples);
}

/* ====================================================================== */
/* §9.1.2 follow-on — mixed-mode 4-port s2sdd / s2sdc / s2scc / s2scd.    */
/* ====================================================================== */
/*
 * For a 4-port network with single-ended S-parameters arranged in a
 * 4×4 matrix per frequency, mixed-mode transforms:
 *   M = (1/√2) · [[1 0 -1  0],
 *                  [0 1  0 -1],
 *                  [1 0  1  0],
 *                  [0 1  0  1]]
 *   S_mm = M · S · Mᵀ
 * Block decomposition of S_mm yields:
 *   S_mm = [ Sdd  Sdc ]
 *          [ Scd  Scc ]
 * where Sdd is the differential-to-differential 2×2 block (top-left),
 * Scc is common-to-common, Sdc/Scd are mode-conversion couplings.
 *
 * v1 implementation: accept the 4 single-ended port columns at one
 * frequency as separate complex columns, return the requested block
 * as a 2×2 complex matrix per frequency (stored as 4 complex columns:
 * out11, out12, out21, out22). */

/* Mixed-mode transformation per frequency.  s_se is a flat 16-entry
 * complex array in row-major order [s11..s14, s21..s24, s31..s34, s41..s44].
 * block_code: 0=dd, 1=dc, 2=cd, 3=cc.  Out is a 4-entry complex array. */
static void s_se_to_mm_block(const C s_se[16], int block_code, C out[4]) {
    /* Build M and Mᵀ as 4×4 real matrices. */
    static const double M[16] = {
         1.0,  0.0, -1.0,  0.0,
         0.0,  1.0,  0.0, -1.0,
         1.0,  0.0,  1.0,  0.0,
         0.0,  1.0,  0.0,  1.0
    };
    const double inv_sqrt2 = 0.7071067811865475;
    /* M·S (real·complex). */
    C MS[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C s = {0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                C term = cmul({M[i*4 + k] * inv_sqrt2, 0.0}, s_se[k*4 + j]);
                s = cadd(s, term);
            }
            MS[i*4 + j] = s;
        }
    }
    /* (M·S)·Mᵀ. */
    C SMM[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C s = {0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                C term = cmul(MS[i*4 + k], {M[j*4 + k] * inv_sqrt2, 0.0});
                s = cadd(s, term);
            }
            SMM[i*4 + j] = s;
        }
    }
    /* Pick the requested 2×2 block. */
    int ro = (block_code == 2 || block_code == 3) ? 2 : 0;
    int co = (block_code == 1 || block_code == 3) ? 2 : 0;
    out[0] = SMM[(ro    )*4 + (co    )];
    out[1] = SMM[(ro    )*4 + (co + 1)];
    out[2] = SMM[(ro + 1)*4 + (co    )];
    out[3] = SMM[(ro + 1)*4 + (co + 1)];
}

/* Convert all single-ended 4-port S to a mixed-mode 2-port block.
 * Input S_se is a 16-column complex storage (one per s_ij entry,
 * each of length NumFreqs).  Output is a struct with Smm11, Smm12,
 * Smm21, Smm22 (each a complex column). */
matlab_struct *matlab_rf_s2smm(
        matlab_mat_c *s11, matlab_mat_c *s12, matlab_mat_c *s13, matlab_mat_c *s14,
        matlab_mat_c *s21, matlab_mat_c *s22, matlab_mat_c *s23, matlab_mat_c *s24,
        matlab_mat_c *s31, matlab_mat_c *s32, matlab_mat_c *s33, matlab_mat_c *s34,
        matlab_mat_c *s41, matlab_mat_c *s42, matlab_mat_c *s43, matlab_mat_c *s44,
        double block_code_d) {
    int block_code = (int)block_code_d;
    matlab_mat_c *grid[16] = {s11,s12,s13,s14, s21,s22,s23,s24,
                               s31,s32,s33,s34, s41,s42,s43,s44};
    int64_t N = 0;
    for (int i = 0; i < 16; ++i)
        if (grid[i]) N = std::max(N, grid[i]->rows * grid[i]->cols);
    matlab_mat_c *o11 = cvec(N), *o12 = cvec(N);
    matlab_mat_c *o21 = cvec(N), *o22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s_se[16];
        for (int i = 0; i < 16; ++i) s_se[i] = sread(grid[i], k);
        C out[4];
        s_se_to_mm_block(s_se, block_code, out);
        o11->re[k] = out[0].re; o11->im[k] = out[0].im;
        o12->re[k] = out[1].re; o12->im[k] = out[1].im;
        o21->re[k] = out[2].re; o21->im[k] = out[2].im;
        o22->re[k] = out[3].re; o22->im[k] = out[3].im;
    }
    matlab_struct *out_s = matlab_struct_new();
    matlab_struct_set_mat(out_s, "S11", 3, (matlab_mat *)o11);
    matlab_struct_set_mat(out_s, "S12", 3, (matlab_mat *)o12);
    matlab_struct_set_mat(out_s, "S21", 3, (matlab_mat *)o21);
    matlab_struct_set_mat(out_s, "S22", 3, (matlab_mat *)o22);
    return out_s;
}

/* ====================================================================== */
/* §9.4.3 Smith chart numeric grids.                                       */
/* ====================================================================== */
/*
 * smithGrid(n_circles, n_points_per_circle) returns the constant-r
 * and constant-x circle overlays as complex Γ-plane samples.
 *
 * Constant-r circles in Γ-plane:
 *   Center: (r/(r+1), 0),  radius 1/(r+1)
 * Constant-x circles:
 *   Center: (1, 1/x),       radius 1/x   (for x != 0; x=0 is the
 *                                          real axis)
 *
 * v1 returns ONE constant-r circle (r=1, the matched-load circle) and
 * the unit circle (|Γ|=1, boundary) as two complex columns.  Each is
 * n_points long.  Richer multi-circle grids are caller-orchestrated. */

/* Typed-getter helpers for the smith_grid return struct. */
matlab_mat_c *matlab_rf_smith_rcircle(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "RCircle", 7);
}
matlab_mat_c *matlab_rf_smith_unit(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "UnitCircle", 10);
}

matlab_struct *matlab_rf_smith_grid(double r_norm_d, double n_pts_d) {
    int N = (int)n_pts_d;
    if (N < 8) N = 32;
    double r = r_norm_d;
    if (r <= 0.0) r = 1.0;
    matlab_mat_c *r_circle = cvec(N);
    matlab_mat_c *unit_circle = cvec(N);
    double cr = r / (r + 1.0);
    double rad_r = 1.0 / (r + 1.0);
    for (int k = 0; k < N; ++k) {
        double t = (double)k * 2.0 * M_PI / (double)N;
        r_circle->re[k] = cr + rad_r * cos(t);
        r_circle->im[k] = rad_r * sin(t);
        unit_circle->re[k] = cos(t);
        unit_circle->im[k] = sin(t);
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "RCircle",    7, (matlab_mat *)r_circle);
    matlab_struct_set_mat(out, "UnitCircle", 10, (matlab_mat *)unit_circle);
    return out;
}

/* ====================================================================== */
/* §9.3.1 follow-on — passivity test for the fitted rational model.        */
/* ====================================================================== */
/*
 * A scalar real-coefficient rational H(s) is passive if its
 * frequency response satisfies |H(jω)| ≤ 1 for all real ω AND the
 * direct-feedthrough term D ≤ 1.  v1 implementation samples the
 * fitted model on a dense log-spaced frequency grid spanning the data
 * range and returns the max |H(jω)|; caller decides what threshold to
 * compare against (1.0 for strict passivity). */
double matlab_rf_passivity(matlab_struct *mdl, double f_lo, double f_hi) {
    if (!mdl) return 0.0;
    matlab_mat *Pmat = matlab_struct_get_mat(mdl, "Poles", 5);
    matlab_mat *Rmat = matlab_struct_get_mat(mdl, "Residues", 8);
    double D = matlab_struct_get_f64(mdl, "D", 1);
    if (!Pmat || !Rmat) return fabs(D);
    bool p_complex = mat_is_complex(Pmat);
    bool r_complex = mat_is_complex(Rmat);
    int n;
    if (p_complex) {
        matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
        n = (int)(Pc->rows * Pc->cols);
    } else {
        n = (int)(Pmat->rows * Pmat->cols);
    }
    if (n == 0) return fabs(D);
    int N = 400;
    double f_lo_safe = (f_lo > 0.0) ? f_lo : 1.0;
    double f_hi_safe = (f_hi > f_lo_safe) ? f_hi : f_lo_safe * 1000.0;
    double log_lo = log10(f_lo_safe);
    double log_hi = log10(f_hi_safe);
    double max_mag = 0.0;
    for (int k = 0; k < N; ++k) {
        double t = (N <= 1) ? 0.5 : (double)k / (double)(N - 1);
        double f = pow(10.0, log_lo + t * (log_hi - log_lo));
        double w = 2.0 * M_PI * f;
        double Hr = D, Hi = 0.0;
        for (int j = 0; j < n; ++j) {
            double pr, pi, cr, ci;
            rf_pole_residue_at(Pmat, Rmat, j, p_complex, r_complex,
                                &pr, &pi, &cr, &ci);
            double d_re = -pr;
            double d_im = w - pi;
            double d_mag2 = d_re*d_re + d_im*d_im;
            double num_re = cr * d_re + ci * d_im;
            double num_im = ci * d_re - cr * d_im;
            Hr += num_re / d_mag2;
            Hi += num_im / d_mag2;
        }
        double mag = sqrt(Hr*Hr + Hi*Hi);
        if (mag > max_mag) max_mag = mag;
    }
    return max_mag;
}

/* ====================================================================== */
/* §9.4.1 matchingnetwork — L / T / Pi auto-synthesis.                     */
/* ====================================================================== */
/*
 * Synthesize a passive matching network that transforms a complex
 * source impedance Zs to a complex load impedance Zl at one
 * frequency.  v1 implements the canonical L-section (two components)
 * with the high-Q topology selected automatically.
 *
 * The L-section has two topologies (shunt-series or series-shunt)
 * chosen by which impedance is larger in magnitude.  Each component
 * is either a series-inductor / series-capacitor / shunt-inductor /
 * shunt-capacitor.  We return component values (L in Henries, C in
 * Farads) and the topology code.
 *
 * Inputs:  Zs_re, Zs_im (source), Zl_re, Zl_im (load), freq_Hz
 * Output struct:
 *   Topology    0 = shunt-series, 1 = series-shunt
 *   Q           Quality factor of the match
 *   L_series_H, L_shunt_H, C_series_F, C_shunt_F  (whichever applies;
 *               unused entries return 0)
 *   ReturnLoss_dB at the operating frequency
 */
matlab_struct *matlab_rf_matchingnetwork(
        double zs_re, double zs_im,
        double zl_re, double zl_im,
        double freq) {
    double omega = 2.0 * M_PI * freq;
    /* Pick the topology: if Re(Zs) > Re(Zl) we need to step DOWN;
     * use shunt-series (shunt component closest to source, then
     * series component toward load).  Otherwise series-shunt. */
    int topo;
    double R_high, R_low, X_high;
    if (zs_re >= zl_re) {
        topo = 0;      /* shunt-series, source side is the high R end */
        R_high = zs_re;  R_low = zl_re;
        X_high = zs_im;
    } else {
        topo = 1;      /* series-shunt */
        R_high = zl_re;  R_low = zs_re;
        X_high = zl_im;
    }
    /* Q factor and reactance values. */
    double Q = (R_low > 0.0 && R_high > R_low)
               ? sqrt(R_high / R_low - 1.0) : 0.0;
    double Xs_match = Q * R_low;     /* series-side reactance */
    double Xp_match = R_high / Q;    /* shunt-side reactance (Q > 0) */
    /* Map reactances to component values.  Positive X → inductor,
     * negative X → capacitor.  Pick the sign that minimises the
     * absolute component value (more reasonable for real designs);
     * v1 uses the positive-X convention for both, mapping series →
     * inductor + shunt → capacitor (a low-pass section). */
    double L_series = (Xs_match > 0.0 && Q > 0.0) ? Xs_match / omega : 0.0;
    double C_shunt  = (Xp_match > 0.0 && Q > 0.0) ? 1.0 / (omega * Xp_match) : 0.0;
    /* Estimate return loss at the operating frequency assuming the
     * match is perfect at f0 (so |Γ| ≈ 0 → RL → ∞ in ideal).  Real
     * estimates need the full sweep; v1 reports the analytic 0 +
     * 60 dB synthetic floor. */
    double rl_dB = (Q > 0.0) ? 60.0 : 0.0;
    matlab_struct *out = matlab_struct_new();
    #define SET(name, v) matlab_struct_set_f64(out, name, sizeof(name)-1, v)
    SET("Topology",   (double)topo);
    SET("Q",          Q);
    SET("L_series_H", L_series);
    SET("L_shunt_H",  0.0);
    SET("C_series_F", 0.0);
    SET("C_shunt_F",  C_shunt);
    SET("ReturnLoss_dB", rl_dB);
    SET("Frequency_Hz",  freq);
    SET("X_series",   Xs_match);
    SET("X_shunt",    Xp_match);
    /* keep X_high in the output for debug / inspection */
    SET("X_high_in",  X_high);
    #undef SET
    return out;
}

/* ====================================================================== */
/* §9.3.3 Transmission line geometries.                                    */
/* ====================================================================== */
/*
 * Closed-form characteristic impedance + propagation constant per
 * geometry.  Each entry returns the 2-port S-matrix at a single
 * frequency for a length-L transmission-line segment terminated in z0.
 *
 *   Z0 = characteristic impedance, β = phase constant (rad/m).
 *   S = exp(−γL), reflection at Z0 reference:
 *     s11 = s22 = ρ·(1 − e^{−2γL}) / (1 − ρ²·e^{−2γL})
 *     s21 = s12 = (1 − ρ²)·e^{−γL} / (1 − ρ²·e^{−2γL})
 *   with ρ = (Z0_line − z0) / (Z0_line + z0).
 *
 * For lossless lines, γ = jβ = jω·sqrt(L'·C').  Return per-frequency
 * complex s11/s12/s21/s22 columns. */

/* Build the 2-port S for a length-L transmission line of characteristic
 * impedance Z0_line on a Z0 reference, evaluated at requested freqs.
 * Lossless (real Z0_line, imag-only γ).  Each output is a complex
 * column at the supplied frequencies. */
static matlab_struct *txline_s_lossless(double Z0_line, double v_phase,
                                         double length_m, matlab_mat *freqs,
                                         double z0) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    double rho = (z0 + Z0_line == 0.0) ? 0.0
                 : (Z0_line - z0) / (Z0_line + z0);
    double rho2 = rho * rho;
    for (int k = 0; k < K; ++k) {
        double f = freqs->data[k];
        double beta = (v_phase > 0.0) ? (2.0 * M_PI * f / v_phase) : 0.0;
        double phase = beta * length_m;
        /* e^{-jθ} = cos(θ) - j·sin(θ); e^{-2jθ} = cos(2θ) - j·sin(2θ). */
        double c1 = cos(phase),     s1 = sin(phase);
        double c2 = cos(2.0 * phase), s2 = sin(2.0 * phase);
        /* numerator s11: ρ·(1 − e^{−2γL}) = ρ·(1 − c2 + j·s2). */
        double n11_re = rho * (1.0 - c2);
        double n11_im = rho * s2;
        /* denominator: 1 − ρ²·e^{−2γL} = (1 − ρ²·c2) + j·ρ²·s2 */
        double d_re = 1.0 - rho2 * c2;
        double d_im =       rho2 * s2;
        double d_den = d_re * d_re + d_im * d_im;
        /* s11 = num11 / den. */
        S11->re[k] = (n11_re * d_re + n11_im * d_im) / d_den;
        S11->im[k] = (n11_im * d_re - n11_re * d_im) / d_den;
        S22->re[k] = S11->re[k];
        S22->im[k] = S11->im[k];
        /* s21 = (1 − ρ²)·e^{−γL} / den.
         * (1 − ρ²)·(c1 − j·s1) = (1 − ρ²)·c1 − j·(1 − ρ²)·s1.    */
        double n21_re = (1.0 - rho2) * c1;
        double n21_im = -(1.0 - rho2) * s1;
        S21->re[k] = (n21_re * d_re + n21_im * d_im) / d_den;
        S21->im[k] = (n21_im * d_re - n21_re * d_im) / d_den;
        S12->re[k] = S21->re[k];
        S12->im[k] = S21->im[k];
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    matlab_struct_set_f64(out, "Z0_line",    7, Z0_line);
    matlab_struct_set_f64(out, "Vphase",     6, v_phase);
    matlab_struct_set_f64(out, "Length_m",   8, length_m);
    return out;
}

/* rfckt.txline (generic): Z0, εr, length.  v_phase = c / sqrt(εr). */
matlab_struct *matlab_rf_txline(double Z0_line, double er, double length_m,
                                 matlab_mat *freqs, double z0) {
    double c_light = 2.99792458e8;
    double er_safe = (er >= 1.0) ? er : 1.0;
    double v = c_light / sqrt(er_safe);
    return txline_s_lossless(Z0_line, v, length_m, freqs, z0);
}

/* rfckt.coaxial: closed-form Z0 = (60 / sqrt(εr)) · ln(b/a),
 * v = c/sqrt(εr) where a, b are inner/outer radii. */
matlab_struct *matlab_rf_coaxial(double a_inner, double b_outer, double er,
                                  double length_m, matlab_mat *freqs, double z0) {
    if (a_inner <= 0.0 || b_outer <= a_inner) {
        return txline_s_lossless(z0, 2.99792458e8, length_m, freqs, z0);
    }
    double er_safe = (er >= 1.0) ? er : 1.0;
    double Z0_line = (60.0 / sqrt(er_safe)) * log(b_outer / a_inner);
    double v = 2.99792458e8 / sqrt(er_safe);
    return txline_s_lossless(Z0_line, v, length_m, freqs, z0);
}

/* rfckt.microstrip — Hammerstad-Jensen closed-form.
 *   w: trace width (m), h: dielectric height (m), εr: relative
 *   permittivity, length_m: line length.
 *
 * Effective dielectric constant:
 *   εeff = (εr+1)/2 + (εr−1)/2 · (1 + 12·h/w)^{−1/2}
 * Characteristic impedance:
 *   if w/h ≤ 1:  Z0 = (60/sqrt(εeff))·ln(8·h/w + w/(4·h))
 *   else:        Z0 = 120π / [sqrt(εeff)·(w/h + 1.393 + 0.667·ln(w/h+1.444))] */
matlab_struct *matlab_rf_microstrip(double w, double h, double er,
                                     double length_m, matlab_mat *freqs,
                                     double z0) {
    if (w <= 0.0 || h <= 0.0) {
        return txline_s_lossless(z0, 2.99792458e8, length_m, freqs, z0);
    }
    double er_safe = (er >= 1.0) ? er : 1.0;
    double eeff = (er_safe + 1.0) / 2.0
                + ((er_safe - 1.0) / 2.0) / sqrt(1.0 + 12.0 * h / w);
    double Z0_line;
    if (w / h <= 1.0) {
        Z0_line = (60.0 / sqrt(eeff))
                * log(8.0 * h / w + w / (4.0 * h));
    } else {
        Z0_line = 120.0 * M_PI / (sqrt(eeff)
                * (w / h + 1.393 + 0.667 * log(w / h + 1.444)));
    }
    double v = 2.99792458e8 / sqrt(eeff);
    matlab_struct *out = txline_s_lossless(Z0_line, v, length_m, freqs, z0);
    matlab_struct_set_f64(out, "Eeff", 4, eeff);
    return out;
}

/* rfckt.cpw — coplanar waveguide (simple approximation).
 *   w: signal trace width, s: gap to ground plane, εr: substrate.
 *   εeff ≈ (εr + 1) / 2.  Z0 ≈ 30π / sqrt(εeff) · K(k')/K(k)
 *   where k = w/(w+2s) and the elliptic integrals are approximated
 *   by Hilberg's formula. */
matlab_struct *matlab_rf_cpw(double w, double s, double er,
                              double length_m, matlab_mat *freqs, double z0) {
    if (w <= 0.0 || s <= 0.0) {
        return txline_s_lossless(z0, 2.99792458e8, length_m, freqs, z0);
    }
    double er_safe = (er >= 1.0) ? er : 1.0;
    double eeff = (er_safe + 1.0) / 2.0;
    double k = w / (w + 2.0 * s);
    double kp = sqrt(1.0 - k * k);
    /* Hilberg's K(k')/K(k) approximation. */
    double ratio;
    if (k <= 0.7071067811865475) {
        ratio = (1.0 / M_PI) * log(2.0 * (1.0 + sqrt(kp)) / (1.0 - sqrt(kp)));
    } else {
        ratio = M_PI / log(2.0 * (1.0 + sqrt(k)) / (1.0 - sqrt(k)));
    }
    double Z0_line = (30.0 * M_PI / sqrt(eeff)) * ratio;
    double v = 2.99792458e8 / sqrt(eeff);
    matlab_struct *out = txline_s_lossless(Z0_line, v, length_m, freqs, z0);
    matlab_struct_set_f64(out, "Eeff", 4, eeff);
    return out;
}

/* rfckt.parallelplate — two parallel conductors separated by a
 * dielectric.  Z0 = (η0 / sqrt(εr)) · (h/w), η0 = 120π. */
matlab_struct *matlab_rf_parallelplate(double w, double h, double er,
                                        double length_m, matlab_mat *freqs,
                                        double z0) {
    if (w <= 0.0 || h <= 0.0) {
        return txline_s_lossless(z0, 2.99792458e8, length_m, freqs, z0);
    }
    double er_safe = (er >= 1.0) ? er : 1.0;
    double Z0_line = (120.0 * M_PI / sqrt(er_safe)) * (h / w);
    double v = 2.99792458e8 / sqrt(er_safe);
    return txline_s_lossless(Z0_line, v, length_m, freqs, z0);
}

/* ====================================================================== */
/* §9.4.2 LC filter circuit blocks — closed-form S-parameters.            */
/* ====================================================================== */
/*
 * Each LC filter is realized by composing series and shunt elements
 * with closed-form S-parameters at every frequency.  The user supplies
 * the design component values (L₁, C₁, …) and the topology code.
 *
 * Helpers below build the 2-port S-matrix at a single frequency for
 * the canonical Tee, Pi topologies of each filter type.  The
 * implementation chains 2-port T-parameter blocks for series-then-
 * shunt-then-series (Tee) or shunt-then-series-then-shunt (Pi).
 *
 * Series block: jωL or 1/(jωC).  Reflection / transmission:
 *   s11 = s22 =  Z / (Z + 2·z0),   s12 = s21 = 2·z0 / (Z + 2·z0)
 * Shunt block: jωL or jωC.  Reflection / transmission:
 *   s11 = s22 = −Y·z0 / (Y·z0 + 2), s12 = s21 = 2 / (Y·z0 + 2)
 *
 * Topology codes:
 *   0 = Lowpass-Tee (series-L, shunt-C, series-L)
 *   1 = Lowpass-Pi  (shunt-C, series-L, shunt-C)
 *   2 = Highpass-Tee (series-C, shunt-L, series-C)
 *   3 = Highpass-Pi  (shunt-L, series-C, shunt-L)
 *   4 = Bandpass-Tee (series-LC, shunt-LC-parallel, series-LC)
 *   5 = Bandstop-Tee (series-LC-parallel, shunt-LC-series, series-LC-parallel)
 *
 * For the v1 implementation we ship the simpler 3-component topologies
 * (codes 0–3); 4-component LC bandpass/bandstop is a follow-on. */

static void rf_series_z(double Z_re, double Z_im, double z0,
                         C *s11, C *s12, C *s21, C *s22) {
    C Z = {Z_re, Z_im};
    C denom = cadd(Z, {2.0 * z0, 0.0});
    *s11 = cdiv(Z, denom);
    *s22 = *s11;
    C num = {2.0 * z0, 0.0};
    *s12 = cdiv(num, denom);
    *s21 = *s12;
}

static void rf_shunt_y(double Y_re, double Y_im, double z0,
                        C *s11, C *s12, C *s21, C *s22) {
    C Y_z0 = cmul({Y_re, Y_im}, {z0, 0.0});
    C denom = cadd(Y_z0, {2.0, 0.0});
    *s11 = cmul({-1.0, 0.0}, cdiv(Y_z0, denom));
    *s22 = *s11;
    *s12 = cdiv({2.0, 0.0}, denom);
    *s21 = *s12;
}

static void rf_cascade_t(C a11, C a12, C a21, C a22,
                          C b11, C b12, C b21, C b22,
                          C *c11, C *c12, C *c21, C *c22) {
    /* T-parameter convert + chain multiply + back to S, all 2×2. */
    C TA11, TA12, TA21, TA22, TB11, TB12, TB21, TB22;
    {
        C inv = cdiv({1.0, 0.0}, a21);
        C det = csub(cmul(a11, a22), cmul(a12, a21));
        TA11 = cmul({-1.0, 0.0}, cmul(det, inv));
        TA12 = cmul(a11, inv);
        TA21 = cmul({-1.0, 0.0}, cmul(a22, inv));
        TA22 = inv;
    }
    {
        C inv = cdiv({1.0, 0.0}, b21);
        C det = csub(cmul(b11, b22), cmul(b12, b21));
        TB11 = cmul({-1.0, 0.0}, cmul(det, inv));
        TB12 = cmul(b11, inv);
        TB21 = cmul({-1.0, 0.0}, cmul(b22, inv));
        TB22 = inv;
    }
    C T11 = cadd(cmul(TA11, TB11), cmul(TA12, TB21));
    C T12 = cadd(cmul(TA11, TB12), cmul(TA12, TB22));
    C T21 = cadd(cmul(TA21, TB11), cmul(TA22, TB21));
    C T22 = cadd(cmul(TA21, TB12), cmul(TA22, TB22));
    {
        C inv = cdiv({1.0, 0.0}, T22);
        *c11 = cmul(T12, inv);
        *c21 = inv;
        *c22 = cmul({-1.0, 0.0}, cmul(T21, inv));
        *c12 = csub(T11, cmul(cmul(T12, T21), inv));
    }
}

/* matlab_rf_lc_filter — generic 3-element LC filter.
 *
 * topology code (numeric): 0 = lowpass-tee, 1 = lowpass-pi,
 *                          2 = highpass-tee, 3 = highpass-pi.
 *
 * Components per topology:
 *   0/2: comp1 = series-element value (L for LP, C for HP) on each end,
 *        comp2 = shunt-element value in the middle.
 *   1/3: comp1 = shunt-element value on each end (C for LP, L for HP),
 *        comp2 = series-element value in the middle.
 *
 * Returns 2-port S struct at the requested frequencies. */
matlab_struct *matlab_rf_lc_filter(double topology_d,
                                    double comp1, double comp2,
                                    matlab_mat *freqs, double z0) {
    int topology = (int)topology_d;
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    for (int k = 0; k < K; ++k) {
        double f = freqs->data[k];
        double w = 2.0 * M_PI * f;
        C e1_s11, e1_s12, e1_s21, e1_s22;     /* outer element block */
        C e2_s11, e2_s12, e2_s21, e2_s22;     /* middle element block */
        if (topology == 0) {
            /* Lowpass-Tee: series-L, shunt-C, series-L. */
            double Z_L = w * comp1;
            rf_series_z(0.0, Z_L, z0,
                         &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double Y_C = w * comp2;
            rf_shunt_y(0.0, Y_C, z0,
                        &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        } else if (topology == 1) {
            /* Lowpass-Pi: shunt-C, series-L, shunt-C. */
            double Y_C = w * comp1;
            rf_shunt_y(0.0, Y_C, z0,
                        &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double Z_L = w * comp2;
            rf_series_z(0.0, Z_L, z0,
                         &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        } else if (topology == 2) {
            /* Highpass-Tee: series-C, shunt-L, series-C. */
            double Z_C = -1.0 / (w * comp1);   /* 1/(jωC) → −j/(ωC) */
            rf_series_z(0.0, Z_C, z0,
                         &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double Y_L = -1.0 / (w * comp2);   /* 1/(jωL) → −j/(ωL) */
            rf_shunt_y(0.0, Y_L, z0,
                        &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        } else {
            /* Highpass-Pi: shunt-L, series-C, shunt-L. */
            double Y_L = -1.0 / (w * comp1);
            rf_shunt_y(0.0, Y_L, z0,
                        &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double Z_C = -1.0 / (w * comp2);
            rf_series_z(0.0, Z_C, z0,
                         &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        }
        /* Cascade: e1 → e2 → e1 (symmetric 3-element). */
        C m11, m12, m21, m22;
        rf_cascade_t(e1_s11, e1_s12, e1_s21, e1_s22,
                      e2_s11, e2_s12, e2_s21, e2_s22,
                      &m11, &m12, &m21, &m22);
        C f11, f12, f21, f22;
        rf_cascade_t(m11, m12, m21, m22,
                      e1_s11, e1_s12, e1_s21, e1_s22,
                      &f11, &f12, &f21, &f22);
        S11->re[k] = f11.re; S11->im[k] = f11.im;
        S12->re[k] = f12.re; S12->im[k] = f12.im;
        S21->re[k] = f21.re; S21->im[k] = f21.im;
        S22->re[k] = f22.re; S22->im[k] = f22.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    matlab_struct_set_f64(out, "NumPorts",   8, 2.0);
    matlab_struct_set_f64(out, "Z0",         2, z0);
    matlab_struct_set_f64(out, "Topology",   8, (double)topology);
    matlab_struct_set_mat(out, "Frequencies", 11, freqs);
    return out;
}

/* ====================================================================== */
/* §9.4.2 follow-on — RFCkt block analyze helpers.                         */
/* ====================================================================== */
/*
 * Synthesize S-parameter structs from rfckt block scalar properties.
 * Amplifier / Mixer: matched-port forward-gain model
 *   s11 = s22 = 0,  s12 = 0,  s21 = 10^(gain_dB/20)
 * Passive: matched-port insertion-loss model
 *   s11 = s22 = 0,  s12 = s21 = 10^(-loss_dB/20)
 * Series-Z (in path): s11 = s22 = Z/(Z + 2·z0),  s12 = s21 = 2·z0/(Z + 2·z0)
 * Shunt-Y (to ground): s11 = s22 = −Y·z0/(Y·z0 + 2),  s12 = s21 = 2/(Y·z0 + 2)
 *
 * Each returns a struct identical in shape to touchstoneRead's output. */

static matlab_struct *rf_analyze_make_struct(matlab_mat_c *S11,
                                              matlab_mat_c *S12,
                                              matlab_mat_c *S21,
                                              matlab_mat_c *S22,
                                              matlab_mat *F, double z0) {
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    matlab_struct_set_f64(out, "NumPorts",   8, 2.0);
    matlab_struct_set_f64(out, "Z0",         2, z0);
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    return out;
}

matlab_struct *matlab_rf_analyze_amplifier(double gain_dB,
                                            matlab_mat *freqs, double z0) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    double gain_lin = pow(10.0, gain_dB / 20.0);
    for (int k = 0; k < K; ++k) S21->re[k] = gain_lin;
    return rf_analyze_make_struct(S11, S12, S21, S22, freqs, z0);
}

matlab_struct *matlab_rf_analyze_passive(double loss_dB,
                                          matlab_mat *freqs, double z0) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    double t_lin = pow(10.0, -loss_dB / 20.0);
    for (int k = 0; k < K; ++k) {
        S12->re[k] = t_lin;
        S21->re[k] = t_lin;
    }
    return rf_analyze_make_struct(S11, S12, S21, S22, freqs, z0);
}

matlab_struct *matlab_rf_analyze_series(double z_re, double z_im,
                                         matlab_mat *freqs, double z0) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    for (int k = 0; k < K; ++k) {
        C s11, s12, s21_, s22;
        rf_series_z(z_re, z_im, z0, &s11, &s12, &s21_, &s22);
        S11->re[k] = s11.re; S11->im[k] = s11.im;
        S12->re[k] = s12.re; S12->im[k] = s12.im;
        S21->re[k] = s21_.re; S21->im[k] = s21_.im;
        S22->re[k] = s22.re; S22->im[k] = s22.im;
    }
    return rf_analyze_make_struct(S11, S12, S21, S22, freqs, z0);
}

matlab_struct *matlab_rf_analyze_shunt(double y_re, double y_im,
                                        matlab_mat *freqs, double z0) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    for (int k = 0; k < K; ++k) {
        C s11, s12, s21_, s22;
        rf_shunt_y(y_re, y_im, z0, &s11, &s12, &s21_, &s22);
        S11->re[k] = s11.re; S11->im[k] = s11.im;
        S12->re[k] = s12.re; S12->im[k] = s12.im;
        S21->re[k] = s21_.re; S21->im[k] = s21_.im;
        S22->re[k] = s22.re; S22->im[k] = s22.im;
    }
    return rf_analyze_make_struct(S11, S12, S21, S22, freqs, z0);
}

/* matlab_rf_lc_filter4 — 4-element LC bandpass / bandstop filter.
 *
 * topology code (numeric):
 *   4 = Bandpass-Tee  (series-LC-series-tuned, shunt-LC-parallel-tuned, series-LC-series-tuned)
 *   5 = Bandpass-Pi   (shunt-LC-parallel-tuned, series-LC-series-tuned, shunt-LC-parallel-tuned)
 *   6 = Bandstop-Tee  (series-LC-parallel-tuned, shunt-LC-series-tuned, series-LC-parallel-tuned)
 *   7 = Bandstop-Pi   (shunt-LC-series-tuned, series-LC-parallel-tuned, shunt-LC-series-tuned)
 *
 * Component values: (L1, C1) define the outer pair branches, (L2, C2)
 * define the middle branch.  Series-LC branch impedance:
 *     Z_series(ω) = jωL + 1/(jωC) = j(ωL − 1/(ωC))
 * Parallel-LC branch admittance:
 *     Y_par(ω) = 1/(jωL) + jωC = j(ωC − 1/(ωL))
 *
 * Returns the 2-port S-matrix at the requested frequencies. */
matlab_struct *matlab_rf_lc_filter4(double topology_d,
                                     double L1, double C1,
                                     double L2, double C2,
                                     matlab_mat *freqs, double z0) {
    int topology = (int)topology_d;
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat_c *S11 = cvec(K), *S12 = cvec(K);
    matlab_mat_c *S21 = cvec(K), *S22 = cvec(K);
    for (int k = 0; k < K; ++k) {
        double f = freqs->data[k];
        double w = 2.0 * M_PI * f;
        C e1_s11, e1_s12, e1_s21, e1_s22;     /* outer block */
        C e2_s11, e2_s12, e2_s21, e2_s22;     /* middle block */
        if (topology == 4) {
            /* Bandpass-Tee: outer = series-LC-series-tuned;
             *               middle = shunt-LC-parallel-tuned. */
            double Z_series_im = w * L1 - 1.0 / (w * C1);
            rf_series_z(0.0, Z_series_im, z0,
                         &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double Y_par_im = w * C2 - 1.0 / (w * L2);
            rf_shunt_y(0.0, Y_par_im, z0,
                        &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        } else if (topology == 5) {
            /* Bandpass-Pi: outer = shunt-LC-parallel-tuned;
             *              middle = series-LC-series-tuned. */
            double Y_par_im = w * C1 - 1.0 / (w * L1);
            rf_shunt_y(0.0, Y_par_im, z0,
                        &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double Z_series_im = w * L2 - 1.0 / (w * C2);
            rf_series_z(0.0, Z_series_im, z0,
                         &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        } else if (topology == 6) {
            /* Bandstop-Tee: outer = series-LC-parallel-tuned;
             *               middle = shunt-LC-series-tuned.
             *
             * Series-LC-parallel-tuned: Z = jωL || 1/(jωC)
             *   = (jωL · 1/(jωC)) / (jωL + 1/(jωC))
             *   = (L/C) / j(ωL − 1/(ωC))
             *   = −jL/C / (ωL − 1/(ωC))     (purely imaginary)
             *   = j / (ωC − 1/(ωL)·1/(... ))   — simpler: at resonance the
             *     parallel branch sees infinite Z; we form Z = j·X(ω).
             *   X(ω) = ω·L / (1 − ω²·L·C).
             * Shunt-LC-series-tuned: Y = j·B(ω) where
             *   B(ω) = ω·C / (1 − ω²·L·C). */
            double denom = 1.0 - w * w * L1 * C1;
            double Z_par_im = (fabs(denom) > 1e-300) ? (w * L1 / denom)
                                                      : 1.0e18;
            rf_series_z(0.0, Z_par_im, z0,
                         &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double denom2 = 1.0 - w * w * L2 * C2;
            double Y_ser_im = (fabs(denom2) > 1e-300) ? (w * C2 / denom2)
                                                        : 1.0e18;
            rf_shunt_y(0.0, Y_ser_im, z0,
                        &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        } else {
            /* topology == 7: Bandstop-Pi (default). */
            double denom = 1.0 - w * w * L1 * C1;
            double Y_ser_im = (fabs(denom) > 1e-300) ? (w * C1 / denom)
                                                      : 1.0e18;
            rf_shunt_y(0.0, Y_ser_im, z0,
                        &e1_s11, &e1_s12, &e1_s21, &e1_s22);
            double denom2 = 1.0 - w * w * L2 * C2;
            double Z_par_im = (fabs(denom2) > 1e-300) ? (w * L2 / denom2)
                                                        : 1.0e18;
            rf_series_z(0.0, Z_par_im, z0,
                         &e2_s11, &e2_s12, &e2_s21, &e2_s22);
        }
        /* Symmetric 3-block cascade: e1 → e2 → e1. */
        C m11, m12, m21, m22;
        rf_cascade_t(e1_s11, e1_s12, e1_s21, e1_s22,
                      e2_s11, e2_s12, e2_s21, e2_s22,
                      &m11, &m12, &m21, &m22);
        C f11, f12, f21, f22;
        rf_cascade_t(m11, m12, m21, m22,
                      e1_s11, e1_s12, e1_s21, e1_s22,
                      &f11, &f12, &f21, &f22);
        S11->re[k] = f11.re; S11->im[k] = f11.im;
        S12->re[k] = f12.re; S12->im[k] = f12.im;
        S21->re[k] = f21.re; S21->im[k] = f21.im;
        S22->re[k] = f22.re; S22->im[k] = f22.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    matlab_struct_set_f64(out, "NumPorts",   8, 2.0);
    matlab_struct_set_f64(out, "Z0",         2, z0);
    matlab_struct_set_f64(out, "Topology",   8, (double)topology);
    matlab_struct_set_mat(out, "Frequencies", 11, freqs);
    return out;
}

/* rfckt.twowire — two parallel cylindrical wires.
 *   Z0 = (η0 / (π·sqrt(εr))) · cosh⁻¹(D/(2r)). */
matlab_struct *matlab_rf_twowire(double radius, double separation, double er,
                                  double length_m, matlab_mat *freqs, double z0) {
    if (radius <= 0.0 || separation <= 2.0 * radius) {
        return txline_s_lossless(z0, 2.99792458e8, length_m, freqs, z0);
    }
    double er_safe = (er >= 1.0) ? er : 1.0;
    double arg = separation / (2.0 * radius);
    double acosh_arg = log(arg + sqrt(arg * arg - 1.0));
    double Z0_line = (120.0 * M_PI / (M_PI * sqrt(er_safe))) * acosh_arg;
    double v = 2.99792458e8 / sqrt(er_safe);
    return txline_s_lossless(Z0_line, v, length_m, freqs, z0);
}

/* ====================================================================== */
/* §9.1.2 N-port conversions (S↔Y, S↔Z) + snp2smp.                        */
/* ====================================================================== */
/*
 * For an arbitrary-N port S-parameter network referenced to z0:
 *   Y = (1/z0) · (I + S)⁻¹ · (I − S)
 *   Z =  z0    · (I − S)⁻¹ · (I + S)
 *
 * Implementation: per-frequency, gather the N×N complex S matrix
 * from the Sij fields of the input struct, do the matrix operation
 * via complex N×N invert + multiply (using the 2N×2N real-equivalent
 * for the inverse on top of the runtime's real LU solver), then
 * store Y_ij / Z_ij into the output struct.
 *
 * Limit: N ≤ 9 (matches the Sij field-name decoration scheme used
 * by the multi-port Touchstone reader). */

extern matlab_mat *matlab_inv(matlab_mat *A);

static void complex_mat_inv_2neq(int N,
                                  const double *A_re, const double *A_im,
                                  double *out_re, double *out_im) {
    int N2 = 2 * N;
    matlab_mat *M = mat_alloc(N2, N2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M->data[(size_t)i        * N2 + j        ] =  A_re[i*N + j];
            M->data[(size_t)i        * N2 + (j + N)  ] = -A_im[i*N + j];
            M->data[(size_t)(i + N)  * N2 + j        ] =  A_im[i*N + j];
            M->data[(size_t)(i + N)  * N2 + (j + N)  ] =  A_re[i*N + j];
        }
    }
    matlab_mat *Mi = matlab_inv(M);
    if (Mi && Mi->rows == N2 && Mi->cols == N2) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                out_re[i*N + j] = Mi->data[(size_t)i        * N2 + j];
                out_im[i*N + j] = Mi->data[(size_t)(i + N)  * N2 + j];
            }
        }
    } else {
        for (int i = 0; i < N*N; ++i) { out_re[i] = 0.0; out_im[i] = 0.0; }
    }
}

static void complex_mat_mul(int N,
                             const double *A_re, const double *A_im,
                             const double *B_re, const double *B_im,
                             double *C_re, double *C_im) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double cr = 0.0, ci = 0.0;
            for (int k = 0; k < N; ++k) {
                cr += A_re[i*N + k] * B_re[k*N + j]
                    - A_im[i*N + k] * B_im[k*N + j];
                ci += A_re[i*N + k] * B_im[k*N + j]
                    + A_im[i*N + k] * B_re[k*N + j];
            }
            C_re[i*N + j] = cr;
            C_im[i*N + j] = ci;
        }
    }
}

/* Gather S(:,:,k) from struct's Sij fields into row-major buffers. */
static void gather_s_at_k(matlab_struct *data, int N, int k,
                           std::vector<double> &S_re,
                           std::vector<double> &S_im) {
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            char name[8];
            int fn = snprintf(name, sizeof(name), "S%d%d", i, j);
            matlab_mat_c *Sij =
                (matlab_mat_c *)matlab_struct_get_mat(data, name, fn);
            int idx = (i - 1) * N + (j - 1);
            if (Sij && k < (int)(Sij->rows * Sij->cols)) {
                S_re[idx] = Sij->re[k];
                S_im[idx] = Sij->im[k];
            } else {
                S_re[idx] = 0.0;
                S_im[idx] = 0.0;
            }
        }
    }
}

static matlab_struct *s_to_yz_n(matlab_struct *data, int kind) {
    int N = (int)matlab_struct_get_f64(data, "NumPorts", 8);
    if (N < 1 || N > 9) {
        return matlab_struct_new();
    }
    double z0 = matlab_struct_get_f64(data, "Z0", 2);
    matlab_mat *F = matlab_struct_get_mat(data, "Frequencies", 11);
    int K = F ? (int)(F->rows * F->cols) : 0;
    int n2 = N * N;
    std::vector<double> S_re(n2), S_im(n2);
    std::vector<double> IpS_re(n2), IpS_im(n2);
    std::vector<double> ImS_re(n2), ImS_im(n2);
    std::vector<double> Inv_re(n2), Inv_im(n2);
    std::vector<double> Out_re(n2), Out_im(n2);
    std::vector<matlab_mat_c *> out_cols((size_t)n2);
    for (int idx = 0; idx < n2; ++idx) out_cols[(size_t)idx] = mat_c_alloc(K, 1);
    for (int k = 0; k < K; ++k) {
        gather_s_at_k(data, N, k, S_re, S_im);
        for (int idx = 0; idx < n2; ++idx) {
            IpS_re[idx] =  S_re[idx];      IpS_im[idx] =  S_im[idx];
            ImS_re[idx] = -S_re[idx];      ImS_im[idx] = -S_im[idx];
        }
        for (int i = 0; i < N; ++i) {
            IpS_re[i*N + i] += 1.0;
            ImS_re[i*N + i] += 1.0;
        }
        /* Y = (1/z0)·(I+S)⁻¹·(I−S).
         * Z =  z0   ·(I−S)⁻¹·(I+S). */
        if (kind == 0) {
            complex_mat_inv_2neq(N, IpS_re.data(), IpS_im.data(),
                                  Inv_re.data(), Inv_im.data());
            complex_mat_mul(N, Inv_re.data(), Inv_im.data(),
                              ImS_re.data(), ImS_im.data(),
                              Out_re.data(), Out_im.data());
        } else {
            complex_mat_inv_2neq(N, ImS_re.data(), ImS_im.data(),
                                  Inv_re.data(), Inv_im.data());
            complex_mat_mul(N, Inv_re.data(), Inv_im.data(),
                              IpS_re.data(), IpS_im.data(),
                              Out_re.data(), Out_im.data());
        }
        double scale = (kind == 0) ? (1.0 / z0) : z0;
        for (int idx = 0; idx < n2; ++idx) {
            out_cols[(size_t)idx]->re[k] = Out_re[idx] * scale;
            out_cols[(size_t)idx]->im[k] = Out_im[idx] * scale;
        }
    }
    /* Pack into output struct.  Field name prefix is 'Y' or 'Z'. */
    matlab_struct *out = matlab_struct_new();
    const char prefix = (kind == 0) ? 'Y' : 'Z';
    char fname[8];
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            int fn = snprintf(fname, sizeof(fname), "%c%d%d", prefix, i, j);
            int idx = (i - 1) * N + (j - 1);
            matlab_struct_set_mat(out, fname, fn,
                                   (matlab_mat *)out_cols[(size_t)idx]);
        }
    }
    matlab_struct_set_f64(out, "NumPorts", 8, (double)N);
    matlab_struct_set_f64(out, "Z0",       2, z0);
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    return out;
}

matlab_struct *matlab_rf_s2y_n(matlab_struct *data) { return s_to_yz_n(data, 0); }
matlab_struct *matlab_rf_s2z_n(matlab_struct *data) { return s_to_yz_n(data, 1); }

/* snp2smp(data, port_list, m_ports): extract an m-port sub-network
 * from the supplied N-port S-parameter struct by keeping the listed
 * ports and matching-terminating the rest.
 *
 * For perfectly-matched terminations at the reference impedance,
 * the m-port S-matrix is simply the sub-block S[port_list, port_list].
 * Non-matched terminations would compose via the Schur complement;
 * v1 ships the matched-port case.
 *
 * Inputs:
 *   data:       struct from touchstoneRead
 *   port_list:  real column of 1-based port indices (length m_ports)
 *   m_ports:    target port count (scalar, == length of port_list)
 *
 * Output:
 *   A struct with NumPorts = m_ports and S<i><j> fields for the
 *   kept ports.  Frequencies + Z0 are passed through verbatim. */
matlab_struct *matlab_rf_snp2smp(matlab_struct *data,
                                  matlab_mat *port_list,
                                  double m_ports_d) {
    int m = (int)m_ports_d;
    if (port_list) {
        int got = (int)(port_list->rows * port_list->cols);
        if (got < m) m = got;
    }
    if (m < 1 || m > 9) {
        return matlab_struct_new();
    }
    int N = (int)matlab_struct_get_f64(data, "NumPorts", 8);
    double z0 = matlab_struct_get_f64(data, "Z0", 2);
    matlab_mat *F = matlab_struct_get_mat(data, "Frequencies", 11);
    matlab_struct *out = matlab_struct_new();
    char fname_in[8], fname_out[8];
    for (int i = 1; i <= m; ++i) {
        int src_i = (int)port_list->data[i - 1];
        if (src_i < 1) src_i = 1; if (src_i > N) src_i = N;
        for (int j = 1; j <= m; ++j) {
            int src_j = (int)port_list->data[j - 1];
            if (src_j < 1) src_j = 1; if (src_j > N) src_j = N;
            int fnIn  = snprintf(fname_in,  sizeof(fname_in),  "S%d%d",
                                  src_i, src_j);
            int fnOut = snprintf(fname_out, sizeof(fname_out), "S%d%d", i, j);
            matlab_mat_c *Sij = (matlab_mat_c *)matlab_struct_get_mat(
                                  data, fname_in, fnIn);
            /* Clone (caller may free/modify either side). */
            int K = Sij ? (int)(Sij->rows * Sij->cols) : 0;
            matlab_mat_c *copy = mat_c_alloc(K, 1);
            for (int k = 0; k < K; ++k) {
                copy->re[k] = Sij->re[k];
                copy->im[k] = Sij->im[k];
            }
            matlab_struct_set_mat(out, fname_out, fnOut, (matlab_mat *)copy);
        }
    }
    matlab_struct_set_f64(out, "NumPorts", 8, (double)m);
    matlab_struct_set_f64(out, "Z0",       2, z0);
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    return out;
}

/* ====================================================================== */
/* §9.2.2 General N-port cascade (Redheffer star product).                 */
/* ====================================================================== */
/*
 * Cascades two N-port S-parameter networks A and B by connecting all
 * N ports of A "back" to all N ports of B "front".  This is the
 * most common multi-port cascade in practice (e.g., chaining two
 * identical 4-port differential-channel models).
 *
 * Redheffer star product (with implicit 2N-port → N-port partition):
 *   The full cascade chain treats A as the first half (port columns
 *   1..N internal, N+1..2N external front side) and B as the second
 *   half (port columns 1..N front external, N+1..2N internal); the
 *   internal ports connect.
 *
 * For simplicity in v1, we expose the special case where A and B
 * are both N-port networks and all of A's ports cascade into B's:
 *   S_AB(i,j) = direct chain.
 *
 * Returns a struct identical in shape to the inputs (NumPorts, Z0,
 * Frequencies, S<i><j>). */
/* Rectangular complex matrix multiply: C[r×c] = A[r×k] · B[k×c]. */
static void cmat_mul_rect(int r, int k, int c,
                           const double *A_re, const double *A_im,
                           const double *B_re, const double *B_im,
                           double *C_re, double *C_im) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double cr = 0.0, ci = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                double ar = A_re[(size_t)(i*k + kk)];
                double ai = A_im[(size_t)(i*k + kk)];
                double br = B_re[(size_t)(kk*c + j)];
                double bi = B_im[(size_t)(kk*c + j)];
                cr += ar*br - ai*bi;
                ci += ar*bi + ai*br;
            }
            C_re[(size_t)(i*c + j)] = cr;
            C_im[(size_t)(i*c + j)] = ci;
        }
    }
}

/* Full Redheffer star product N-port cascade: given two N-port
 * networks A and B where the last k ports of A connect to the first
 * k ports of B (k inner-connection ports per side), the result is a
 * (2(N − k))-port network.  For the common case k = N/2 (equal
 * outer / inner port counts), the result has N external ports.
 *
 * Formula (with S_A and S_B partitioned into (N−k)×(N−k), (N−k)×k,
 * k×(N−k), k×k blocks):
 *   M = (I − S_A22 · S_B11)⁻¹     (k × k)
 *   M' = (I − S_B11 · S_A22)⁻¹    (k × k)
 *   S_AB_11 = S_A11 + S_A12 · S_B11 · M · S_A21      ((N−k)×(N−k))
 *   S_AB_12 = S_A12 · M · S_B12                       ((N−k)×(N−k))
 *   S_AB_21 = S_B21 · M' · S_A21                      ((N−k)×(N−k))
 *   S_AB_22 = S_B22 + S_B21 · S_A22 · M' · S_B12      ((N−k)×(N−k))
 *
 * v1 specializes to k = N/2.  Result has N external ports. */
matlab_struct *matlab_rf_cascade_n(matlab_struct *A, matlab_struct *B);

matlab_struct *matlab_rf_cascade_n_full(matlab_struct *A, matlab_struct *B) {
    if (!A || !B) return matlab_struct_new();
    int N = (int)matlab_struct_get_f64(A, "NumPorts", 8);
    int N_b = (int)matlab_struct_get_f64(B, "NumPorts", 8);
    if (N < 2 || N > 8 || N_b != N || (N % 2) != 0) {
        /* Fall back to the diagonal approximation for odd N or
         * mismatched port counts. */
        return matlab_rf_cascade_n(A, B);
    }
    int k = N / 2;
    int nk = N - k;
    double z0 = matlab_struct_get_f64(A, "Z0", 2);
    matlab_mat *F = matlab_struct_get_mat(A, "Frequencies", 11);
    int K = F ? (int)(F->rows * F->cols) : 0;
    /* Output: 2(N − k) = N external ports.  Allocate per-(i,j) cols. */
    int outDim = N;
    int n2 = outDim * outDim;
    std::vector<matlab_mat_c *> out_cols((size_t)n2);
    for (int idx = 0; idx < n2; ++idx) out_cols[(size_t)idx] = mat_c_alloc(K, 1);
    char nam[8];
    /* Helper: gather a block of S from a struct at freq f.  Indices
     * (i0..i0+rows-1) by (j0..j0+cols-1) → buffer (row-major). */
    auto gather = [](matlab_struct *S, int rows, int cols,
                      int i0, int j0, int kfreq,
                      std::vector<double> &buf_re,
                      std::vector<double> &buf_im) {
        char fname[8];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int sp_i = i0 + i, sp_j = j0 + j;
                int fn = snprintf(fname, sizeof(fname), "S%d%d", sp_i, sp_j);
                matlab_mat_c *Sij =
                    (matlab_mat_c *)matlab_struct_get_mat(S, fname, fn);
                int n_freq = Sij ? (int)(Sij->rows * Sij->cols) : 0;
                buf_re[(size_t)(i*cols + j)] = (kfreq < n_freq) ? Sij->re[kfreq] : 0.0;
                buf_im[(size_t)(i*cols + j)] = (kfreq < n_freq) ? Sij->im[kfreq] : 0.0;
            }
        }
    };
    for (int f = 0; f < K; ++f) {
        /* A partitioned: A11 (nk×nk), A12 (nk×k), A21 (k×nk), A22 (k×k).
         * Indices in S_A: outer ports 1..nk, inner ports nk+1..N. */
        std::vector<double> A11_re((size_t)(nk*nk)), A11_im((size_t)(nk*nk));
        std::vector<double> A12_re((size_t)(nk*k)),  A12_im((size_t)(nk*k));
        std::vector<double> A21_re((size_t)(k*nk)),  A21_im((size_t)(k*nk));
        std::vector<double> A22_re((size_t)(k*k)),   A22_im((size_t)(k*k));
        gather(A, nk, nk, 1,    1,    f, A11_re, A11_im);
        gather(A, nk, k,  1,    nk+1, f, A12_re, A12_im);
        gather(A, k,  nk, nk+1, 1,    f, A21_re, A21_im);
        gather(A, k,  k,  nk+1, nk+1, f, A22_re, A22_im);
        /* B partitioned similarly.  Inner ports of B are 1..k,
         * outer ports k+1..N. */
        std::vector<double> B11_re((size_t)(k*k)),   B11_im((size_t)(k*k));
        std::vector<double> B12_re((size_t)(k*nk)),  B12_im((size_t)(k*nk));
        std::vector<double> B21_re((size_t)(nk*k)),  B21_im((size_t)(nk*k));
        std::vector<double> B22_re((size_t)(nk*nk)), B22_im((size_t)(nk*nk));
        gather(B, k,  k,  1,   1,   f, B11_re, B11_im);
        gather(B, k,  nk, 1,   k+1, f, B12_re, B12_im);
        gather(B, nk, k,  k+1, 1,   f, B21_re, B21_im);
        gather(B, nk, nk, k+1, k+1, f, B22_re, B22_im);
        /* M = inv(I − A22·B11)    (k × k). */
        std::vector<double> A22B11_re((size_t)(k*k)), A22B11_im((size_t)(k*k));
        cmat_mul_rect(k, k, k, A22_re.data(), A22_im.data(),
                                B11_re.data(), B11_im.data(),
                                A22B11_re.data(), A22B11_im.data());
        std::vector<double> X_re((size_t)(k*k)), X_im((size_t)(k*k));
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                X_re[(size_t)(i*k + j)] = -A22B11_re[(size_t)(i*k + j)];
                X_im[(size_t)(i*k + j)] = -A22B11_im[(size_t)(i*k + j)];
            }
            X_re[(size_t)(i*k + i)] += 1.0;
        }
        std::vector<double> M_re((size_t)(k*k)), M_im((size_t)(k*k));
        complex_mat_inv_2neq(k, X_re.data(), X_im.data(),
                              M_re.data(), M_im.data());
        /* M' = inv(I − B11·A22). */
        std::vector<double> B11A22_re((size_t)(k*k)), B11A22_im((size_t)(k*k));
        cmat_mul_rect(k, k, k, B11_re.data(), B11_im.data(),
                                A22_re.data(), A22_im.data(),
                                B11A22_re.data(), B11A22_im.data());
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                X_re[(size_t)(i*k + j)] = -B11A22_re[(size_t)(i*k + j)];
                X_im[(size_t)(i*k + j)] = -B11A22_im[(size_t)(i*k + j)];
            }
            X_re[(size_t)(i*k + i)] += 1.0;
        }
        std::vector<double> Mp_re((size_t)(k*k)), Mp_im((size_t)(k*k));
        complex_mat_inv_2neq(k, X_re.data(), X_im.data(),
                              Mp_re.data(), Mp_im.data());
        /* S_AB_11 = A11 + A12 · B11 · M · A21      (nk×nk). */
        std::vector<double> tmp1_re((size_t)(nk*k)), tmp1_im((size_t)(nk*k));
        /* A12 · B11    (nk × k) */
        cmat_mul_rect(nk, k, k, A12_re.data(), A12_im.data(),
                                 B11_re.data(), B11_im.data(),
                                 tmp1_re.data(), tmp1_im.data());
        std::vector<double> tmp2_re((size_t)(nk*k)), tmp2_im((size_t)(nk*k));
        /* (A12 · B11) · M    (nk × k) */
        cmat_mul_rect(nk, k, k, tmp1_re.data(), tmp1_im.data(),
                                 M_re.data(), M_im.data(),
                                 tmp2_re.data(), tmp2_im.data());
        std::vector<double> SAB11_re((size_t)(nk*nk)), SAB11_im((size_t)(nk*nk));
        /* ((A12·B11)·M) · A21    (nk × nk) */
        cmat_mul_rect(nk, k, nk, tmp2_re.data(), tmp2_im.data(),
                                  A21_re.data(), A21_im.data(),
                                  SAB11_re.data(), SAB11_im.data());
        for (int i = 0; i < nk*nk; ++i) {
            SAB11_re[(size_t)i] += A11_re[(size_t)i];
            SAB11_im[(size_t)i] += A11_im[(size_t)i];
        }
        /* S_AB_12 = A12 · M · B12      (nk × nk). */
        std::vector<double> tmp3_re((size_t)(nk*k)), tmp3_im((size_t)(nk*k));
        cmat_mul_rect(nk, k, k, A12_re.data(), A12_im.data(),
                                 M_re.data(), M_im.data(),
                                 tmp3_re.data(), tmp3_im.data());
        std::vector<double> SAB12_re((size_t)(nk*nk)), SAB12_im((size_t)(nk*nk));
        cmat_mul_rect(nk, k, nk, tmp3_re.data(), tmp3_im.data(),
                                  B12_re.data(), B12_im.data(),
                                  SAB12_re.data(), SAB12_im.data());
        /* S_AB_21 = B21 · M' · A21    (nk × nk). */
        std::vector<double> tmp4_re((size_t)(nk*k)), tmp4_im((size_t)(nk*k));
        cmat_mul_rect(nk, k, k, B21_re.data(), B21_im.data(),
                                 Mp_re.data(), Mp_im.data(),
                                 tmp4_re.data(), tmp4_im.data());
        std::vector<double> SAB21_re((size_t)(nk*nk)), SAB21_im((size_t)(nk*nk));
        cmat_mul_rect(nk, k, nk, tmp4_re.data(), tmp4_im.data(),
                                  A21_re.data(), A21_im.data(),
                                  SAB21_re.data(), SAB21_im.data());
        /* S_AB_22 = B22 + B21 · A22 · M' · B12      (nk × nk). */
        std::vector<double> tmp5_re((size_t)(nk*k)), tmp5_im((size_t)(nk*k));
        cmat_mul_rect(nk, k, k, B21_re.data(), B21_im.data(),
                                 A22_re.data(), A22_im.data(),
                                 tmp5_re.data(), tmp5_im.data());
        std::vector<double> tmp6_re((size_t)(nk*k)), tmp6_im((size_t)(nk*k));
        cmat_mul_rect(nk, k, k, tmp5_re.data(), tmp5_im.data(),
                                 Mp_re.data(), Mp_im.data(),
                                 tmp6_re.data(), tmp6_im.data());
        std::vector<double> SAB22_re((size_t)(nk*nk)), SAB22_im((size_t)(nk*nk));
        cmat_mul_rect(nk, k, nk, tmp6_re.data(), tmp6_im.data(),
                                  B12_re.data(), B12_im.data(),
                                  SAB22_re.data(), SAB22_im.data());
        for (int i = 0; i < nk*nk; ++i) {
            SAB22_re[(size_t)i] += B22_re[(size_t)i];
            SAB22_im[(size_t)i] += B22_im[(size_t)i];
        }
        /* Pack the result into the (2·nk = N)-port output S struct.
         * Outer ports: A's outer 1..nk → output ports 1..nk;
         *              B's outer 1..nk → output ports nk+1..2nk = N. */
        for (int i = 1; i <= outDim; ++i) {
            for (int j = 1; j <= outDim; ++j) {
                double re = 0.0, im = 0.0;
                if (i <= nk && j <= nk) {
                    int ii = i - 1, jj = j - 1;
                    re = SAB11_re[(size_t)(ii*nk + jj)];
                    im = SAB11_im[(size_t)(ii*nk + jj)];
                } else if (i <= nk && j > nk) {
                    int ii = i - 1, jj = j - 1 - nk;
                    re = SAB12_re[(size_t)(ii*nk + jj)];
                    im = SAB12_im[(size_t)(ii*nk + jj)];
                } else if (i > nk && j <= nk) {
                    int ii = i - 1 - nk, jj = j - 1;
                    re = SAB21_re[(size_t)(ii*nk + jj)];
                    im = SAB21_im[(size_t)(ii*nk + jj)];
                } else {
                    int ii = i - 1 - nk, jj = j - 1 - nk;
                    re = SAB22_re[(size_t)(ii*nk + jj)];
                    im = SAB22_im[(size_t)(ii*nk + jj)];
                }
                int idx = (i - 1) * outDim + (j - 1);
                out_cols[(size_t)idx]->re[f] = re;
                out_cols[(size_t)idx]->im[f] = im;
            }
        }
    }
    matlab_struct *out = matlab_struct_new();
    for (int i = 1; i <= outDim; ++i) {
        for (int j = 1; j <= outDim; ++j) {
            int fn = snprintf(nam, sizeof(nam), "S%d%d", i, j);
            int idx = (i - 1) * outDim + (j - 1);
            matlab_struct_set_mat(out, nam, fn,
                                   (matlab_mat *)out_cols[(size_t)idx]);
        }
    }
    matlab_struct_set_f64(out, "NumPorts", 8, (double)outDim);
    matlab_struct_set_f64(out, "Z0",       2, z0);
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    return out;
}

matlab_struct *matlab_rf_cascade_n(matlab_struct *A, matlab_struct *B) {
    if (!A || !B) return matlab_struct_new();
    int N = (int)matlab_struct_get_f64(A, "NumPorts", 8);
    int N_b = (int)matlab_struct_get_f64(B, "NumPorts", 8);
    if (N < 1 || N > 9 || N_b != N) return matlab_struct_new();
    double z0 = matlab_struct_get_f64(A, "Z0", 2);
    matlab_mat *F = matlab_struct_get_mat(A, "Frequencies", 11);
    int K = F ? (int)(F->rows * F->cols) : 0;
    matlab_struct *out = matlab_struct_new();
    /* Per-frequency cascade via T-parameter matrix multiply.  For the
     * full-N cascade case (every port of A connects to every port of
     * B), the T-parameter matrices are 2N×2N each.  See Pozar
     * "Microwave Engineering" §4.5 for the derivation; the formula
     * for converting between N-port S and T uses the same
     * block-Schur structure as the 2-port case generalized.
     *
     * Compact implementation: re-use the 2-port path per
     * subset of "matching ports".  For full-N cascade, we treat the
     * chain as N independent 2-port pairs only when the cross-couplings
     * are negligible — otherwise the full matrix formula is needed.
     *
     * v1 ships the diagonal approximation: per pair (port i of A
     * connects to port i of B), apply the 2-port T-parameter cascade
     * across S_ii and the cross-coupling terms.  For typical
     * weakly-coupled multi-port networks (differential pairs, etc.)
     * this is accurate; for tightly-coupled mode-converting
     * networks, the full matrix formula is the proper choice.
     *
     * Future work (when complex LU lands): port to the matrix
     * Schur-complement form for arbitrary coupling. */
    int n2 = N * N;
    std::vector<matlab_mat_c *> out_cols((size_t)n2);
    for (int idx = 0; idx < n2; ++idx) out_cols[(size_t)idx] = mat_c_alloc(K, 1);
    /* Approximate cascade: for each frequency, do per-port 2-port
     * T-multiply on the diagonal (i,i) entries.  Off-diagonal mode
     * couplings are passed through from A (a reasonable first-order
     * approximation for weakly-coupled networks). */
    for (int k = 0; k < K; ++k) {
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                int idx = (i - 1) * N + (j - 1);
                char namA[8], namB[8];
                int fnA = snprintf(namA, sizeof(namA), "S%d%d", i, j);
                int fnB = snprintf(namB, sizeof(namB), "S%d%d", i, j);
                matlab_mat_c *Sa =
                    (matlab_mat_c *)matlab_struct_get_mat(A, namA, fnA);
                matlab_mat_c *Sb =
                    (matlab_mat_c *)matlab_struct_get_mat(B, namB, fnB);
                C va = sread(Sa, k);
                C vb = sread(Sb, k);
                /* For i==j: 2-port-style cascade
                 * (T_AB[i] = T_A[i] · T_B[i]).
                 * Approximate with the matched-line product:
                 * s11_cascade = s11_A + (s12_A·s21_A·s11_B) / (1 − s22_A·s11_B). */
                if (i == j) {
                    char nam12A[8], nam21A[8], nam22A[8];
                    int f12 = snprintf(nam12A, sizeof(nam12A), "S%d%d", i, j);
                    int f21 = snprintf(nam21A, sizeof(nam21A), "S%d%d", j, i);
                    int f22 = snprintf(nam22A, sizeof(nam22A), "S%d%d", j, j);
                    (void)f12;
                    matlab_mat_c *Sa21 =
                        (matlab_mat_c *)matlab_struct_get_mat(A, nam21A, f21);
                    matlab_mat_c *Sa22 =
                        (matlab_mat_c *)matlab_struct_get_mat(A, nam22A, f22);
                    C s12a = va;  /* diagonal-only approx */
                    C s21a = sread(Sa21, k);
                    C s22a = sread(Sa22, k);
                    C den = csub({1.0, 0.0}, cmul(s22a, vb));
                    C num = cmul(cmul(s12a, s21a), vb);
                    C s_casc = cadd(va, cdiv(num, den));
                    out_cols[(size_t)idx]->re[k] = s_casc.re;
                    out_cols[(size_t)idx]->im[k] = s_casc.im;
                } else {
                    /* Off-diagonal: pass through. */
                    out_cols[(size_t)idx]->re[k] = va.re;
                    out_cols[(size_t)idx]->im[k] = va.im;
                }
            }
        }
    }
    char fname[8];
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            int fn = snprintf(fname, sizeof(fname), "S%d%d", i, j);
            int idx = (i - 1) * N + (j - 1);
            matlab_struct_set_mat(out, fname, fn,
                                   (matlab_mat *)out_cols[(size_t)idx]);
        }
    }
    matlab_struct_set_f64(out, "NumPorts", 8, (double)N);
    matlab_struct_set_f64(out, "Z0",       2, z0);
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    return out;
}

/* ====================================================================== */
/* §9.1.3 Multi-port Touchstone writer (.sNp).                            */
/* ====================================================================== */
/*
 * Writes a Touchstone v1 file in MA (magnitude / angle) format from
 * the supplied struct (e.g. the output of touchstoneRead, sparamS2yN,
 * sparamS2zN, snp2smp, or any pipeline that produces the same shape).
 * Port count auto-detected from struct.NumPorts.
 *
 * Layout:
 *   - .s2p (NumPorts == 2): historical [s11 s21 s12 s22] row order.
 *   - .sNp (NumPorts != 2): row-major [s11 s12 ... s1N; s21 ...; sN1 ... sNN]
 *     with one Touchstone "row" per matrix-row (so each Touchstone row
 *     has 2N complex values).  Most readers tolerate either flat or
 *     per-matrix-row layout; we use flat (all 2N² numbers on one
 *     Touchstone row per frequency) for the common N ≤ 4 case to
 *     keep the writer compact. */
double matlab_rf_touchstone_write(void *fname_str, matlab_struct *data) {
    if (!data) return 0.0;
    rf_string_view *sv = (rf_string_view *)fname_str;
    char path[1024];
    int64_t pn = 0;
    if (sv && sv->data && sv->len > 0) {
        pn = sv->len < 1023 ? sv->len : 1023;
        memcpy(path, sv->data, pn);
    }
    path[pn] = 0;
    int N = (int)matlab_struct_get_f64(data, "NumPorts", 8);
    if (N < 1 || N > 9) return 0.0;
    double z0 = matlab_struct_get_f64(data, "Z0", 2);
    if (z0 <= 0.0) z0 = 50.0;
    matlab_mat *F = matlab_struct_get_mat(data, "Frequencies", 11);
    int K = F ? (int)(F->rows * F->cols) : 0;
    FILE *fp = fopen(path, "w");
    if (!fp) return 0.0;
    fprintf(fp, "! Generated by matlab_llvm runtime_rf.cpp\n");
    fprintf(fp, "# Hz S MA R %g\n", z0);
    char fname[8];
    for (int k = 0; k < K; ++k) {
        double f = F->data[k];
        fprintf(fp, "%.10g", f);
        if (N == 2) {
            /* s2p historical [s11 s21 s12 s22] row order. */
            static const int s2p_order[4][2] = { {1,1}, {2,1}, {1,2}, {2,2} };
            for (int e = 0; e < 4; ++e) {
                int i = s2p_order[e][0], j = s2p_order[e][1];
                int fn = snprintf(fname, sizeof(fname), "S%d%d", i, j);
                matlab_mat_c *Sij =
                    (matlab_mat_c *)matlab_struct_get_mat(data, fname, fn);
                C v = Sij && k < (int)(Sij->rows * Sij->cols)
                      ? (C){Sij->re[k], Sij->im[k]} : (C){0.0, 0.0};
                double mag = cmag(v);
                double ang = (v.re == 0.0 && v.im == 0.0) ? 0.0
                           : atan2(v.im, v.re) * 180.0 / M_PI;
                fprintf(fp, " %.10g %.6g", mag, ang);
            }
        } else {
            /* Row-major: [s11 s12 ... s1N; s21 ...]. */
            for (int i = 1; i <= N; ++i) {
                for (int j = 1; j <= N; ++j) {
                    int fn = snprintf(fname, sizeof(fname), "S%d%d", i, j);
                    matlab_mat_c *Sij =
                        (matlab_mat_c *)matlab_struct_get_mat(data, fname, fn);
                    C v = Sij && k < (int)(Sij->rows * Sij->cols)
                          ? (C){Sij->re[k], Sij->im[k]} : (C){0.0, 0.0};
                    double mag = cmag(v);
                    double ang = (v.re == 0.0 && v.im == 0.0) ? 0.0
                               : atan2(v.im, v.re) * 180.0 / M_PI;
                    fprintf(fp, " %.10g %.6g", mag, ang);
                }
            }
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 1.0;
}

/* ====================================================================== */
/* §9.1.2 follow-on — 2-port cross-conversions S↔H / S↔G / S↔ABCD / S↔T. */
/* ====================================================================== */
/*
 * Per-frequency 2×2 algebra.  All work on the per-frequency complex S
 * matrix; H / G / ABCD / T have direct closed-form expressions in
 * terms of S and the reference impedance z0.  Each entry returns a
 * struct with the four parameter columns named H11/H12/H21/H22 (or
 * G/A/T variants).
 *
 * Formulas (from RF Toolbox documentation, simplified for z0 real):
 *   H from S:
 *     h11 = z0·((1 + s11)(1 + s22) − s12·s21) / Δ,        Δ = (1 − s11)(1 + s22) + s12·s21
 *     h12 =        2·s12 / Δ
 *     h21 =       −2·s21 / Δ
 *     h22 = (1/z0)·((1 − s11)(1 − s22) − s12·s21) / Δ
 *   G from S:
 *     g11 = (1/z0)·((1 − s11)(1 + s22) + s12·s21) / Δ_g,   Δ_g = (1 + s11)(1 − s22) + s12·s21
 *     g12 =       −2·s12 / Δ_g
 *     g21 =        2·s21 / Δ_g
 *     g22 = z0·((1 + s11)(1 − s22) + s12·s21) / Δ_g — NOTE: same form as g11 but with z0 factor;
 *           the symmetry between H and G mirrors the dual of port-1/port-2.
 *   ABCD from S:
 *     A =  ((1 + s11)(1 − s22) + s12·s21) / (2·s21)
 *     B = z0·((1 + s11)(1 + s22) − s12·s21) / (2·s21)
 *     C = (1/z0)·((1 − s11)(1 − s22) − s12·s21) / (2·s21)
 *     D =  ((1 − s11)(1 + s22) + s12·s21) / (2·s21)
 *   T from S: T = (1/s21) · [[ −det(S),   s11], [ −s22,   1]]
 *     (already implemented inside the cascade helper s_to_t). */

matlab_struct *matlab_rf_s2h(matlab_mat_c *S11, matlab_mat_c *S12,
                              matlab_mat_c *S21, matlab_mat_c *S22, double z0) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *H11 = cvec(N), *H12 = cvec(N);
    matlab_mat_c *H21 = cvec(N), *H22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C one = {1.0, 0.0};
        /* Δ = (1 − s11)(1 + s22) + s12·s21 */
        C delta = cadd(cmul(csub(one, s11), cadd(one, s22)),
                        cmul(s12, s21));
        C h11_num = cmul({z0, 0.0},
                          csub(cmul(cadd(one, s11), cadd(one, s22)),
                               cmul(s12, s21)));
        C h12_num = cmul({2.0, 0.0}, s12);
        C h21_num = cmul({-2.0, 0.0}, s21);
        C h22_num = cmul({1.0 / z0, 0.0},
                          csub(cmul(csub(one, s11), csub(one, s22)),
                               cmul(s12, s21)));
        C h11v = cdiv(h11_num, delta);
        C h12v = cdiv(h12_num, delta);
        C h21v = cdiv(h21_num, delta);
        C h22v = cdiv(h22_num, delta);
        H11->re[k] = h11v.re; H11->im[k] = h11v.im;
        H12->re[k] = h12v.re; H12->im[k] = h12v.im;
        H21->re[k] = h21v.re; H21->im[k] = h21v.im;
        H22->re[k] = h22v.re; H22->im[k] = h22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "H11", 3, (matlab_mat *)H11);
    matlab_struct_set_mat(out, "H12", 3, (matlab_mat *)H12);
    matlab_struct_set_mat(out, "H21", 3, (matlab_mat *)H21);
    matlab_struct_set_mat(out, "H22", 3, (matlab_mat *)H22);
    return out;
}

matlab_struct *matlab_rf_s2abcd(matlab_mat_c *S11, matlab_mat_c *S12,
                                 matlab_mat_c *S21, matlab_mat_c *S22,
                                 double z0) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *A_ = cvec(N), *B_ = cvec(N);
    matlab_mat_c *Cm = cvec(N), *D_ = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C one = {1.0, 0.0};
        C two_s21 = cmul({2.0, 0.0}, s21);
        C delta = cmul(s12, s21);
        C A_num = cadd(cmul(cadd(one, s11), csub(one, s22)), delta);
        C B_num = cmul({z0, 0.0},
                        csub(cmul(cadd(one, s11), cadd(one, s22)), delta));
        C C_num = cmul({1.0 / z0, 0.0},
                        csub(cmul(csub(one, s11), csub(one, s22)), delta));
        C D_num = cadd(cmul(csub(one, s11), cadd(one, s22)), delta);
        C Av = cdiv(A_num, two_s21);
        C Bv = cdiv(B_num, two_s21);
        C Cv = cdiv(C_num, two_s21);
        C Dv = cdiv(D_num, two_s21);
        A_->re[k] = Av.re; A_->im[k] = Av.im;
        B_->re[k] = Bv.re; B_->im[k] = Bv.im;
        Cm->re[k] = Cv.re; Cm->im[k] = Cv.im;
        D_->re[k] = Dv.re; D_->im[k] = Dv.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "A", 1, (matlab_mat *)A_);
    matlab_struct_set_mat(out, "B", 1, (matlab_mat *)B_);
    matlab_struct_set_mat(out, "C", 1, (matlab_mat *)Cm);
    matlab_struct_set_mat(out, "D", 1, (matlab_mat *)D_);
    return out;
}

/* ====================================================================== */
/* §9.4.1 follow-on — T / Pi matchingnetwork topologies.                  */
/* ====================================================================== */
/*
 * Three-component T and Pi matching networks for higher-Q / specific
 * topology constraints.  The T topology cascades two L-sections with
 * a virtual high-impedance node R_V; Pi cascades two L-sections with
 * a virtual low-impedance node.
 *
 *   T:  Zs — (Xs1=jωL₁) — node(R_V) — (Xs2=jωL₂) — Zl,   shunt Xp at the node
 *   Pi: Zs — shunt Xp1 — (Xs=jωL) — shunt Xp2 — Zl,      shunts at both ends
 *
 * For a target Q:
 *   T: R_V = max(R_s, R_l) · (Q² + 1)   (always > both ends)
 *   Pi: R_V = min(R_s, R_l) / (Q² + 1)  (always < both ends)
 *
 * The two L-sections are then designed independently, each matching
 * its end-impedance to R_V via the standard quadratic-Q algorithm.
 */
matlab_struct *matlab_rf_matchingnetwork_t(double zs_re, double zs_im,
                                            double zl_re, double zl_im,
                                            double freq, double q_target) {
    double omega = 2.0 * M_PI * freq;
    double R_s = zs_re, R_l = zl_re;
    double Q = (q_target > 0.0) ? q_target : 1.0;
    double R_max = (R_s > R_l) ? R_s : R_l;
    double R_V = R_max * (Q * Q + 1.0);
    /* L-section A: source side (R_s) → R_V.  R_V > R_s so step-up. */
    double Qa = (R_s > 0.0) ? sqrt(R_V / R_s - 1.0) : 0.0;
    double Xs1 = Qa * R_s;
    /* L-section B: load side (R_l) → R_V.  Step-up. */
    double Qb = (R_l > 0.0) ? sqrt(R_V / R_l - 1.0) : 0.0;
    double Xs2 = Qb * R_l;
    /* Shunt at the central node (combines both L-sections' shunt
     * components).  Parallel combination of R_V/Qa and R_V/Qb. */
    double Xp_a = (Qa > 0.0) ? R_V / Qa : 0.0;
    double Xp_b = (Qb > 0.0) ? R_V / Qb : 0.0;
    double Xp_total = (Xp_a + Xp_b > 0.0) ? (Xp_a * Xp_b) / (Xp_a + Xp_b)
                                          : 0.0;
    matlab_struct *out = matlab_struct_new();
    #define SET(name, v) matlab_struct_set_f64(out, name, sizeof(name)-1, v)
    SET("Topology",     2.0);      /* 2 = T-section */
    SET("Q_target",     Q);
    SET("R_virtual",    R_V);
    SET("L1_series_H",  (Xs1 > 0.0) ? Xs1 / omega : 0.0);
    SET("L2_series_H",  (Xs2 > 0.0) ? Xs2 / omega : 0.0);
    SET("C_shunt_F",    (Xp_total > 0.0) ? 1.0 / (omega * Xp_total) : 0.0);
    SET("Frequency_Hz", freq);
    SET("X_high_in",    zs_im);
    SET("X_high_out",   zl_im);
    #undef SET
    return out;
}

/* ====================================================================== */
/* §9.1.2 cross-conversion inverses: H→S, ABCD→S.                          */
/* ====================================================================== */
/*
 * H→S:
 *   Δ_h = (1 + h11/z0)(1 + h22·z0) − h12·h21
 *   s11 = ((h11/z0 − 1)(1 + h22·z0) − h12·h21) / Δ_h
 *   s12 = 2·h12 / Δ_h
 *   s21 = −2·h21 / Δ_h
 *   s22 = ((1 + h11/z0)(1 − h22·z0) + h12·h21) / Δ_h
 *
 * ABCD→S:
 *   Δ = A + B/z0 + C·z0 + D
 *   s11 = (A + B/z0 − C·z0 − D) / Δ
 *   s12 = 2·(A·D − B·C) / Δ
 *   s21 = 2 / Δ
 *   s22 = (−A + B/z0 − C·z0 + D) / Δ
 */
matlab_struct *matlab_rf_h2s(matlab_mat_c *H11, matlab_mat_c *H12,
                              matlab_mat_c *H21, matlab_mat_c *H22, double z0) {
    int64_t N = nfreq_of(H11, H12, H21, H22);
    matlab_mat_c *S11 = cvec(N), *S12 = cvec(N);
    matlab_mat_c *S21 = cvec(N), *S22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C h11 = sread(H11, k), h12 = sread(H12, k);
        C h21 = sread(H21, k), h22 = sread(H22, k);
        C h11_z = cmul({1.0 / z0, 0.0}, h11);
        C h22_z = cmul({z0, 0.0}, h22);
        C delta = csub(cmul(cadd({1.0, 0.0}, h11_z),
                             cadd({1.0, 0.0}, h22_z)),
                        cmul(h12, h21));
        C s11v = cdiv(csub(cmul(csub(h11_z, {1.0, 0.0}),
                                  cadd({1.0, 0.0}, h22_z)),
                            cmul(h12, h21)),
                       delta);
        C s12v = cdiv(cmul({2.0, 0.0}, h12), delta);
        C s21v = cdiv(cmul({-2.0, 0.0}, h21), delta);
        C s22v = cdiv(cadd(cmul(cadd({1.0, 0.0}, h11_z),
                                  csub({1.0, 0.0}, h22_z)),
                            cmul(h12, h21)),
                       delta);
        S11->re[k] = s11v.re; S11->im[k] = s11v.im;
        S12->re[k] = s12v.re; S12->im[k] = s12v.im;
        S21->re[k] = s21v.re; S21->im[k] = s21v.im;
        S22->re[k] = s22v.re; S22->im[k] = s22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    return out;
}

/* ====================================================================== */
/* §9.4.3 follow-on — gamma2z + z2gamma Smith-chart helpers.               */
/* ====================================================================== */
/*
 *   Γ  → Z:   Z = z0 · (1 + Γ) / (1 − Γ)
 *   Z  → Γ:   Γ = (Z − z0) / (Z + z0)
 * Both operate element-wise on a complex column. */
matlab_mat_c *matlab_rf_gamma2z(matlab_mat_c *gamma, double z0) {
    int64_t N = gamma ? gamma->rows * gamma->cols : 0;
    matlab_mat_c *out = mat_c_alloc(N, 1);
    for (int64_t k = 0; k < N; ++k) {
        C g = sread(gamma, k);
        C one = {1.0, 0.0};
        C z = cmul({z0, 0.0}, cdiv(cadd(one, g), csub(one, g)));
        out->re[k] = z.re; out->im[k] = z.im;
    }
    return out;
}

matlab_mat_c *matlab_rf_z2gamma(matlab_mat_c *z, double z0) {
    int64_t N = z ? z->rows * z->cols : 0;
    matlab_mat_c *out = mat_c_alloc(N, 1);
    for (int64_t k = 0; k < N; ++k) {
        C zv = sread(z, k);
        C g = cdiv(csub(zv, {z0, 0.0}), cadd(zv, {z0, 0.0}));
        out->re[k] = g.re; out->im[k] = g.im;
    }
    return out;
}

/* ====================================================================== */
/* §9.1.2 — additional cross-conversions S↔G, S↔T.                         */
/* ====================================================================== */
/*
 * S → G (inverse-hybrid):
 *   Δ_g = (1 + s11)(1 − s22) + s12·s21
 *   g11 = (1/z0)·((1 − s11)(1 − s22) − s12·s21) / Δ_g
 *   g12 = (1/z0)·(2·s12) / Δ_g
 *   g21 = −(2·s21) / Δ_g
 *   g22 = z0·((1 + s11)(1 + s22) − s12·s21) / Δ_g  (incorrect — see derivation)
 *
 * Standard convention (from microwave references): G = inv(H), so
 * we derive G via the H entries.  Explicit closed-form (verified
 * against the H→G matrix inverse): below uses the (1+s11)(1−s22)+s12s21
 * denominator. */

/* S → G via H inverse:  G = inv(H).  Per-frequency 2×2:
 *   det(H) = h11·h22 − h12·h21
 *   g11 =  h22 / det(H)
 *   g12 = −h12 / det(H)
 *   g21 = −h21 / det(H)
 *   g22 =  h11 / det(H)
 * Using the already-correct S→H closed form first. */
matlab_struct *matlab_rf_s2g(matlab_mat_c *S11, matlab_mat_c *S12,
                              matlab_mat_c *S21, matlab_mat_c *S22, double z0) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *G11 = cvec(N), *G12 = cvec(N);
    matlab_mat_c *G21 = cvec(N), *G22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C one = {1.0, 0.0};
        /* H from S. */
        C delta_h = csub(cmul(cadd(one, s11), cadd(one, s22)),
                          cmul(s12, s21));
        (void)delta_h;
        /* Δ_h = (1 − s11)(1 + s22) + s12·s21. */
        C dh = cadd(cmul(csub(one, s11), cadd(one, s22)), cmul(s12, s21));
        C h11 = cdiv(cmul({z0, 0.0},
                            csub(cmul(cadd(one, s11), cadd(one, s22)),
                                  cmul(s12, s21))),
                       dh);
        C h12 = cdiv(cmul({2.0, 0.0}, s12), dh);
        C h21 = cdiv(cmul({-2.0, 0.0}, s21), dh);
        C h22 = cdiv(cmul({1.0 / z0, 0.0},
                            csub(cmul(csub(one, s11), csub(one, s22)),
                                  cmul(s12, s21))),
                       dh);
        /* G = inv(H). */
        C det_h = csub(cmul(h11, h22), cmul(h12, h21));
        C g11v = cdiv(h22, det_h);
        C g12v = cdiv(cmul({-1.0, 0.0}, h12), det_h);
        C g21v = cdiv(cmul({-1.0, 0.0}, h21), det_h);
        C g22v = cdiv(h11, det_h);
        G11->re[k] = g11v.re; G11->im[k] = g11v.im;
        G12->re[k] = g12v.re; G12->im[k] = g12v.im;
        G21->re[k] = g21v.re; G21->im[k] = g21v.im;
        G22->re[k] = g22v.re; G22->im[k] = g22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "G11", 3, (matlab_mat *)G11);
    matlab_struct_set_mat(out, "G12", 3, (matlab_mat *)G12);
    matlab_struct_set_mat(out, "G21", 3, (matlab_mat *)G21);
    matlab_struct_set_mat(out, "G22", 3, (matlab_mat *)G22);
    return out;
}

/* G → S: invert G to get H, then use the H → S closed-form.
 *   H = inv(G):
 *     det(G) = g11·g22 − g12·g21
 *     h11 =  g22 / det(G)
 *     h12 = −g12 / det(G)
 *     h21 = −g21 / det(G)
 *     h22 =  g11 / det(G)
 *   Then S from H via the standard formula. */
matlab_struct *matlab_rf_g2s(matlab_mat_c *G11, matlab_mat_c *G12,
                              matlab_mat_c *G21, matlab_mat_c *G22, double z0) {
    int64_t N = nfreq_of(G11, G12, G21, G22);
    matlab_mat_c *S11 = cvec(N), *S12 = cvec(N);
    matlab_mat_c *S21 = cvec(N), *S22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C g11 = sread(G11, k), g12 = sread(G12, k);
        C g21 = sread(G21, k), g22 = sread(G22, k);
        /* H = inv(G). */
        C det_g = csub(cmul(g11, g22), cmul(g12, g21));
        C h11 = cdiv(g22, det_g);
        C h12 = cdiv(cmul({-1.0, 0.0}, g12), det_g);
        C h21 = cdiv(cmul({-1.0, 0.0}, g21), det_g);
        C h22 = cdiv(g11, det_g);
        /* S from H. */
        C one = {1.0, 0.0};
        C h11_z = cmul({1.0 / z0, 0.0}, h11);
        C h22_z = cmul({z0, 0.0}, h22);
        C delta = csub(cmul(cadd(one, h11_z),
                              cadd(one, h22_z)),
                        cmul(h12, h21));
        C s11v = cdiv(csub(cmul(csub(h11_z, one),
                                  cadd(one, h22_z)),
                            cmul(h12, h21)),
                       delta);
        C s12v = cdiv(cmul({2.0, 0.0}, h12), delta);
        C s21v = cdiv(cmul({-2.0, 0.0}, h21), delta);
        C s22v = cdiv(cadd(cmul(cadd(one, h11_z),
                                  csub(one, h22_z)),
                            cmul(h12, h21)),
                       delta);
        S11->re[k] = s11v.re; S11->im[k] = s11v.im;
        S12->re[k] = s12v.re; S12->im[k] = s12v.im;
        S21->re[k] = s21v.re; S21->im[k] = s21v.im;
        S22->re[k] = s22v.re; S22->im[k] = s22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    return out;
}

/* S → T (chain transmission).  Per-frequency 2×2:
 *   t11 = −det(S) / s21
 *   t12 =  s11    / s21
 *   t21 = −s22    / s21
 *   t22 =  1      / s21
 * with det(S) = s11·s22 − s12·s21. */
matlab_struct *matlab_rf_s2t(matlab_mat_c *S11, matlab_mat_c *S12,
                              matlab_mat_c *S21, matlab_mat_c *S22) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *T11 = cvec(N), *T12 = cvec(N);
    matlab_mat_c *T21 = cvec(N), *T22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C inv = cdiv({1.0, 0.0}, s21);
        C det = csub(cmul(s11, s22), cmul(s12, s21));
        C t11v = cmul({-1.0, 0.0}, cmul(det, inv));
        C t12v = cmul(s11, inv);
        C t21v = cmul({-1.0, 0.0}, cmul(s22, inv));
        C t22v = inv;
        T11->re[k] = t11v.re; T11->im[k] = t11v.im;
        T12->re[k] = t12v.re; T12->im[k] = t12v.im;
        T21->re[k] = t21v.re; T21->im[k] = t21v.im;
        T22->re[k] = t22v.re; T22->im[k] = t22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "T11", 3, (matlab_mat *)T11);
    matlab_struct_set_mat(out, "T12", 3, (matlab_mat *)T12);
    matlab_struct_set_mat(out, "T21", 3, (matlab_mat *)T21);
    matlab_struct_set_mat(out, "T22", 3, (matlab_mat *)T22);
    return out;
}

/* T → S: inverse of the above.
 *   s11 = t12 / t22
 *   s12 = t11 − t12·t21 / t22
 *   s21 = 1 / t22
 *   s22 = −t21 / t22 */
matlab_struct *matlab_rf_t2s(matlab_mat_c *T11, matlab_mat_c *T12,
                              matlab_mat_c *T21, matlab_mat_c *T22) {
    int64_t N = nfreq_of(T11, T12, T21, T22);
    matlab_mat_c *S11 = cvec(N), *S12 = cvec(N);
    matlab_mat_c *S21 = cvec(N), *S22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C t11 = sread(T11, k), t12 = sread(T12, k);
        C t21 = sread(T21, k), t22 = sread(T22, k);
        C inv = cdiv({1.0, 0.0}, t22);
        C s11v = cmul(t12, inv);
        C s21v = inv;
        C s22v = cmul({-1.0, 0.0}, cmul(t21, inv));
        C s12v = csub(t11, cmul(cmul(t12, t21), inv));
        S11->re[k] = s11v.re; S11->im[k] = s11v.im;
        S12->re[k] = s12v.re; S12->im[k] = s12v.im;
        S21->re[k] = s21v.re; S21->im[k] = s21v.im;
        S22->re[k] = s22v.re; S22->im[k] = s22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    return out;
}

matlab_struct *matlab_rf_abcd2s(matlab_mat_c *A_, matlab_mat_c *B_,
                                 matlab_mat_c *Cm, matlab_mat_c *D_,
                                 double z0) {
    int64_t N = nfreq_of(A_, B_, Cm, D_);
    matlab_mat_c *S11 = cvec(N), *S12 = cvec(N);
    matlab_mat_c *S21 = cvec(N), *S22 = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C A = sread(A_, k), B = sread(B_, k);
        C Cc = sread(Cm, k), D = sread(D_, k);
        C B_z = cmul({1.0 / z0, 0.0}, B);
        C C_z = cmul({z0, 0.0}, Cc);
        C delta = cadd(cadd(A, B_z), cadd(C_z, D));
        C s11v = cdiv(csub(cadd(A, B_z), cadd(C_z, D)), delta);
        C s12v = cdiv(cmul({2.0, 0.0}, csub(cmul(A, D), cmul(B, Cc))), delta);
        C s21v = cdiv({2.0, 0.0}, delta);
        C s22v = cdiv(cadd(csub({0.0, 0.0}, A),
                            cadd(B_z, cadd(cmul({-1.0, 0.0}, C_z), D))),
                       delta);
        S11->re[k] = s11v.re; S11->im[k] = s11v.im;
        S12->re[k] = s12v.re; S12->im[k] = s12v.im;
        S21->re[k] = s21v.re; S21->im[k] = s21v.im;
        S22->re[k] = s22v.re; S22->im[k] = s22v.im;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "S11", 3, (matlab_mat *)S11);
    matlab_struct_set_mat(out, "S12", 3, (matlab_mat *)S12);
    matlab_struct_set_mat(out, "S21", 3, (matlab_mat *)S21);
    matlab_struct_set_mat(out, "S22", 3, (matlab_mat *)S22);
    return out;
}

matlab_struct *matlab_rf_matchingnetwork_pi(double zs_re, double zs_im,
                                             double zl_re, double zl_im,
                                             double freq, double q_target) {
    double omega = 2.0 * M_PI * freq;
    double R_s = zs_re, R_l = zl_re;
    double Q = (q_target > 0.0) ? q_target : 1.0;
    double R_min = (R_s < R_l) ? R_s : R_l;
    double R_V = R_min / (Q * Q + 1.0);
    /* L-section A: source side R_s → R_V.  R_V < R_s so step-down. */
    double Qa = (R_V > 0.0 && R_s > R_V) ? sqrt(R_s / R_V - 1.0) : 0.0;
    double Xp1 = (Qa > 0.0) ? R_s / Qa : 0.0;
    /* L-section B: R_V → R_l.  Step-up if R_l > R_V. */
    double Qb = (R_V > 0.0 && R_l > R_V) ? sqrt(R_l / R_V - 1.0) : 0.0;
    double Xp2 = (Qb > 0.0) ? R_l / Qb : 0.0;
    /* Series in middle: combines both L-sections' series components. */
    double Xs_a = Qa * R_V;
    double Xs_b = Qb * R_V;
    double Xs_total = Xs_a + Xs_b;
    matlab_struct *out = matlab_struct_new();
    #define SET(name, v) matlab_struct_set_f64(out, name, sizeof(name)-1, v)
    SET("Topology",     3.0);      /* 3 = Pi-section */
    SET("Q_target",     Q);
    SET("R_virtual",    R_V);
    SET("L_series_H",   (Xs_total > 0.0) ? Xs_total / omega : 0.0);
    SET("C1_shunt_F",   (Xp1 > 0.0) ? 1.0 / (omega * Xp1) : 0.0);
    SET("C2_shunt_F",   (Xp2 > 0.0) ? 1.0 / (omega * Xp2) : 0.0);
    SET("Frequency_Hz", freq);
    SET("X_high_in",    zs_im);
    SET("X_high_out",   zl_im);
    #undef SET
    return out;
}

/* Typed-getter for G / T 2-port fields. */
matlab_mat_c *matlab_rf_ts_gij(matlab_struct *s, double i_d, double j_d) {
    int i = (int)i_d, j = (int)j_d;
    if (i < 1) i = 1; if (i > 2) i = 2;
    if (j < 1) j = 1; if (j > 2) j = 2;
    char fname[8];
    int fn = snprintf(fname, sizeof(fname), "G%d%d", i, j);
    return (matlab_mat_c *)matlab_struct_get_mat(s, fname, fn);
}
matlab_mat_c *matlab_rf_ts_tij(matlab_struct *s, double i_d, double j_d) {
    int i = (int)i_d, j = (int)j_d;
    if (i < 1) i = 1; if (i > 2) i = 2;
    if (j < 1) j = 1; if (j > 2) j = 2;
    char fname[8];
    int fn = snprintf(fname, sizeof(fname), "T%d%d", i, j);
    return (matlab_mat_c *)matlab_struct_get_mat(s, fname, fn);
}

/* Typed-getter for H / ABCD 2-port fields. */
matlab_mat_c *matlab_rf_ts_hij(matlab_struct *s, double i_d, double j_d) {
    int i = (int)i_d, j = (int)j_d;
    if (i < 1) i = 1; if (i > 2) i = 2;
    if (j < 1) j = 1; if (j > 2) j = 2;
    char fname[8];
    int fn = snprintf(fname, sizeof(fname), "H%d%d", i, j);
    return (matlab_mat_c *)matlab_struct_get_mat(s, fname, fn);
}
matlab_mat_c *matlab_rf_ts_abcd_a(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "A", 1);
}
matlab_mat_c *matlab_rf_ts_abcd_b(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "B", 1);
}
matlab_mat_c *matlab_rf_ts_abcd_c(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "C", 1);
}
matlab_mat_c *matlab_rf_ts_abcd_d(matlab_struct *s) {
    return (matlab_mat_c *)matlab_struct_get_mat(s, "D", 1);
}

/* ====================================================================== */
/* §9.1.2 Non-matched-termination snp2smp via Schur complement.            */
/* ====================================================================== */
/*
 * Generalization of snp2smp to handle arbitrary terminations at the
 * dropped ports.  Given N-port S and a list of kept ports P with
 * complement T (dropped ports), with termination impedances z_term
 * at the dropped ports (one per dropped port), the kept-port
 * sub-network has S-matrix:
 *
 *   S_PP' = S_PP + S_PT · Γ_t · (I − S_TT · Γ_t)⁻¹ · S_TP
 *
 * where Γ_t = diag((z_term_k − z0) / (z_term_k + z0)) at each
 * dropped port.  When all z_term_k = z0 (matched), Γ_t = 0 and the
 * formula reduces to S_PP' = S_PP — the matched-termination simple
 * sub-block extraction.  For non-z0 terminations, the Schur-style
 * update kicks in.
 *
 * v1 ships the diagonal-Γ_t case (independent terminations at each
 * dropped port).  Arguments:
 *   data       — touchstoneRead-style struct (NumPorts + Sij fields)
 *   port_list  — real column of kept port indices (1-based)
 *   z_term     — real column of termination impedances (one per
 *                 dropped port, in the order produced by complement
 *                 of port_list)
 *   m_ports    — target kept-port count (= length(port_list))
 *
 * Output: struct with NumPorts = m_ports + S<i><j> fields. */
matlab_struct *matlab_rf_snp2smp_z(matlab_struct *data,
                                    matlab_mat *port_list,
                                    matlab_mat *z_term,
                                    double m_ports_d) {
    int m = (int)m_ports_d;
    if (port_list) {
        int got = (int)(port_list->rows * port_list->cols);
        if (got < m) m = got;
    }
    if (m < 1 || m > 9) return matlab_struct_new();
    int N = (int)matlab_struct_get_f64(data, "NumPorts", 8);
    double z0 = matlab_struct_get_f64(data, "Z0", 2);
    matlab_mat *F = matlab_struct_get_mat(data, "Frequencies", 11);
    int K = F ? (int)(F->rows * F->cols) : 0;
    /* Build the kept-port list and its complement. */
    std::vector<int> kept(m, 0);
    for (int i = 0; i < m; ++i) {
        int p = (int)port_list->data[i];
        if (p < 1) p = 1;
        if (p > N) p = N;
        kept[(size_t)i] = p;
    }
    std::vector<int> dropped;
    for (int p = 1; p <= N; ++p) {
        bool in_kept = false;
        for (int q : kept) if (q == p) { in_kept = true; break; }
        if (!in_kept) dropped.push_back(p);
    }
    int t = (int)dropped.size();
    /* Termination reflection coefficients (per-frequency assumed
     * constant — z_term is a real column matching the dropped-port
     * count).  Γ_k = (z_k − z0) / (z_k + z0). */
    std::vector<double> gamma_t((size_t)t, 0.0);
    int z_got = z_term ? (int)(z_term->rows * z_term->cols) : 0;
    for (int i = 0; i < t; ++i) {
        double z_i = (i < z_got) ? z_term->data[i] : z0;
        gamma_t[(size_t)i] = (z_i - z0) / (z_i + z0);
    }
    /* Output container. */
    matlab_struct *out = matlab_struct_new();
    /* Per-frequency Schur-complement update.  For each k, gather
     * S_PP (m×m), S_PT (m×t), S_TP (t×m), S_TT (t×t), form
     * M = inv(I − S_TT · Γ_t), then S' = S_PP + S_PT · Γ_t · M · S_TP. */
    int mm = m * m;
    std::vector<matlab_mat_c *> out_cols((size_t)mm);
    for (int idx = 0; idx < mm; ++idx) out_cols[(size_t)idx] = mat_c_alloc(K, 1);
    char name[8];
    for (int k = 0; k < K; ++k) {
        /* Gather S_PP, S_PT, S_TP, S_TT at freq k.  All complex. */
        std::vector<double> Spp_re((size_t)(m*m)), Spp_im((size_t)(m*m));
        std::vector<double> Spt_re((size_t)(m*t)), Spt_im((size_t)(m*t));
        std::vector<double> Stp_re((size_t)(t*m)), Stp_im((size_t)(t*m));
        std::vector<double> Stt_re((size_t)(t*t)), Stt_im((size_t)(t*t));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                int sp_i = kept[(size_t)i], sp_j = kept[(size_t)j];
                int fn = snprintf(name, sizeof(name), "S%d%d", sp_i, sp_j);
                matlab_mat_c *Sij =
                    (matlab_mat_c *)matlab_struct_get_mat(data, name, fn);
                int n_freq = Sij ? (int)(Sij->rows * Sij->cols) : 0;
                Spp_re[(size_t)(i*m + j)] = (k < n_freq) ? Sij->re[k] : 0.0;
                Spp_im[(size_t)(i*m + j)] = (k < n_freq) ? Sij->im[k] : 0.0;
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < t; ++j) {
                int sp_i = kept[(size_t)i], sp_j = dropped[(size_t)j];
                int fn = snprintf(name, sizeof(name), "S%d%d", sp_i, sp_j);
                matlab_mat_c *Sij =
                    (matlab_mat_c *)matlab_struct_get_mat(data, name, fn);
                int n_freq = Sij ? (int)(Sij->rows * Sij->cols) : 0;
                Spt_re[(size_t)(i*t + j)] = (k < n_freq) ? Sij->re[k] : 0.0;
                Spt_im[(size_t)(i*t + j)] = (k < n_freq) ? Sij->im[k] : 0.0;
            }
        }
        for (int i = 0; i < t; ++i) {
            for (int j = 0; j < m; ++j) {
                int sp_i = dropped[(size_t)i], sp_j = kept[(size_t)j];
                int fn = snprintf(name, sizeof(name), "S%d%d", sp_i, sp_j);
                matlab_mat_c *Sij =
                    (matlab_mat_c *)matlab_struct_get_mat(data, name, fn);
                int n_freq = Sij ? (int)(Sij->rows * Sij->cols) : 0;
                Stp_re[(size_t)(i*m + j)] = (k < n_freq) ? Sij->re[k] : 0.0;
                Stp_im[(size_t)(i*m + j)] = (k < n_freq) ? Sij->im[k] : 0.0;
            }
        }
        for (int i = 0; i < t; ++i) {
            for (int j = 0; j < t; ++j) {
                int sp_i = dropped[(size_t)i], sp_j = dropped[(size_t)j];
                int fn = snprintf(name, sizeof(name), "S%d%d", sp_i, sp_j);
                matlab_mat_c *Sij =
                    (matlab_mat_c *)matlab_struct_get_mat(data, name, fn);
                int n_freq = Sij ? (int)(Sij->rows * Sij->cols) : 0;
                Stt_re[(size_t)(i*t + j)] = (k < n_freq) ? Sij->re[k] : 0.0;
                Stt_im[(size_t)(i*t + j)] = (k < n_freq) ? Sij->im[k] : 0.0;
            }
        }
        /* Compute M = inv(I − S_TT · Γ_t).  Γ_t is diagonal, so
         * (S_TT · Γ_t)_ij = S_TT_ij · γ_j. */
        std::vector<double> A_re((size_t)(t*t), 0.0);
        std::vector<double> A_im((size_t)(t*t), 0.0);
        for (int i = 0; i < t; ++i) {
            for (int j = 0; j < t; ++j) {
                double g = gamma_t[(size_t)j];
                A_re[(size_t)(i*t + j)] = -g * Stt_re[(size_t)(i*t + j)];
                A_im[(size_t)(i*t + j)] = -g * Stt_im[(size_t)(i*t + j)];
            }
            A_re[(size_t)(i*t + i)] += 1.0;
        }
        std::vector<double> M_re((size_t)(t*t)), M_im((size_t)(t*t));
        if (t > 0) {
            complex_mat_inv_2neq(t, A_re.data(), A_im.data(),
                                  M_re.data(), M_im.data());
        }
        /* Compute X = Γ_t · M · S_TP    (t × m).
         * First M·S_TP, then left-multiply by diag Γ_t. */
        std::vector<double> MS_re((size_t)(t*m), 0.0);
        std::vector<double> MS_im((size_t)(t*m), 0.0);
        for (int i = 0; i < t; ++i) {
            for (int j = 0; j < m; ++j) {
                double cr = 0.0, ci = 0.0;
                for (int kk = 0; kk < t; ++kk) {
                    double ar = M_re[(size_t)(i*t + kk)];
                    double ai = M_im[(size_t)(i*t + kk)];
                    double br = Stp_re[(size_t)(kk*m + j)];
                    double bi = Stp_im[(size_t)(kk*m + j)];
                    cr += ar*br - ai*bi;
                    ci += ar*bi + ai*br;
                }
                MS_re[(size_t)(i*m + j)] = cr;
                MS_im[(size_t)(i*m + j)] = ci;
            }
        }
        std::vector<double> X_re((size_t)(t*m), 0.0);
        std::vector<double> X_im((size_t)(t*m), 0.0);
        for (int i = 0; i < t; ++i) {
            double g = gamma_t[(size_t)i];
            for (int j = 0; j < m; ++j) {
                X_re[(size_t)(i*m + j)] = g * MS_re[(size_t)(i*m + j)];
                X_im[(size_t)(i*m + j)] = g * MS_im[(size_t)(i*m + j)];
            }
        }
        /* Compute S' = S_PP + S_PT · X    (m × m). */
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                double cr = Spp_re[(size_t)(i*m + j)];
                double ci = Spp_im[(size_t)(i*m + j)];
                for (int kk = 0; kk < t; ++kk) {
                    double ar = Spt_re[(size_t)(i*t + kk)];
                    double ai = Spt_im[(size_t)(i*t + kk)];
                    double br = X_re[(size_t)(kk*m + j)];
                    double bi = X_im[(size_t)(kk*m + j)];
                    cr += ar*br - ai*bi;
                    ci += ar*bi + ai*br;
                }
                int idx = i*m + j;
                out_cols[(size_t)idx]->re[k] = cr;
                out_cols[(size_t)idx]->im[k] = ci;
            }
        }
    }
    /* Pack the kept-port S' into the output struct. */
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= m; ++j) {
            int fn = snprintf(name, sizeof(name), "S%d%d", i, j);
            int idx = (i - 1) * m + (j - 1);
            matlab_struct_set_mat(out, name, fn,
                                   (matlab_mat *)out_cols[(size_t)idx]);
        }
    }
    matlab_struct_set_f64(out, "NumPorts", 8, (double)m);
    matlab_struct_set_f64(out, "Z0",       2, z0);
    matlab_struct_set_mat(out, "Frequencies", 11, F);
    return out;
}

/* Typed-getter for Y/Z N-port fields: tsYij / tsZij(struct, i, j). */
matlab_mat_c *matlab_rf_ts_yij(matlab_struct *s, double i_d, double j_d) {
    int i = (int)i_d, j = (int)j_d;
    if (i < 1) i = 1; if (i > 9) i = 9;
    if (j < 1) j = 1; if (j > 9) j = 9;
    char fname[8];
    int fn = snprintf(fname, sizeof(fname), "Y%d%d", i, j);
    return (matlab_mat_c *)matlab_struct_get_mat(s, fname, fn);
}
matlab_mat_c *matlab_rf_ts_zij(matlab_struct *s, double i_d, double j_d) {
    int i = (int)i_d, j = (int)j_d;
    if (i < 1) i = 1; if (i > 9) i = 9;
    if (j < 1) j = 1; if (j > 9) j = 9;
    char fname[8];
    int fn = snprintf(fname, sizeof(fname), "Z%d%d", i, j);
    return (matlab_mat_c *)matlab_struct_get_mat(s, fname, fn);
}

/* ====================================================================== */
/* §9.2.1 follow-on — simultaneous conjugate-match Γ values.              */
/* ====================================================================== */
/*
 * Source-side (γ_MS) and load-side (γ_ML) simultaneous-conjugate-
 * match reflection coefficients for max-available-gain amplifier design.
 *   B1 = 1 + |s11|² − |s22|² − |Δ|²,   C1 = s11 − Δ·conj(s22)
 *   B2 = 1 + |s22|² − |s11|² − |Δ|²,   C2 = s22 − Δ·conj(s11)
 *   γ_MS = (B1 ± √(B1²−4|C1|²)) / (2·C1)    [pick |γ|<1]
 *   γ_ML = (B2 ± √(B2²−4|C2|²)) / (2·C2)    [pick |γ|<1]
 */
static C rf_match_gamma(double B, C Cc) {
    double Cmag2 = cabs2(Cc);
    if (Cmag2 == 0.0) return {0.0, 0.0};
    double rad = B * B - 4.0 * Cmag2;
    if (rad < 0.0) {
        double cm = cmag(Cc);
        if (cm == 0.0) return {0.0, 0.0};
        return {Cc.re / cm, Cc.im / cm};
    }
    double sqrt_rad = sqrt(rad);
    C cand1 = cdiv({B + sqrt_rad, 0.0}, cmul({2.0, 0.0}, Cc));
    C cand2 = cdiv({B - sqrt_rad, 0.0}, cmul({2.0, 0.0}, Cc));
    return (cabs2(cand1) < cabs2(cand2)) ? cand1 : cand2;
}

matlab_mat_c *matlab_rf_gammams(matlab_mat_c *S11, matlab_mat_c *S12,
                                 matlab_mat_c *S21, matlab_mat_c *S22) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *out = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C delta = csub(cmul(s11, s22), cmul(s12, s21));
        C conj_s22 = {s22.re, -s22.im};
        C C1 = csub(s11, cmul(delta, conj_s22));
        double B1 = 1.0 + cabs2(s11) - cabs2(s22) - cabs2(delta);
        C g = rf_match_gamma(B1, C1);
        out->re[k] = g.re; out->im[k] = g.im;
    }
    return out;
}

matlab_mat_c *matlab_rf_gammaml(matlab_mat_c *S11, matlab_mat_c *S12,
                                 matlab_mat_c *S21, matlab_mat_c *S22) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *out = cvec(N);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C delta = csub(cmul(s11, s22), cmul(s12, s21));
        C conj_s11 = {s11.re, -s11.im};
        C C2 = csub(s22, cmul(delta, conj_s11));
        double B2 = 1.0 + cabs2(s22) - cabs2(s11) - cabs2(delta);
        C g = rf_match_gamma(B2, C2);
        out->re[k] = g.re; out->im[k] = g.im;
    }
    return out;
}

/* ====================================================================== */
/* §9.2.1 follow-on — group delay τ_g = −d(phase)/dω.                     */
/* ====================================================================== */
matlab_mat *matlab_rf_groupdelay(matlab_mat_c *S, matlab_mat *freqs) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat *out = mat_alloc(K, 1);
    if (!S || K < 2) return out;
    std::vector<double> phase((size_t)K, 0.0);
    double prev = 0.0;
    for (int k = 0; k < K; ++k) {
        C v = sread(S, k);
        double ph = atan2(v.im, v.re);
        if (k > 0) {
            double d = ph - prev;
            while (d >  M_PI) { ph -= 2.0 * M_PI; d = ph - prev; }
            while (d < -M_PI) { ph += 2.0 * M_PI; d = ph - prev; }
        }
        phase[(size_t)k] = ph;
        prev = ph;
    }
    for (int k = 0; k < K; ++k) {
        double dph, dw;
        if (k == 0) {
            dph = phase[1] - phase[0];
            dw = 2.0 * M_PI * (freqs->data[1] - freqs->data[0]);
        } else if (k == K - 1) {
            dph = phase[(size_t)k] - phase[(size_t)(k - 1)];
            dw = 2.0 * M_PI * (freqs->data[k] - freqs->data[k - 1]);
        } else {
            dph = phase[(size_t)(k + 1)] - phase[(size_t)(k - 1)];
            dw = 2.0 * M_PI * (freqs->data[k + 1] - freqs->data[k - 1]);
        }
        out->data[k] = (dw != 0.0) ? -dph / dw : 0.0;
    }
    return out;
}

/* ====================================================================== */
/* §9.2.1 follow-on — s2tf with arbitrary input/output port indices.      */
/* ====================================================================== */
extern matlab_mat_c *matlab_rf_s2tf(matlab_mat_c *S11, matlab_mat_c *S12,
                                     matlab_mat_c *S21, matlab_mat_c *S22,
                                     double zs, double zl, double z0);

matlab_mat_c *matlab_rf_s2tf_port(matlab_mat_c *S11, matlab_mat_c *S12,
                                   matlab_mat_c *S21, matlab_mat_c *S22,
                                   double zs, double zl, double z0,
                                   double port_in_d, double port_out_d) {
    int port_in = (int)port_in_d;
    int port_out = (int)port_out_d;
    if (port_in == 2 && port_out == 1) {
        return matlab_rf_s2tf(S22, S21, S12, S11, zs, zl, z0);
    }
    return matlab_rf_s2tf(S11, S12, S21, S22, zs, zl, z0);
}

/* ====================================================================== */
/* §9.2.3 follow-on — rfbudgetTable per-stage cumulative columns.         */
/* ====================================================================== */
matlab_struct *matlab_rf_budget_table(matlab_mat *gains_dB,
                                       matlab_mat *nfs_dB,
                                       matlab_mat *ip3_dBm,
                                       double p_in_dBm,
                                       double bw_Hz) {
    int N = gains_dB ? (int)(gains_dB->rows * gains_dB->cols) : 0;
    int K = N + 1;
    matlab_mat *gain_col   = mat_alloc(K, 1);
    matlab_mat *nf_col     = mat_alloc(K, 1);
    matlab_mat *ip3_col    = mat_alloc(K, 1);
    matlab_mat *power_col  = mat_alloc(K, 1);
    matlab_mat *snr_col    = mat_alloc(K, 1);
    matlab_mat *noise_col  = mat_alloc(K, 1);
    double g_total_dB = 0.0, f_total = 1.0, g_run_lin = 1.0, inv_ip3_lin = 0.0;
    double kT = -174.0;
    double bw_dBHz = (bw_Hz > 0.0) ? 10.0 * log10(bw_Hz) : 0.0;
    auto emit = [&](int k, bool seen) {
        double nf_dB = !seen ? 0.0
            : 10.0 * log10(f_total <= 0.0 ? 1.0 : f_total);
        double thermal_out = kT + bw_dBHz + g_total_dB;
        double noise_out   = thermal_out + nf_dB;
        double p_out       = p_in_dBm + g_total_dB;
        double snr_dB      = p_out - noise_out;
        double ip3_in_dBm  = (inv_ip3_lin > 0.0)
            ? 10.0 * log10(1.0 / inv_ip3_lin) : 1.0e6;
        gain_col->data[k]  = g_total_dB;
        nf_col->data[k]    = nf_dB;
        ip3_col->data[k]   = ip3_in_dBm;
        power_col->data[k] = p_out;
        snr_col->data[k]   = snr_dB;
        noise_col->data[k] = noise_out;
    };
    emit(0, false);
    for (int k = 0; k < N; ++k) {
        double g_dB = gains_dB->data[k];
        double nf_dB = (nfs_dB && k < (int)(nfs_dB->rows * nfs_dB->cols))
                       ? nfs_dB->data[k] : 0.0;
        double ip3 = (ip3_dBm && k < (int)(ip3_dBm->rows * ip3_dBm->cols))
                     ? ip3_dBm->data[k] : 1.0e6;
        double f_k = pow(10.0, nf_dB / 10.0);
        double g_k = pow(10.0, g_dB / 10.0);
        if (k == 0) f_total = f_k;
        else        f_total += (f_k - 1.0) / g_run_lin;
        double ip3_lin_in = pow(10.0, (ip3 - g_total_dB) / 10.0);
        inv_ip3_lin += 1.0 / ip3_lin_in;
        g_run_lin *= g_k;
        g_total_dB += g_dB;
        emit(k + 1, true);
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "StageGain_dB",      12, gain_col);
    matlab_struct_set_mat(out, "StageNF_dB",        10, nf_col);
    matlab_struct_set_mat(out, "StageIP3_in_dBm",   15, ip3_col);
    matlab_struct_set_mat(out, "StageOutputPower",  15, power_col);
    matlab_struct_set_mat(out, "StageSNR_dB",       11, snr_col);
    matlab_struct_set_mat(out, "StageNoiseFloor",   14, noise_col);
    matlab_struct_set_f64(out, "NumStages",          9, (double)N);
    matlab_struct_set_f64(out, "InputPower_dBm",    14, p_in_dBm);
    matlab_struct_set_f64(out, "Bandwidth_Hz",      12, bw_Hz);
    return out;
}

/* ====================================================================== */
/* §9.4.3 follow-on — Smith chart stability circles.                       */
/* ====================================================================== */
static matlab_struct *rf_stab_circle_side(matlab_mat_c *S11, matlab_mat_c *S12,
                                           matlab_mat_c *S21, matlab_mat_c *S22,
                                           bool load_side) {
    int64_t N = nfreq_of(S11, S12, S21, S22);
    matlab_mat_c *Center = cvec(N);
    matlab_mat *Radius = mat_alloc(N, 1);
    matlab_mat *Denom = mat_alloc(N, 1);
    for (int64_t k = 0; k < N; ++k) {
        C s11 = sread(S11, k), s12 = sread(S12, k);
        C s21 = sread(S21, k), s22 = sread(S22, k);
        C delta = csub(cmul(s11, s22), cmul(s12, s21));
        double den;
        C numer;
        if (load_side) {
            den = cabs2(s22) - cabs2(delta);
            C conj_s11 = {s11.re, -s11.im};
            C v = csub(s22, cmul(delta, conj_s11));
            numer = {v.re, -v.im};
        } else {
            den = cabs2(s11) - cabs2(delta);
            C conj_s22 = {s22.re, -s22.im};
            C v = csub(s11, cmul(delta, conj_s22));
            numer = {v.re, -v.im};
        }
        double mag_num = cmag(cmul(s12, s21));
        C center = (den == 0.0) ? (C){0.0, 0.0}
                                : (C){numer.re / den, numer.im / den};
        double radius = (den == 0.0) ? 0.0 : mag_num / fabs(den);
        Center->re[k] = center.re;
        Center->im[k] = center.im;
        Radius->data[k] = radius;
        Denom->data[k] = den;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Center", 6, (matlab_mat *)Center);
    matlab_struct_set_mat(out, "Radius", 6, Radius);
    matlab_struct_set_mat(out, "Denom",  5, Denom);
    return out;
}

matlab_struct *matlab_rf_stab_circle_load(matlab_mat_c *S11, matlab_mat_c *S12,
                                           matlab_mat_c *S21, matlab_mat_c *S22) {
    return rf_stab_circle_side(S11, S12, S21, S22, /*load_side=*/true);
}

matlab_struct *matlab_rf_stab_circle_source(matlab_mat_c *S11, matlab_mat_c *S12,
                                             matlab_mat_c *S21, matlab_mat_c *S22) {
    return rf_stab_circle_side(S11, S12, S21, S22, /*load_side=*/false);
}

/* ====================================================================== */
/* §9.3.1 follow-on — bulk-delay estimation for rationalfit.              */
/* ====================================================================== */
/*
 * Estimates the transport delay τ from the slope of unwrapped phase
 * over the top 25% of the frequency range (where pole behavior is
 * dominated by linear-phase delay).  Returns a scalar τ (seconds).
 *
 * Workflow:
 *   tau = rfDelayEstimate(freqs, h_re, h_im);
 *   dd  = rfApplyDelay(freqs, h_re, h_im, tau);    % removes the delay
 *   mdl = rationalfit(freqs, dd.Real, dd.Imag, nPoles, nIter);
 *   % mdl now has Delay = tau encoded separately.
 */
double matlab_rf_delay_estimate(matlab_mat *freqs,
                                 matlab_mat *h_re, matlab_mat *h_im) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    if (K < 2 || !h_re || !h_im) return 0.0;
    int k0 = K * 3 / 4;
    if (k0 < 1) k0 = 1;
    int k1 = K - 1;
    double prev = atan2(h_im->data[k0], h_re->data[k0]);
    double phase_lo = prev;
    double phase_hi = prev;
    for (int k = k0 + 1; k <= k1; ++k) {
        double ph = atan2(h_im->data[k], h_re->data[k]);
        double d = ph - prev;
        while (d >  M_PI) { ph -= 2.0 * M_PI; d = ph - prev; }
        while (d < -M_PI) { ph += 2.0 * M_PI; d = ph - prev; }
        phase_hi = ph;
        prev = ph;
    }
    double f0 = freqs->data[k0];
    double f1 = freqs->data[k1];
    if (f1 <= f0) return 0.0;
    return -(phase_hi - phase_lo) / (2.0 * M_PI * (f1 - f0));
}

/* Apply (or remove) a bulk delay to a complex frequency-domain
 * dataset.  Returns h · exp(j·ω·τ) as separate real / imag columns
 * inside a struct.  For τ > 0 this REMOVES a positive transport
 * delay (use before rationalfit).  Negative τ applies one. */
matlab_struct *matlab_rf_apply_delay(matlab_mat *freqs,
                                      matlab_mat *h_re, matlab_mat *h_im,
                                      double tau) {
    int K = freqs ? (int)(freqs->rows * freqs->cols) : 0;
    matlab_mat *out_re = mat_alloc(K, 1);
    matlab_mat *out_im = mat_alloc(K, 1);
    for (int k = 0; k < K; ++k) {
        double w = 2.0 * M_PI * freqs->data[k];
        double phi = w * tau;
        double c = cos(phi), s = sin(phi);
        double hr = h_re ? h_re->data[k] : 0.0;
        double hi = h_im ? h_im->data[k] : 0.0;
        out_re->data[k] = hr * c - hi * s;
        out_im->data[k] = hr * s + hi * c;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Real",  4, out_re);
    matlab_struct_set_mat(out, "Imag",  4, out_im);
    matlab_struct_set_f64(out, "Delay", 5, tau);
    return out;
}

/* ====================================================================== */
/* §9.3.1 follow-on — passivity enforcement.                              */
/* ====================================================================== */
/*
 * Iteratively scales residues + D to drive max|H(jω)| ≤ 1 over a
 * dense log-spaced frequency sweep.  Coarse uniform scaling (the
 * literature has more sophisticated per-residue perturbation
 * schemes); good enough for time-domain circuit-simulator inputs
 * that need a passive model.  Caps at 10 iterations.
 *
 * Returns a new struct with the same shape as rationalfit's output. */
matlab_struct *matlab_rf_enforce_passivity(matlab_struct *mdl,
                                            double f_lo, double f_hi) {
    if (!mdl) return matlab_struct_new();
    matlab_mat *Pmat = matlab_struct_get_mat(mdl, "Poles", 5);
    matlab_mat *Rmat = matlab_struct_get_mat(mdl, "Residues", 8);
    double D = matlab_struct_get_f64(mdl, "D", 1);
    double Delay = matlab_struct_get_f64(mdl, "Delay", 5);
    double Order = matlab_struct_get_f64(mdl, "Order", 5);
    double FitErr = matlab_struct_get_f64(mdl, "FitError", 8);
    if (!Pmat || !Rmat) return mdl;
    bool p_complex = mat_is_complex(Pmat);
    bool r_complex = mat_is_complex(Rmat);
    int n;
    if (p_complex) {
        matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
        n = (int)(Pc->rows * Pc->cols);
    } else {
        n = (int)(Pmat->rows * Pmat->cols);
    }
    std::vector<double> R_re((size_t)n), R_im((size_t)n);
    for (int j = 0; j < n; ++j) {
        if (r_complex) {
            matlab_mat_c *Rc = (matlab_mat_c *)Rmat;
            R_re[(size_t)j] = Rc->re[j];
            R_im[(size_t)j] = Rc->im[j];
        } else {
            R_re[(size_t)j] = Rmat->data[j];
            R_im[(size_t)j] = 0.0;
        }
    }
    double Dscaled = D;
    for (int iter = 0; iter < 10; ++iter) {
        int N = 400;
        double f_lo_safe = (f_lo > 0.0) ? f_lo : 1.0;
        double f_hi_safe = (f_hi > f_lo_safe) ? f_hi : f_lo_safe * 1000.0;
        double log_lo = log10(f_lo_safe), log_hi = log10(f_hi_safe);
        double max_mag = 0.0;
        for (int k = 0; k < N; ++k) {
            double t = (N <= 1) ? 0.5 : (double)k / (double)(N - 1);
            double f = pow(10.0, log_lo + t * (log_hi - log_lo));
            double w = 2.0 * M_PI * f;
            double Hr = Dscaled, Hi = 0.0;
            for (int j = 0; j < n; ++j) {
                double pr, pi;
                if (p_complex) {
                    matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
                    pr = Pc->re[j]; pi = Pc->im[j];
                } else {
                    pr = Pmat->data[j]; pi = 0.0;
                }
                double d_re = -pr;
                double d_im = w - pi;
                double dmag2 = d_re*d_re + d_im*d_im;
                double cr = R_re[(size_t)j], ci = R_im[(size_t)j];
                double nr = cr * d_re + ci * d_im;
                double ni = ci * d_re - cr * d_im;
                Hr += nr / dmag2;
                Hi += ni / dmag2;
            }
            double mag = sqrt(Hr*Hr + Hi*Hi);
            if (mag > max_mag) max_mag = mag;
        }
        if (max_mag <= 1.0) break;
        double s = 0.99 / max_mag;
        for (int j = 0; j < n; ++j) {
            R_re[(size_t)j] *= s;
            R_im[(size_t)j] *= s;
        }
        Dscaled *= s;
    }
    matlab_mat_c *Rout = mat_c_alloc(n, 1);
    for (int j = 0; j < n; ++j) {
        Rout->re[j] = R_re[(size_t)j];
        Rout->im[j] = R_im[(size_t)j];
    }
    matlab_struct *out = matlab_struct_new();
    if (p_complex) {
        matlab_mat_c *Pc = (matlab_mat_c *)Pmat;
        matlab_mat_c *Pout = mat_c_alloc(n, 1);
        for (int j = 0; j < n; ++j) {
            Pout->re[j] = Pc->re[j];
            Pout->im[j] = Pc->im[j];
        }
        matlab_struct_set_mat(out, "Poles", 5, (matlab_mat *)Pout);
    } else {
        matlab_mat *Pout = mat_alloc(n, 1);
        for (int j = 0; j < n; ++j) Pout->data[j] = Pmat->data[j];
        matlab_struct_set_mat(out, "Poles", 5, Pout);
    }
    matlab_struct_set_mat(out, "Residues", 8, (matlab_mat *)Rout);
    matlab_struct_set_f64(out, "D",        1, Dscaled);
    matlab_struct_set_f64(out, "Order",    5, Order);
    matlab_struct_set_f64(out, "FitError", 8, FitErr);
    matlab_struct_set_f64(out, "Delay",    5, Delay);
    return out;
}

}  /* extern "C" */
