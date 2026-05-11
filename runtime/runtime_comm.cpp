/* runtime_comm.cpp — Communications Toolbox Tier-1 base layer.
 *
 * Function-form prerequisites that gate every higher Comm tier:
 *   §2.1 randi (uniform integer source)
 *   §2.2 rng   (seed control: set / default / shuffle / save / restore)
 *   §2.3 randsrc / randerr (alphabet sampling + binary error vectors)
 *   §2.4 int2bit / bit2int / de2bi / bi2de (bit/int conversion)
 *   §2.5 awgn (white Gaussian noise channel)
 *   §2.6 biterr / symerr (BER / SER statistics)
 *
 * String selectors are deliberately avoided — every dispatch in the
 * tensor-lowering table routes f64 or ptr operands, and we stay
 * inside that contract by exposing numeric helpers (rngDefault /
 * rngShuffle) plus the canonical scalar / matrix forms.
 *
 * Shared global PRNG state with matlab_rand / matlab_randn lives in
 * matlab_runtime.cpp (`matlab_rng_state`, declared extern below).
 * Reusing the same state means `rng(seed); rand(...)` is deterministic
 * end-to-end across the existing rand/randn path AND the new comm
 * primitives.
 */

#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Shared global PRNG state declared in matlab_runtime.cpp. */
extern "C" uint64_t matlab_rng_state;

/* Local copies of the static rng helpers — needed because the
 * originals are TU-private in matlab_runtime.cpp. Behaviour is
 * byte-identical because the global state is shared. */
static inline double comm_uniform(void) {
    uint64_t x = matlab_rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    matlab_rng_state = x;
    return (double)(x >> 11) / (double)(1ULL << 53);
}

static inline double comm_normal(void) {
    double u1 = comm_uniform();
    double u2 = comm_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

extern "C" {

/* ===== §2.2 rng — seed control ============================================
 *
 * MATLAB's rng() has four canonical forms:
 *   rng(seed)        — deterministic integer seed; current generator preserved.
 *   rng('default')   — seed = 0 with the Mersenne Twister default state.
 *   rng('shuffle')   — seed = current wall-clock-derived randomness.
 *   s = rng()        — save current state to a struct.
 *   rng(s)           — restore state from a saved struct.
 *
 * Since string selectors are off the table for the function-dispatch
 * lane, ship the three string variants as named functions
 * (rngDefault / rngShuffle) and represent the save / restore state
 * as a scalar f64 (the xorshift seed is a 64-bit integer; we round-
 * trip it through a double which loses the bottom 11 bits but is
 * adequate for save-then-restore-within-session use). */

void matlab_comm_rng(double seed) {
    /* MATLAB convention: rng(0) maps to the "default" state. The
     * xorshift kernel collapses to a fixed point at state == 0, so
     * substitute a non-zero seed identical to the original
     * boot-time constant. Mirrors the deterministic-default
     * behaviour of MATLAB's default Mersenne Twister. */
    uint64_t s = (uint64_t)seed;
    if (s == 0) s = 0x243f6a8885a308d3ULL;
    matlab_rng_state = s;
}

void matlab_comm_rng_default(void) {
    matlab_rng_state = 0x243f6a8885a308d3ULL;
}

void matlab_comm_rng_shuffle(void) {
    /* Seed from wall-clock + process-relative ticks. Not
     * cryptographic; matches MATLAB's "non-reproducible across
     * runs" semantics. */
    uint64_t s = (uint64_t)time(NULL);
    s ^= (uint64_t)clock();
    s = s * 0x9E3779B97F4A7C15ULL + 0x123456789ABCDEF0ULL;
    if (s == 0) s = 1;
    matlab_rng_state = s;
}

double matlab_comm_rng_get(void) {
    /* Save the current state as a scalar. The double loses the
     * bottom 11 bits of the 64-bit state, but the high 53 bits are
     * enough to reseed the xorshift kernel — its mixing
     * propagates lost entropy back in within a couple of advances. */
    return (double)matlab_rng_state;
}

void matlab_comm_rng_set(double state) {
    uint64_t s = (uint64_t)state;
    if (s == 0) s = 0x243f6a8885a308d3ULL;
    matlab_rng_state = s;
}

/* ===== §2.1 randi — uniform integer source =============================== */

/* randi(imax)            -> scalar int in [1, imax]. */
double matlab_comm_randi_s(double imax) {
    int64_t hi = (int64_t)imax;
    if (hi < 1) hi = 1;
    return floor(comm_uniform() * (double)hi) + 1.0;
}

/* randi(imax, n)         -> n x n int matrix. */
matlab_mat *matlab_comm_randi_nn(double imax, double n) {
    int64_t sz = (int64_t)n;
    if (sz < 0) sz = 0;
    int64_t hi = (int64_t)imax;
    if (hi < 1) hi = 1;
    matlab_mat *A = mat_alloc(sz, sz);
    for (int64_t k = 0; k < sz * sz; ++k)
        A->data[k] = floor(comm_uniform() * (double)hi) + 1.0;
    return A;
}

/* randi(imax, m, n)      -> m x n int matrix in [1, imax]. */
matlab_mat *matlab_comm_randi_mn(double imax, double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    if (rm < 0) rm = 0; if (cn < 0) cn = 0;
    int64_t hi = (int64_t)imax;
    if (hi < 1) hi = 1;
    matlab_mat *A = mat_alloc(rm, cn);
    for (int64_t k = 0; k < rm * cn; ++k)
        A->data[k] = floor(comm_uniform() * (double)hi) + 1.0;
    return A;
}

/* randi([imin, imax], m, n) -> m x n int matrix in [imin, imax]. The
 * bracketed range is a 1x2 row vector at the MATLAB level; we expose
 * it as a separate runtime entry taking the two scalars explicitly
 * (callers can either spell the range out or wrap the bracketed form
 * with a tiny dispatcher in the front-end shim). */
matlab_mat *matlab_comm_randi_range(double imin, double imax,
                                     double m, double n) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    if (rm < 0) rm = 0; if (cn < 0) cn = 0;
    int64_t lo = (int64_t)imin;
    int64_t hi = (int64_t)imax;
    if (hi < lo) { int64_t t = lo; lo = hi; hi = t; }
    int64_t span = hi - lo + 1;
    if (span < 1) span = 1;
    matlab_mat *A = mat_alloc(rm, cn);
    for (int64_t k = 0; k < rm * cn; ++k)
        A->data[k] = (double)(lo + (int64_t)floor(comm_uniform() * (double)span));
    return A;
}

/* ===== §2.3 randsrc / randerr ============================================ */

/* randsrc(m, n, alphabet) where `alphabet` is a column vector of
 * values to sample uniformly. */
matlab_mat *matlab_comm_randsrc(double m, double n, matlab_mat *alphabet) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    if (rm < 0) rm = 0; if (cn < 0) cn = 0;
    matlab_mat *A = mat_alloc(rm, cn);
    if (!alphabet || alphabet->rows * alphabet->cols == 0) return A;
    int64_t na = alphabet->rows * alphabet->cols;
    for (int64_t k = 0; k < rm * cn; ++k) {
        int64_t idx = (int64_t)floor(comm_uniform() * (double)na);
        if (idx >= na) idx = na - 1;
        A->data[k] = alphabet->data[idx];
    }
    return A;
}

/* Weighted variant: alphabet is the value vector, probs is the
 * matching probability vector (auto-normalised). */
matlab_mat *matlab_comm_randsrc_weighted(double m, double n,
                                          matlab_mat *alphabet,
                                          matlab_mat *probs) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    if (rm < 0) rm = 0; if (cn < 0) cn = 0;
    matlab_mat *A = mat_alloc(rm, cn);
    if (!alphabet || !probs) return A;
    int64_t na = alphabet->rows * alphabet->cols;
    int64_t np = probs->rows * probs->cols;
    if (na == 0 || np == 0) return A;
    if (np != na) return A;        /* shape mismatch -> zeros */
    /* Build cumulative distribution. */
    std::vector<double> cdf(na);
    double sum = 0.0;
    for (int64_t i = 0; i < na; ++i) sum += probs->data[i];
    if (sum <= 0.0) sum = 1.0;
    double running = 0.0;
    for (int64_t i = 0; i < na; ++i) {
        running += probs->data[i] / sum;
        cdf[i] = running;
    }
    cdf[na - 1] = 1.0;             /* guard rounding */
    for (int64_t k = 0; k < rm * cn; ++k) {
        double u = comm_uniform();
        int64_t lo = 0, hi = na - 1;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (cdf[mid] < u) lo = mid + 1; else hi = mid;
        }
        A->data[k] = alphabet->data[lo];
    }
    return A;
}

/* randerr(m, n, errs) -> m x n binary error matrix with exactly
 * `errs` ones per row, placed at uniform-random positions without
 * replacement. */
matlab_mat *matlab_comm_randerr(double m, double n, double errs) {
    int64_t rm = (int64_t)m, cn = (int64_t)n;
    if (rm < 0) rm = 0; if (cn < 0) cn = 0;
    int64_t e = (int64_t)errs;
    if (e < 0) e = 0;
    if (e > cn) e = cn;
    matlab_mat *A = mat_alloc(rm, cn);
    std::vector<int64_t> idx(cn);
    for (int64_t i = 0; i < rm; ++i) {
        for (int64_t j = 0; j < cn; ++j) idx[j] = j;
        /* Fisher-Yates partial shuffle: pull `e` indices to the
         * front, set those columns to 1.0. */
        for (int64_t k = 0; k < e; ++k) {
            int64_t pick = k + (int64_t)floor(comm_uniform() * (double)(cn - k));
            if (pick >= cn) pick = cn - 1;
            int64_t tmp = idx[k]; idx[k] = idx[pick]; idx[pick] = tmp;
            A->data[i * cn + idx[k]] = 1.0;
        }
    }
    return A;
}

/* ===== §2.4 bit / integer conversion ===================================== */

/* int2bit(ints, nbits) — MSB-first.
 *   Input  : column vector of non-negative integers, length L.
 *   Output : (L*nbits) x 1 column of bits; bit i of integer k is
 *            at row k*nbits + (nbits-1-i). */
matlab_mat *matlab_comm_int2bit(matlab_mat *ints, double nbits) {
    if (!ints) return mat_alloc(0, 0);
    int64_t L = ints->rows * ints->cols;
    int64_t nb = (int64_t)nbits;
    if (nb < 1) nb = 1;
    if (nb > 53) nb = 53;
    matlab_mat *out = mat_alloc(L * nb, 1);
    for (int64_t k = 0; k < L; ++k) {
        uint64_t v = (uint64_t)ints->data[k];
        for (int64_t i = 0; i < nb; ++i) {
            uint64_t bit = (v >> (nb - 1 - i)) & 1ULL;
            out->data[k * nb + i] = (double)bit;
        }
    }
    return out;
}

/* bit2int(bits, nbits) — MSB-first. Inverse of int2bit. */
matlab_mat *matlab_comm_bit2int(matlab_mat *bits, double nbits) {
    if (!bits) return mat_alloc(0, 0);
    int64_t Nb = bits->rows * bits->cols;
    int64_t nb = (int64_t)nbits;
    if (nb < 1) nb = 1;
    if (nb > 53) nb = 53;
    int64_t L = Nb / nb;
    matlab_mat *out = mat_alloc(L, 1);
    for (int64_t k = 0; k < L; ++k) {
        uint64_t v = 0;
        for (int64_t i = 0; i < nb; ++i) {
            v = (v << 1) | (((uint64_t)bits->data[k * nb + i]) & 1ULL);
        }
        out->data[k] = (double)v;
    }
    return out;
}

/* de2bi(d, n) — legacy LSB-first.
 *   Input  : column vector of L decimal values.
 *   Output : L x n matrix; row k has the bits of d[k] with the LSB
 *            at column 1. */
matlab_mat *matlab_comm_de2bi(matlab_mat *d, double nbits) {
    if (!d) return mat_alloc(0, 0);
    int64_t L = d->rows * d->cols;
    int64_t nb = (int64_t)nbits;
    if (nb < 1) nb = 1;
    if (nb > 53) nb = 53;
    matlab_mat *out = mat_alloc(L, nb);
    for (int64_t k = 0; k < L; ++k) {
        uint64_t v = (uint64_t)d->data[k];
        for (int64_t i = 0; i < nb; ++i) {
            out->data[k * nb + i] = (double)((v >> i) & 1ULL);
        }
    }
    return out;
}

/* bi2de(b) — legacy LSB-first. Input is L x n; output is L x 1. */
matlab_mat *matlab_comm_bi2de(matlab_mat *b) {
    if (!b) return mat_alloc(0, 0);
    int64_t L = b->rows;
    int64_t n = b->cols;
    if (n < 1) n = 1;
    if (n > 53) n = 53;
    matlab_mat *out = mat_alloc(L, 1);
    for (int64_t k = 0; k < L; ++k) {
        uint64_t v = 0;
        for (int64_t i = 0; i < n; ++i) {
            uint64_t bit = ((uint64_t)b->data[k * n + i]) & 1ULL;
            v |= (bit << i);
        }
        out->data[k] = (double)v;
    }
    return out;
}

/* ===== §2.5 awgn — additive white Gaussian noise channel ================== */

/* Internal helper. Computes signal power (mean(|x|^2)) over a real
 * vector / matrix. */
static double signal_power_real(const matlab_mat *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N == 0) return 0.0;
    double acc = 0.0;
    for (int64_t k = 0; k < N; ++k) acc += x->data[k] * x->data[k];
    return acc / (double)N;
}

static double signal_power_complex(const matlab_mat_c *x) {
    if (!x) return 0.0;
    int64_t N = x->rows * x->cols;
    if (N == 0) return 0.0;
    double acc = 0.0;
    for (int64_t k = 0; k < N; ++k)
        acc += x->re[k] * x->re[k] + x->im[k] * x->im[k];
    return acc / (double)N;
}

/* matlab_mat_c constructor exposed by runtime_complex.cpp. */
extern matlab_mat_c *mat_c_alloc(int64_t m, int64_t n);

/* awgn(x, snr_dB) — 'measured' signal power, dB SNR. Returns the
 * same descriptor kind as the input (real -> matlab_mat, complex ->
 * matlab_mat_c). The descriptor magic is sniffed from x's first
 * bytes; both arms reuse the shared PRNG kernel so rng-seeding is
 * deterministic. */
void *matlab_comm_awgn(void *x, double snr_dB) {
    if (!x) return NULL;
    if (mat_is_complex(x)) {
        const matlab_mat_c *xi = (const matlab_mat_c *)x;
        double sigP = signal_power_complex(xi);
        if (sigP <= 0.0) sigP = 1.0;
        double snr_lin = pow(10.0, snr_dB / 10.0);
        double noiseP = sigP / snr_lin;
        /* Complex noise: sigma^2/2 per axis so total variance == noiseP. */
        double sigma = sqrt(noiseP * 0.5);
        matlab_mat_c *out = mat_c_alloc(xi->rows, xi->cols);
        int64_t N = xi->rows * xi->cols;
        for (int64_t k = 0; k < N; ++k) {
            out->re[k] = xi->re[k] + sigma * comm_normal();
            out->im[k] = xi->im[k] + sigma * comm_normal();
        }
        return out;
    }
    const matlab_mat *xi = (const matlab_mat *)x;
    double sigP = signal_power_real(xi);
    if (sigP <= 0.0) sigP = 1.0;
    double snr_lin = pow(10.0, snr_dB / 10.0);
    double noiseP = sigP / snr_lin;
    double sigma = sqrt(noiseP);
    matlab_mat *out = mat_alloc(xi->rows, xi->cols);
    int64_t N = xi->rows * xi->cols;
    for (int64_t k = 0; k < N; ++k)
        out->data[k] = xi->data[k] + sigma * comm_normal();
    return out;
}

/* awgn(x, snr_dB, sigpower_dBW) — explicit signal power. */
void *matlab_comm_awgn_p(void *x, double snr_dB, double sigpower_dBW) {
    if (!x) return NULL;
    double sigP = pow(10.0, sigpower_dBW / 10.0);
    double snr_lin = pow(10.0, snr_dB / 10.0);
    double noiseP = sigP / snr_lin;
    if (mat_is_complex(x)) {
        const matlab_mat_c *xi = (const matlab_mat_c *)x;
        double sigma = sqrt(noiseP * 0.5);
        matlab_mat_c *out = mat_c_alloc(xi->rows, xi->cols);
        int64_t N = xi->rows * xi->cols;
        for (int64_t k = 0; k < N; ++k) {
            out->re[k] = xi->re[k] + sigma * comm_normal();
            out->im[k] = xi->im[k] + sigma * comm_normal();
        }
        return out;
    }
    const matlab_mat *xi = (const matlab_mat *)x;
    double sigma = sqrt(noiseP);
    matlab_mat *out = mat_alloc(xi->rows, xi->cols);
    int64_t N = xi->rows * xi->cols;
    for (int64_t k = 0; k < N; ++k)
        out->data[k] = xi->data[k] + sigma * comm_normal();
    return out;
}

/* ===== §2.6 biterr / symerr ============================================== */

/* biterr(x, y) — element-wise comparison; both args are assumed
 * already to be 0/1 bit columns. Returns the bit-error count. The
 * BER companion `matlab_comm_biterr_ratio` returns nerr / N. */
double matlab_comm_biterr_count(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return 0.0;
    int64_t N = std::min(x->rows * x->cols, y->rows * y->cols);
    int64_t err = 0;
    for (int64_t k = 0; k < N; ++k) {
        uint64_t a = (uint64_t)x->data[k] & 1ULL;
        uint64_t b = (uint64_t)y->data[k] & 1ULL;
        if (a != b) ++err;
    }
    return (double)err;
}

double matlab_comm_biterr_ratio(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return 0.0;
    int64_t N = std::min(x->rows * x->cols, y->rows * y->cols);
    if (N == 0) return 0.0;
    int64_t err = 0;
    for (int64_t k = 0; k < N; ++k) {
        uint64_t a = (uint64_t)x->data[k] & 1ULL;
        uint64_t b = (uint64_t)y->data[k] & 1ULL;
        if (a != b) ++err;
    }
    return (double)err / (double)N;
}

/* biterr(x, y, k) — input is k-bit symbols; unpack to bits first
 * (MSB-first per int2bit's convention). Returns count + ratio
 * variants. */
double matlab_comm_biterr_count_k(matlab_mat *x, matlab_mat *y, double kbits) {
    if (!x || !y) return 0.0;
    int64_t N = std::min(x->rows * x->cols, y->rows * y->cols);
    int64_t kb = (int64_t)kbits;
    if (kb < 1) kb = 1;
    if (kb > 53) kb = 53;
    int64_t err = 0;
    for (int64_t k = 0; k < N; ++k) {
        uint64_t a = (uint64_t)x->data[k];
        uint64_t b = (uint64_t)y->data[k];
        uint64_t d = (a ^ b) & ((kb >= 64) ? ~0ULL : ((1ULL << kb) - 1ULL));
        while (d) { err += d & 1ULL; d >>= 1; }
    }
    return (double)err;
}

double matlab_comm_biterr_ratio_k(matlab_mat *x, matlab_mat *y, double kbits) {
    if (!x || !y) return 0.0;
    int64_t N = std::min(x->rows * x->cols, y->rows * y->cols);
    int64_t kb = (int64_t)kbits;
    if (kb < 1) kb = 1;
    if (kb > 53) kb = 53;
    if (N == 0 || kb == 0) return 0.0;
    int64_t err = 0;
    for (int64_t k = 0; k < N; ++k) {
        uint64_t a = (uint64_t)x->data[k];
        uint64_t b = (uint64_t)y->data[k];
        uint64_t d = (a ^ b) & ((kb >= 64) ? ~0ULL : ((1ULL << kb) - 1ULL));
        while (d) { err += d & 1ULL; d >>= 1; }
    }
    return (double)err / ((double)N * (double)kb);
}

/* symerr(x, y) — element-wise mismatch count and ratio. */
double matlab_comm_symerr_count(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return 0.0;
    int64_t N = std::min(x->rows * x->cols, y->rows * y->cols);
    int64_t err = 0;
    for (int64_t k = 0; k < N; ++k)
        if (x->data[k] != y->data[k]) ++err;
    return (double)err;
}

double matlab_comm_symerr_ratio(matlab_mat *x, matlab_mat *y) {
    if (!x || !y) return 0.0;
    int64_t N = std::min(x->rows * x->cols, y->rows * y->cols);
    if (N == 0) return 0.0;
    int64_t err = 0;
    for (int64_t k = 0; k < N; ++k)
        if (x->data[k] != y->data[k]) ++err;
    return (double)err / (double)N;
}

/* ===== Tier 2 prerequisites — erfc / qfunc =============================== */

/* libc `erfc` is available everywhere we link; expose it through a
 * runtime symbol so the dispatch table can route `erfc` (matrix form
 * comes later — Tier 2 only needs the scalar). */
double matlab_comm_erfc_s(double x) {
    return erfc(x);
}

/* Q-function = 0.5·erfc(x/√2). One-line wrapper. */
double matlab_comm_qfunc_s(double x) {
    return 0.5 * erfc(x / sqrt(2.0));
}

/* ===== §4.x Gray-code helpers (shared between PAM / QAM / PSK) =========== */

/* Binary -> reflected-Gray (k-bit truncation done by caller). */
static inline uint64_t bin2gray_u(uint64_t b) {
    return b ^ (b >> 1);
}

/* Gray -> binary inverse. */
static inline uint64_t gray2bin_u(uint64_t g) {
    uint64_t b = g;
    for (int s = 1; s < 64; s <<= 1) b ^= b >> s;
    return b;
}

/* ===== §4.1 PAM — `pammod` / `pamdemod` ================================== *
 *
 * Map x ∈ [0, M-1] to the canonical M-PAM constellation
 *     a_k = 2·k - (M-1),  k ∈ [0, M-1]
 * Default natural mapping; Gray mapping is the same lookup against the
 * de-Gray'd integer index. Output is real (matlab_mat).
 *
 * Mapping codes (numeric tag instead of string selectors):
 *   0 = natural binary
 *   1 = Gray
 *
 * Phase rotation `ini_phase` (in radians) — when nonzero, the output
 * becomes complex via the dedicated `*_phase` companion.
 */

static inline int64_t pam_index_from_data(uint64_t v, int64_t M, int order) {
    /* `order` ∈ {0 = bin, 1 = gray}. For Gray the data integer is the
     * Gray-coded label; the underlying constellation position is its
     * gray2bin inverse. */
    uint64_t bin = (order == 1) ? gray2bin_u(v) : v;
    if (M < 1) M = 1;
    int64_t idx = (int64_t)(bin % (uint64_t)M);
    return idx;
}

static inline uint64_t pam_data_from_index(int64_t idx, int64_t M, int order) {
    if (M < 1) M = 1;
    uint64_t bin = (uint64_t)((idx % M + M) % M);
    return (order == 1) ? bin2gray_u(bin) : bin;
}

matlab_mat *matlab_comm_pammod(matlab_mat *x, double Md, double order) {
    if (!x) return mat_alloc(0, 0);
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    int ord = (int)order;
    int64_t N = x->rows * x->cols;
    matlab_mat *out = mat_alloc(x->rows, x->cols);
    for (int64_t k = 0; k < N; ++k) {
        int64_t idx = pam_index_from_data((uint64_t)x->data[k], M, ord);
        out->data[k] = (double)(2 * idx - (M - 1));
    }
    return out;
}

/* `pamdemod`: hard-decision threshold to nearest constellation point.
 * Decoded levels live at -(M-1), -(M-3), ..., (M-1) — even integer
 * spacing of 2 — so the threshold rule is `idx = round((y + (M-1)) / 2)`
 * clamped to [0, M-1]. Result is the integer label (Gray-coded if
 * order=1). */
matlab_mat *matlab_comm_pamdemod(matlab_mat *y, double Md, double order) {
    if (!y) return mat_alloc(0, 0);
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    int ord = (int)order;
    int64_t N = y->rows * y->cols;
    matlab_mat *out = mat_alloc(y->rows, y->cols);
    for (int64_t k = 0; k < N; ++k) {
        double v = y->data[k];
        int64_t idx = (int64_t)floor((v + (double)(M - 1)) / 2.0 + 0.5);
        if (idx < 0) idx = 0; if (idx > M - 1) idx = M - 1;
        out->data[k] = (double)pam_data_from_index(idx, M, ord);
    }
    return out;
}

/* ===== §4.3 PSK — `pskmod` / `pskdemod` ================================== *
 *
 * Map x ∈ [0, M-1] to phase = ini_phase + 2π·k/M, where k is the
 * underlying-bin index (with Gray decoding if order=1). Returns
 * matlab_mat_c. Demod picks the nearest phase via atan2; tie-breaks
 * fall to the lower-index neighbour. */

extern matlab_mat_c *mat_c_alloc(int64_t m, int64_t n);

matlab_mat_c *matlab_comm_pskmod(matlab_mat *x, double Md,
                                  double ini_phase, double order) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    int ord = (int)order;
    int64_t N = x->rows * x->cols;
    matlab_mat_c *out = mat_c_alloc(x->rows, x->cols);
    double step = 2.0 * M_PI / (double)M;
    for (int64_t k = 0; k < N; ++k) {
        int64_t idx = pam_index_from_data((uint64_t)x->data[k], M, ord);
        double phase = ini_phase + step * (double)idx;
        out->re[k] = cos(phase);
        out->im[k] = sin(phase);
    }
    return out;
}

matlab_mat *matlab_comm_pskdemod(matlab_mat_c *y, double Md,
                                  double ini_phase, double order) {
    if (!y) return mat_alloc(0, 0);
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    int ord = (int)order;
    int64_t N = y->rows * y->cols;
    matlab_mat *out = mat_alloc(y->rows, y->cols);
    double step = 2.0 * M_PI / (double)M;
    for (int64_t k = 0; k < N; ++k) {
        double phi = atan2(y->im[k], y->re[k]) - ini_phase;
        /* Wrap to [-π, π) then to [0, 2π). */
        double q = phi / step;
        int64_t idx = (int64_t)floor(q + 0.5);
        /* Modulo into [0, M). */
        idx = ((idx % M) + M) % M;
        out->data[k] = (double)pam_data_from_index(idx, M, ord);
    }
    return out;
}

/* ===== §4.2 QAM — `qammod` / `qamdemod` ================================== *
 *
 * Square M-QAM (M = sqrt(M)^2): independent k=log2(sqrt(M)) PAM
 * components on the I and Q axes; the data integer's high bits select
 * the I level, low bits the Q level, with optional Gray re-mapping per
 * axis.
 *
 * Cross-QAM (M = 8, 32, 128) is the standard "L-shape minus corners"
 * arrangement — we ship the M=8 and M=32 tables explicitly; M=128 etc.
 * are deferred to a follow-on slice.
 *
 * UnitAveragePower normalization scales by 1/√(mean_power). For square
 * M-QAM with the canonical {-(M-1), …, (M-1)} per axis,
 *   mean_power = 2 · (M − 1) / 3
 * (standard textbook result). */

static int qam_axis_bits(int64_t M, int *kx, int *ky) {
    /* Square M: M = 4, 16, 64, 256, 1024. */
    int k = 0;
    while ((1LL << k) < M) ++k;
    if ((1LL << k) != M) return -1;
    if ((k % 2) == 0) { *kx = k / 2; *ky = k / 2; return 0; }
    /* Cross-QAM: M = 8 (3 bits, 4x2 minus 0 cross-form just collapses
     * to a 2x4 grid here) — we treat M=8 as 2x4 rectangular; M=32 as
     * the standard 6x6-minus-corner cross. Anything else cross is a
     * not-yet-supported lane. */
    if (k == 3) { *kx = 2; *ky = 1; return 0; }   /* 4x2 */
    if (k == 5) { *kx = 3; *ky = 2; return 0; }   /* 8x4 */
    return -2;
}

/* Compute mean symbol energy for square M-QAM on the canonical
 * {-(L-1), ..., (L-1)} per axis with L = sqrt(M). */
static double qam_mean_power_square(int64_t M) {
    int kx, ky;
    if (qam_axis_bits(M, &kx, &ky) != 0) return 1.0;
    int64_t Lx = 1LL << kx, Ly = 1LL << ky;
    double Px = (double)(Lx * Lx - 1) / 3.0;
    double Py = (double)(Ly * Ly - 1) / 3.0;
    return Px + Py;
}

matlab_mat_c *matlab_comm_qammod(matlab_mat *x, double Md,
                                  double order, double unit_avg) {
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    int ord = (int)order;
    int kx, ky;
    if (!x || qam_axis_bits(M, &kx, &ky) != 0) return mat_c_alloc(0, 0);
    int64_t Lx = 1LL << kx, Ly = 1LL << ky;
    int64_t N = x->rows * x->cols;
    matlab_mat_c *out = mat_c_alloc(x->rows, x->cols);
    double scale = 1.0;
    if (unit_avg > 0.5) {
        double Pavg = qam_mean_power_square(M);
        if (Pavg > 0) scale = 1.0 / sqrt(Pavg);
    }
    for (int64_t k = 0; k < N; ++k) {
        uint64_t v = (uint64_t)x->data[k];
        /* Split bits: high kx for I, low ky for Q. */
        uint64_t hi = (v >> ky) & ((1ULL << kx) - 1);
        uint64_t lo = v & ((1ULL << ky) - 1);
        if (ord == 1) {
            hi = gray2bin_u(hi);
            lo = gray2bin_u(lo);
        }
        out->re[k] = scale * (double)(2 * (int64_t)hi - (Lx - 1));
        out->im[k] = scale * (double)(2 * (int64_t)lo - (Ly - 1));
    }
    return out;
}

/* qamdemod hard-decision: per-axis nearest-PAM-level decode. Result is
 * the integer label (Gray-coded if order=1). */
matlab_mat *matlab_comm_qamdemod(matlab_mat_c *y, double Md,
                                  double order, double unit_avg) {
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    int ord = (int)order;
    int kx, ky;
    if (!y || qam_axis_bits(M, &kx, &ky) != 0) return mat_alloc(0, 0);
    int64_t Lx = 1LL << kx, Ly = 1LL << ky;
    int64_t N = y->rows * y->cols;
    matlab_mat *out = mat_alloc(y->rows, y->cols);
    double scale = 1.0;
    if (unit_avg > 0.5) {
        double Pavg = qam_mean_power_square(M);
        if (Pavg > 0) scale = sqrt(Pavg);
    }
    for (int64_t k = 0; k < N; ++k) {
        double re_v = scale * y->re[k];
        double im_v = scale * y->im[k];
        int64_t i_idx = (int64_t)floor((re_v + (double)(Lx - 1)) / 2.0 + 0.5);
        int64_t q_idx = (int64_t)floor((im_v + (double)(Ly - 1)) / 2.0 + 0.5);
        if (i_idx < 0) i_idx = 0; if (i_idx > Lx - 1) i_idx = Lx - 1;
        if (q_idx < 0) q_idx = 0; if (q_idx > Ly - 1) q_idx = Ly - 1;
        uint64_t hi = (uint64_t)i_idx, lo = (uint64_t)q_idx;
        if (ord == 1) { hi = bin2gray_u(hi); lo = bin2gray_u(lo); }
        out->data[k] = (double)((hi << ky) | lo);
    }
    return out;
}

/* qamdemod 'bit' output: returns (N · log2(M)) x 1 column of bits in
 * MSB-first order (matches the int2bit convention shipped in Tier 1). */
matlab_mat *matlab_comm_qamdemod_bit(matlab_mat_c *y, double Md,
                                      double order, double unit_avg) {
    matlab_mat *labels = matlab_comm_qamdemod(y, Md, order, unit_avg);
    if (!labels) return mat_alloc(0, 0);
    int64_t M = (int64_t)Md;
    int kx, ky;
    if (qam_axis_bits(M, &kx, &ky) != 0) return mat_alloc(0, 0);
    int kb = kx + ky;
    int64_t L = labels->rows * labels->cols;
    matlab_mat *bits = mat_alloc(L * kb, 1);
    for (int64_t k = 0; k < L; ++k) {
        uint64_t v = (uint64_t)labels->data[k];
        for (int i = 0; i < kb; ++i) {
            bits->data[k * kb + i] = (double)((v >> (kb - 1 - i)) & 1ULL);
        }
    }
    free(labels->data); free(labels);
    return bits;
}

/* qamdemod 'llr' output: per-bit log-likelihood ratios via the
 * max-log approximation. Returns (N · log2(M)) x 1 column of LLRs in
 * MSB-first bit order. The noise variance is supplied explicitly
 * (callers pass sigma^2 estimated from `awgn`'s SNR setting). */
matlab_mat *matlab_comm_qamdemod_llr(matlab_mat_c *y, double Md,
                                      double order, double unit_avg,
                                      double noise_var) {
    int64_t M = (int64_t)Md;
    int ord = (int)order;
    int kx, ky;
    if (!y || qam_axis_bits(M, &kx, &ky) != 0) return mat_alloc(0, 0);
    int kb = kx + ky;
    int64_t Lx = 1LL << kx, Ly = 1LL << ky;
    int64_t N = y->rows * y->cols;
    double scale = 1.0;
    if (unit_avg > 0.5) {
        double Pavg = qam_mean_power_square(M);
        if (Pavg > 0) scale = sqrt(Pavg);
    }
    if (noise_var <= 0.0) noise_var = 1e-9;
    matlab_mat *llr = mat_alloc(N * kb, 1);
    /* For each received symbol, score every constellation point and
     * fold the max-log decision per bit position.  Constellation
     * point for data label `s` sits at (2·bin_hi − (Lx−1),
     * 2·bin_lo − (Ly−1)) where (bin_hi, bin_lo) = gray2bin halves of
     * (hi, lo) when order==Gray; for binary mapping bin_*==hi/lo. */
    for (int64_t k = 0; k < N; ++k) {
        double re_v = scale * y->re[k];
        double im_v = scale * y->im[k];
        for (int b = 0; b < kb; ++b) {
            double max0 = -1e300, max1 = -1e300;
            uint64_t mask = 1ULL << (kb - 1 - b);
            for (int64_t s = 0; s < M; ++s) {
                uint64_t v = (uint64_t)s;
                uint64_t hi = (v >> ky) & ((1ULL << kx) - 1);
                uint64_t lo = v & ((1ULL << ky) - 1);
                uint64_t bin_hi = (ord == 1) ? gray2bin_u(hi) : hi;
                uint64_t bin_lo = (ord == 1) ? gray2bin_u(lo) : lo;
                double cx = (double)(2 * (int64_t)bin_hi - (Lx - 1));
                double cy = (double)(2 * (int64_t)bin_lo - (Ly - 1));
                double d2 = (re_v - cx) * (re_v - cx) +
                            (im_v - cy) * (im_v - cy);
                double metric = -d2 / (2.0 * noise_var);
                if (v & mask) { if (metric > max1) max1 = metric; }
                else          { if (metric > max0) max0 = metric; }
            }
            /* Standard sign convention: positive LLR favours bit=0. */
            llr->data[k * kb + b] = max0 - max1;
        }
    }
    return llr;
}

/* ===== §4.6 genqammod / genqamdemod ====================================== *
 *
 * User-supplied constellation: alphabet is a complex matrix
 * (matlab_mat_c) whose entries are the constellation points indexed by
 * the data integer. Output is the per-symbol complex value. Demod is
 * nearest-point in Euclidean distance. */

matlab_mat_c *matlab_comm_genqammod(matlab_mat *x, matlab_mat_c *alphabet) {
    if (!x || !alphabet) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int64_t Na = alphabet->rows * alphabet->cols;
    if (Na == 0) return mat_c_alloc(0, 0);
    matlab_mat_c *out = mat_c_alloc(x->rows, x->cols);
    for (int64_t k = 0; k < N; ++k) {
        int64_t idx = (int64_t)x->data[k];
        idx = ((idx % Na) + Na) % Na;
        out->re[k] = alphabet->re[idx];
        out->im[k] = alphabet->im[idx];
    }
    return out;
}

matlab_mat *matlab_comm_genqamdemod(matlab_mat_c *y, matlab_mat_c *alphabet) {
    if (!y || !alphabet) return mat_alloc(0, 0);
    int64_t N = y->rows * y->cols;
    int64_t Na = alphabet->rows * alphabet->cols;
    matlab_mat *out = mat_alloc(y->rows, y->cols);
    for (int64_t k = 0; k < N; ++k) {
        double best = 1e300;
        int64_t best_i = 0;
        for (int64_t i = 0; i < Na; ++i) {
            double dr = y->re[k] - alphabet->re[i];
            double di = y->im[k] - alphabet->im[i];
            double d2 = dr * dr + di * di;
            if (d2 < best) { best = d2; best_i = i; }
        }
        out->data[k] = (double)best_i;
    }
    return out;
}

/* ===== §4.7 rcosdesign / gaussdesign ===================================== *
 *
 * Root-raised-cosine pulse-shaping FIR.
 *   beta   = roll-off ∈ [0, 1]
 *   span   = symbol span (FIR length = span*sps + 1)
 *   sps    = samples per symbol
 *   shape  = 0 = 'sqrt' (RRC, default), 1 = 'normal' (full RC)
 *
 * Closed-form impulse response with L'Hôpital handling at t=0 and
 * t = ±span/(4·beta); unit-energy normalisation. */

matlab_mat *matlab_comm_rcosdesign(double beta, double span,
                                    double spsd, double shape) {
    int64_t sps = (int64_t)spsd;
    int64_t sp  = (int64_t)span;
    if (sps < 1) sps = 1;
    if (sp  < 1) sp  = 1;
    int64_t N = sp * sps + 1;
    matlab_mat *b = mat_alloc(N, 1);
    double Ts = 1.0;   /* symbol period; we normalise so t-axis is in symbols */
    double dt = Ts / (double)sps;
    int sh = (int)shape;
    double eps = 1e-12;
    for (int64_t n = 0; n < N; ++n) {
        double t = (n - (N - 1) / 2.0) * dt;
        double h;
        if (sh == 1) {
            /* Raised-cosine (normal). */
            if (fabs(t) < eps) {
                h = 1.0;
            } else if (beta > 0.0 &&
                       fabs(fabs(t) - Ts / (2.0 * beta)) < eps) {
                h = (M_PI / 4.0) * sin(M_PI / (2.0 * beta)) /
                    (M_PI / (2.0 * beta));
            } else {
                double a = M_PI * t / Ts;
                double num = sin(a) * cos(beta * a);
                double den = a * (1.0 - 4.0 * beta * beta * t * t / (Ts * Ts));
                h = num / den;
            }
        } else {
            /* Root-raised-cosine. */
            if (fabs(t) < eps) {
                h = (1.0 / sqrt(Ts)) *
                    (1.0 - beta + 4.0 * beta / M_PI);
            } else if (beta > 0.0 &&
                       fabs(fabs(t) - Ts / (4.0 * beta)) < eps) {
                double q1 = (1.0 + 2.0 / M_PI) * sin(M_PI / (4.0 * beta));
                double q2 = (1.0 - 2.0 / M_PI) * cos(M_PI / (4.0 * beta));
                h = (beta / sqrt(2.0 * Ts)) * (q1 + q2);
            } else {
                double a = M_PI * t / Ts;
                double num = sin(a * (1.0 - beta)) +
                             4.0 * beta * t / Ts * cos(a * (1.0 + beta));
                double den = a * (1.0 - (4.0 * beta * t / Ts) *
                                         (4.0 * beta * t / Ts));
                h = (1.0 / sqrt(Ts)) * num / den;
            }
        }
        b->data[n] = h;
    }
    /* Unit-energy normalisation. */
    double e = 0.0;
    for (int64_t n = 0; n < N; ++n) e += b->data[n] * b->data[n];
    if (e > 0.0) {
        double s = 1.0 / sqrt(e);
        for (int64_t n = 0; n < N; ++n) b->data[n] *= s;
    }
    return b;
}

/* Gaussian filter for GMSK / GFSK.
 *   bt    = bandwidth-symbol-time product (0.3 for GSM, 0.5 for Bluetooth)
 *   span  = symbol span (FIR length = span*sps + 1)
 *   sps   = samples per symbol
 *
 * Impulse response per the standard GMSK formula. */
matlab_mat *matlab_comm_gaussdesign(double bt, double span, double spsd) {
    int64_t sps = (int64_t)spsd;
    int64_t sp  = (int64_t)span;
    if (sps < 1) sps = 1;
    if (sp  < 1) sp  = 1;
    int64_t N = sp * sps + 1;
    matlab_mat *b = mat_alloc(N, 1);
    /* α = sqrt(ln(2) / 2) / BT */
    double alpha = sqrt(log(2.0) / 2.0) / bt;
    double dt = 1.0 / (double)sps;
    for (int64_t n = 0; n < N; ++n) {
        double t = (n - (N - 1) / 2.0) * dt;
        b->data[n] = (sqrt(M_PI) / alpha) *
                      exp(-(M_PI * M_PI * t * t) / (alpha * alpha));
    }
    /* Normalise so the impulse response sums to 1 (MATLAB convention). */
    double s = 0.0;
    for (int64_t n = 0; n < N; ++n) s += b->data[n];
    if (s > 0.0) {
        double inv = 1.0 / s;
        for (int64_t n = 0; n < N; ++n) b->data[n] *= inv;
    }
    return b;
}

/* ===== §4.8 berawgn — closed-form BER under AWGN ========================= *
 *
 * Numeric `mod` selector:
 *   0 = PAM   :  P_e = 2·(M-1)/(M·log2(M)) · Q(sqrt(6·log2(M)·EbN0 / (M^2-1)))
 *   1 = PSK   :  for M=2: Q(sqrt(2·EbN0)); for M=4 (QPSK): Q(sqrt(2·EbN0));
 *                M>=8: approx (2/log2(M)) · Q(sqrt(2·log2(M)·EbN0)·sin(π/M))
 *   2 = QAM   :  P_e ≈ (4/log2(M)) · (1 - 1/sqrt(M)) · Q(sqrt(3·log2(M)·EbN0/(M-1)))
 *   3 = DPSK  :  M=2: 0.5·exp(-EbN0); M>=4: approx 2/log2(M) · Q(sqrt(2·log2(M)·EbN0)·sin(π/(2·M)))
 *   4 = FSK orth. coherent :  P_e = Q(sqrt(EbN0·log2(M)))  (binary case; M-ary uses union bound)
 *   5 = FSK orth. noncoh.  :  Binary: 0.5·exp(-EbN0/2); M-ary union bound.
 *
 * EbN0 input is in dB. Output is the per-bit BER. */

static double q_func(double x) { return 0.5 * erfc(x / sqrt(2.0)); }

double matlab_comm_berawgn_s(double ebn0_dB, double Md, double mod) {
    int64_t M = (int64_t)Md;
    if (M < 2) M = 2;
    double k = log(M) / log(2.0);
    if (k < 1.0) k = 1.0;
    double EbN0 = pow(10.0, ebn0_dB / 10.0);
    int m = (int)mod;
    switch (m) {
    case 0: { /* PAM */
        double arg = sqrt(6.0 * k * EbN0 / (double)(M * M - 1));
        return 2.0 * (double)(M - 1) / ((double)M * k) * q_func(arg);
    }
    case 1: { /* PSK */
        if (M == 2 || M == 4) return q_func(sqrt(2.0 * EbN0));
        return (2.0 / k) * q_func(sqrt(2.0 * k * EbN0) * sin(M_PI / (double)M));
    }
    case 2: { /* QAM (square M) */
        double c = 4.0 / k * (1.0 - 1.0 / sqrt((double)M));
        double arg = sqrt(3.0 * k * EbN0 / (double)(M - 1));
        return c * q_func(arg);
    }
    case 3: { /* DPSK */
        if (M == 2) return 0.5 * exp(-EbN0);
        return (2.0 / k) * q_func(sqrt(2.0 * k * EbN0) *
                                   sin(M_PI / (2.0 * (double)M)));
    }
    case 4: { /* FSK orthogonal coherent (binary scope) */
        return q_func(sqrt(EbN0 * k));
    }
    case 5: { /* FSK orthogonal non-coherent (binary scope) */
        return 0.5 * exp(-EbN0 / 2.0);
    }
    default: return 0.5;
    }
}

/* ===== §4.9 scatterplot — numeric return form ============================ *
 *
 * MATLAB's scatterplot opens a figure window; the headless lane just
 * needs the underlying (real, imag) pair as an N x 2 matrix. The Cairo
 * scatter plot is a separate call site the user assembles on top. */
matlab_mat *matlab_comm_scatterplot(matlab_mat_c *x) {
    if (!x) return mat_alloc(0, 2);
    int64_t N = x->rows * x->cols;
    matlab_mat *out = mat_alloc(N, 2);
    for (int64_t k = 0; k < N; ++k) {
        out->data[k * 2 + 0] = x->re[k];
        out->data[k * 2 + 1] = x->im[k];
    }
    return out;
}

/* ===== Tier-3 channel coding (function-form) ============================ *
 *
 * docs/comm_toolbox_roadmap.md §5.  The CRC System Object form
 * (comm.CRCGenerator / comm.CRCDetector) is still gated on the
 * SO lowering fix, but the bare-function CRC interface is shipped
 * here.  poly2trellis / convenc / vitdec, Hamming, and the block
 * interleavers are function-form by default; they land regardless.
 *
 * BCH / Reed-Solomon and the gf(2^m) descriptor are deliberately
 * deferred — they need a new typed runtime descriptor, ~2 wk on its
 * own.  LDPC / Turbo / Polar stay carved-out per the roadmap §5.4.
 */

/* CRC bit-shift-register over the bit stream.  `poly_bits` is the
 * column vector representation of the generator polynomial of
 * length `n+1` (so a degree-16 CRC has 17 bits, MSB first), with
 * a leading 1.  We pass `poly_int` instead — the user passes the
 * polynomial as a non-negative integer whose binary representation
 * starts at the implicit leading 1; e.g. CRC-16-CCITT 0x1021 stays
 * 0x1021 and `nbits` is 16.
 *
 * To avoid string args we expose two siblings:
 *   crcGenerate(bits, poly_int, nbits) -> bits with CRC appended
 *   crcCheck   (bits, poly_int, nbits) -> 0 if CRC matches, 1 otherwise
 *   crcStrip   (bits, nbits)           -> bits[1:end-nbits] convenience
 */

static uint64_t crc_remainder(const matlab_mat *bits, int64_t N,
                               uint64_t poly, int nbits) {
    if (nbits < 1) nbits = 1;
    if (nbits > 63) nbits = 63;
    uint64_t mask = (nbits >= 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
    uint64_t rem = 0;
    for (int64_t k = 0; k < N; ++k) {
        uint64_t b = ((uint64_t)bits->data[k]) & 1ULL;
        uint64_t top = (rem >> (nbits - 1)) & 1ULL;
        rem = ((rem << 1) | b) & mask;
        if (top) rem ^= poly;
    }
    /* Pad with `nbits` trailing zeros to flush the register. */
    for (int i = 0; i < nbits; ++i) {
        uint64_t top = (rem >> (nbits - 1)) & 1ULL;
        rem = (rem << 1) & mask;
        if (top) rem ^= poly;
    }
    return rem;
}

matlab_mat *matlab_comm_crc_generate(matlab_mat *bits, double poly_int_d,
                                      double nbits_d) {
    if (!bits) return mat_alloc(0, 0);
    int64_t N = bits->rows * bits->cols;
    uint64_t poly = (uint64_t)poly_int_d;
    int nbits = (int)nbits_d;
    if (nbits < 1) nbits = 1; if (nbits > 63) nbits = 63;
    uint64_t rem = crc_remainder(bits, N, poly, nbits);
    matlab_mat *out = mat_alloc(N + nbits, 1);
    for (int64_t k = 0; k < N; ++k) out->data[k] = bits->data[k];
    for (int i = 0; i < nbits; ++i) {
        uint64_t b = (rem >> (nbits - 1 - i)) & 1ULL;
        out->data[N + i] = (double)b;
    }
    return out;
}

/* crcCheck — returns 0 if the trailing nbits CRC bits of `bits` match
 * the recomputed CRC over the leading payload; 1 otherwise.  Operates
 * over the full received stream including the appended CRC. */
double matlab_comm_crc_check(matlab_mat *bits, double poly_int_d,
                              double nbits_d) {
    if (!bits) return 1.0;
    int64_t N = bits->rows * bits->cols;
    int nbits = (int)nbits_d;
    if (nbits < 1) nbits = 1; if (nbits > 63) nbits = 63;
    if (N <= nbits) return 1.0;
    uint64_t poly = (uint64_t)poly_int_d;
    /* Recompute the CRC over the payload portion. */
    matlab_mat payload = *bits;
    payload.rows = N - nbits;
    payload.cols = 1;
    uint64_t rem = crc_remainder(&payload, N - nbits, poly, nbits);
    uint64_t received = 0;
    for (int i = 0; i < nbits; ++i) {
        uint64_t b = ((uint64_t)bits->data[N - nbits + i]) & 1ULL;
        received = (received << 1) | b;
    }
    return rem == received ? 0.0 : 1.0;
}

/* crcStrip(bits, nbits) — payload-only view (a fresh allocation;
 * we don't slice in place). */
matlab_mat *matlab_comm_crc_strip(matlab_mat *bits, double nbits_d) {
    if (!bits) return mat_alloc(0, 0);
    int64_t N = bits->rows * bits->cols;
    int nbits = (int)nbits_d;
    if (nbits < 0) nbits = 0;
    if (N <= nbits) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(N - nbits, 1);
    for (int64_t k = 0; k < N - nbits; ++k) out->data[k] = bits->data[k];
    return out;
}

/* ===== §5.2 convolutional codes — poly2trellis / convenc / vitdec ======== *
 *
 * poly2trellis(K, gens) builds the trellis struct for a rate 1/n
 * non-recursive convolutional encoder with constraint length K and
 * generator polynomials `gens` (a 1×n row vector of integers — user
 * supplies them in decimal form; the canonical octal notation is up
 * to them to convert with `oct2dec` for now).
 *
 * Returned struct fields:
 *   numInputSymbols  = 2
 *   numOutputSymbols = 2^n
 *   numStates        = 2^(K-1)
 *   nextStates       (numStates × 2 matrix, next state per input bit)
 *   outputs          (numStates × 2 matrix, output integer per input bit)
 */

extern matlab_struct *matlab_struct_new(void);
extern void matlab_struct_set_f64(matlab_struct *s, const char *name,
                                   int64_t len, double v);
extern void matlab_struct_set_mat(matlab_struct *s, const char *name,
                                   int64_t len, matlab_mat *m);

/* Compute the encoded output bit (parity over (state || input) bits
 * gated by the generator polynomial mask).  `state_in` is the (K-1)
 * lower bits; `input` is the new bit shifted in at the top. */
static int conv_output_bit(uint64_t poly_mask, int K,
                            uint64_t state_in, int input) {
    uint64_t reg = (state_in << 1) | ((uint64_t)input & 1ULL);
    (void)K;
    uint64_t masked = reg & poly_mask;
    int parity = 0;
    while (masked) { parity ^= (int)(masked & 1ULL); masked >>= 1; }
    return parity;
}

matlab_struct *matlab_comm_poly2trellis(double Kd, matlab_mat *gens) {
    int K = (int)Kd;
    if (K < 2) K = 2;
    if (K > 30) K = 30;
    int n = gens ? (int)(gens->rows * gens->cols) : 1;
    if (n < 1) n = 1;
    if (n > 8) n = 8;
    int64_t S = 1LL << (K - 1);
    matlab_mat *nextStates = mat_alloc(S, 2);
    matlab_mat *outputs    = mat_alloc(S, 2);
    /* For each (state, input), shift register on the LEFT (MSB):
     *   reg = (input << (K-1)) | state, except convention varies; we
     * follow MATLAB's: state bits are the (K-1) MOST-RECENT inputs
     * with the newest at the MSB; the encoder receives a new bit and
     * outputs n bits in the order gens[0], gens[1], ... .
     * Specifically: reg_in = ((state << 1) | input) but the parity
     * computation only uses the lowest K bits of the polynomial
     * relative to the register's K bits — gens[i] is interpreted as
     * a K-bit mask with the leading 1 at bit K-1. */
    for (int64_t s = 0; s < S; ++s) {
        for (int u = 0; u <= 1; ++u) {
            /* Next-state is the new register with the oldest bit dropped. */
            uint64_t reg = (((uint64_t)s) << 1) | (uint64_t)u;
            int64_t ns = (int64_t)(reg & (uint64_t)(S - 1));
            nextStates->data[s * 2 + u] = (double)ns;
            /* Compute output integer: gens[i] -> bit i. */
            int out_int = 0;
            for (int i = 0; i < n; ++i) {
                uint64_t poly = (uint64_t)gens->data[i];
                int bit = conv_output_bit(poly, K, (uint64_t)s, u);
                out_int |= (bit << (n - 1 - i));
            }
            outputs->data[s * 2 + u] = (double)out_int;
        }
    }
    matlab_struct *t = matlab_struct_new();
    matlab_struct_set_f64(t, "numInputSymbols",  15, 2.0);
    matlab_struct_set_f64(t, "numOutputSymbols", 16, (double)(1 << n));
    matlab_struct_set_f64(t, "numStates",         9, (double)S);
    matlab_struct_set_f64(t, "K",                 1, (double)K);
    matlab_struct_set_f64(t, "n",                 1, (double)n);
    matlab_struct_set_mat(t, "nextStates",       10, nextStates);
    matlab_struct_set_mat(t, "outputs",           7, outputs);
    return t;
}

/* matlab_struct_get_* are exported from matlab_runtime.cpp. */
extern double matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len);
extern matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name, int64_t len);

/* convenc(msg, trellis) — straight state-machine encoder.  `msg` is a
 * column of message bits; output is `n * length(msg)` bits. */
matlab_mat *matlab_comm_convenc(matlab_mat *msg, matlab_struct *trellis) {
    if (!msg || !trellis) return mat_alloc(0, 0);
    int n = (int)matlab_struct_get_f64(trellis, "n", 1);
    int64_t S = (int64_t)matlab_struct_get_f64(trellis, "numStates", 9);
    matlab_mat *outputs    = matlab_struct_get_mat(trellis, "outputs",      7);
    matlab_mat *nextStates = matlab_struct_get_mat(trellis, "nextStates", 10);
    if (!outputs || !nextStates || n < 1) return mat_alloc(0, 0);
    int64_t L = msg->rows * msg->cols;
    matlab_mat *out = mat_alloc(L * n, 1);
    int64_t state = 0;
    for (int64_t k = 0; k < L; ++k) {
        int u = ((int)msg->data[k]) & 1;
        int out_int = (int)outputs->data[state * 2 + u];
        for (int i = 0; i < n; ++i)
            out->data[k * n + i] = (double)((out_int >> (n - 1 - i)) & 1);
        state = (int64_t)nextStates->data[state * 2 + u];
        if (state < 0 || state >= S) state = 0;
    }
    return out;
}

/* vitdec(code, trellis, tblen, opmode, dectype) — hard-decision Viterbi.
 *   tblen   : traceback depth (typical 5K)
 *   opmode  : 0 trunc, 1 term (assume known final-state 0), 2 cont (defer)
 *   dectype : 0 unquant (== hard for {0,1} inputs), 1 hard
 *
 * Walks the trellis forward computing the cumulative Hamming distance
 * to each state, stores predecessor + input-bit decisions, then
 * tracebacks from the best terminal state.
 */
matlab_mat *matlab_comm_vitdec(matlab_mat *code, matlab_struct *trellis,
                                double tblen_d, double opmode_d, double dectype_d) {
    (void)dectype_d;     /* hard-decision only for the MVP slice */
    if (!code || !trellis) return mat_alloc(0, 0);
    int n = (int)matlab_struct_get_f64(trellis, "n", 1);
    int64_t S = (int64_t)matlab_struct_get_f64(trellis, "numStates", 9);
    matlab_mat *outputs    = matlab_struct_get_mat(trellis, "outputs",      7);
    matlab_mat *nextStates = matlab_struct_get_mat(trellis, "nextStates", 10);
    if (!outputs || !nextStates || n < 1 || S < 1) return mat_alloc(0, 0);
    int64_t total = code->rows * code->cols;
    int64_t T = total / n;
    if (T < 1) return mat_alloc(0, 0);
    int opmode = (int)opmode_d;
    (void)tblen_d;

    /* Build the reverse-edge table: incoming[s] gives the (prev_state,
     * input_bit) pairs that reach state s.  For rate 1/n binary
     * convolutional codes there are always exactly 2 incoming edges
     * per state — we materialise that pair table once. */
    std::vector<int64_t> in_prev(S * 2, -1);
    std::vector<int>     in_bit (S * 2, 0);
    std::vector<int>     in_out (S * 2, 0);
    std::vector<int>     n_in   (S, 0);
    for (int64_t ps = 0; ps < S; ++ps) {
        for (int u = 0; u <= 1; ++u) {
            int64_t ns = (int64_t)nextStates->data[ps * 2 + u];
            if (ns < 0 || ns >= S) continue;
            int slot = n_in[ns]++;
            if (slot >= 2) continue;
            in_prev[ns * 2 + slot] = ps;
            in_bit [ns * 2 + slot] = u;
            in_out [ns * 2 + slot] = (int)outputs->data[ps * 2 + u];
        }
    }

    const double INF = 1e18;
    std::vector<double> pm(S, INF);
    std::vector<double> pm_next(S, INF);
    std::vector<int8_t> bit_dec(T * S, 0);
    std::vector<int32_t> prev_dec(T * S, 0);
    pm[0] = 0.0;

    for (int64_t t = 0; t < T; ++t) {
        /* Decode the received n-tuple into an integer. */
        int rx_int = 0;
        for (int i = 0; i < n; ++i) {
            int b = ((int)code->data[t * n + i]) & 1;
            rx_int |= (b << (n - 1 - i));
        }
        for (int64_t s = 0; s < S; ++s) {
            double best = INF;
            int best_bit = 0;
            int64_t best_prev = 0;
            for (int slot = 0; slot < n_in[s] && slot < 2; ++slot) {
                int64_t ps = in_prev[s * 2 + slot];
                if (ps < 0) continue;
                int u  = in_bit[s * 2 + slot];
                int oi = in_out[s * 2 + slot];
                int diff = rx_int ^ oi;
                int hamming = 0;
                while (diff) { hamming += diff & 1; diff >>= 1; }
                double cand = pm[ps] + (double)hamming;
                if (cand < best) {
                    best = cand;
                    best_bit = u;
                    best_prev = ps;
                }
            }
            pm_next[s] = best;
            bit_dec [t * S + s] = (int8_t)best_bit;
            prev_dec[t * S + s] = (int32_t)best_prev;
        }
        std::swap(pm, pm_next);
        for (int64_t s = 0; s < S; ++s) pm_next[s] = INF;
    }

    /* Terminal state: opmode==1 (term) -> state 0; otherwise -> argmin pm. */
    int64_t end = 0;
    if (opmode != 1) {
        double best = pm[0];
        for (int64_t s = 1; s < S; ++s) {
            if (pm[s] < best) { best = pm[s]; end = s; }
        }
    }
    /* Traceback. */
    matlab_mat *msg = mat_alloc(T, 1);
    int64_t state = end;
    for (int64_t t = T - 1; t >= 0; --t) {
        msg->data[t] = (double)bit_dec[t * S + state];
        state = (int64_t)prev_dec[t * S + state];
    }
    return msg;
}

/* oct2dec(octal_int) - convert a MATLAB-style octal-encoded decimal
 * integer (e.g. 171 representing octal o171) to its decimal value
 * (121).  Bridge for poly2trellis users who copy generator polys
 * straight from textbook octal notation. */
double matlab_comm_oct2dec_s(double v) {
    int64_t x = (int64_t)v;
    int64_t out = 0;
    int64_t mult = 1;
    while (x > 0) {
        int64_t d = x % 10;
        out += d * mult;
        mult *= 8;
        x /= 10;
    }
    return (double)out;
}

/* ===== §5.3 Hamming codes (binary, n = 2^m - 1) ========================== */

/* hammgen(m): returns the [m × n] parity-check matrix H whose columns
 * are the binary expansions of 1..n.  Caller-side, the companion
 * generator matrix G is the systematic form derivable from H —
 * `hammingGen(m)` returns G separately (so we can fit a single matrix
 * per call into the dispatch). */
matlab_mat *matlab_comm_hammgen_parity(double md) {
    int m = (int)md;
    if (m < 2) m = 2;
    if (m > 12) m = 12;
    int n = (1 << m) - 1;
    matlab_mat *H = mat_alloc(m, n);
    /* Columns are the binary expansions of 1..n, MSB at row 0. */
    for (int c = 0; c < n; ++c) {
        int v = c + 1;
        for (int r = 0; r < m; ++r) {
            H->data[r * n + c] = (double)((v >> (m - 1 - r)) & 1);
        }
    }
    return H;
}

/* hammingEncode(msg, m): straightforward systematic Hamming.  msg is
 * a column of k = (2^m - m - 1) bits per code word; we encode in
 * blocks.  The systematic encoding places message bits at non-
 * power-of-two column positions; parity bits at positions 1, 2, 4,
 * 8, ..., 2^(m-1).
 *
 * For simplicity we exhibit a single-block (no length padding) form
 * that requires `length(msg) == k`.  Callers can batch outside. */
matlab_mat *matlab_comm_hamming_encode(matlab_mat *msg, double md) {
    int m = (int)md;
    if (m < 2) m = 2; if (m > 12) m = 12;
    int n = (1 << m) - 1;
    int k = n - m;
    if (!msg || msg->rows * msg->cols != k) return mat_alloc(0, 0);
    matlab_mat *code = mat_alloc(n, 1);
    int msg_idx = 0;
    /* Pre-place message bits. */
    for (int pos = 1; pos <= n; ++pos) {
        if ((pos & (pos - 1)) == 0) continue;     /* skip parity positions */
        code->data[pos - 1] = msg->data[msg_idx++];
    }
    /* Compute parity bits at positions 1, 2, 4, ..., 2^(m-1). */
    for (int p = 0; p < m; ++p) {
        int pos = 1 << p;
        int parity = 0;
        for (int j = 1; j <= n; ++j) {
            if (j == pos) continue;
            if ((j & pos) == 0) continue;
            parity ^= (int)code->data[j - 1] & 1;
        }
        code->data[pos - 1] = (double)parity;
    }
    return code;
}

/* hammingDecode(code, m): syndrome-based single-error correction.
 * Returns the k = n - m message bits (extracted from non-power-of-two
 * positions after correcting any single-bit error). */
matlab_mat *matlab_comm_hamming_decode(matlab_mat *code, double md) {
    int m = (int)md;
    if (m < 2) m = 2; if (m > 12) m = 12;
    int n = (1 << m) - 1;
    int k = n - m;
    if (!code || code->rows * code->cols != n) return mat_alloc(0, 0);
    /* Compute syndrome. */
    matlab_mat *work = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) work->data[i] = code->data[i];
    int syndrome = 0;
    for (int p = 0; p < m; ++p) {
        int pos = 1 << p;
        int parity = 0;
        for (int j = 1; j <= n; ++j) {
            if ((j & pos) == 0) continue;
            parity ^= (int)work->data[j - 1] & 1;
        }
        if (parity) syndrome |= pos;
    }
    /* Correct the bit at position `syndrome` if non-zero. */
    if (syndrome >= 1 && syndrome <= n) {
        work->data[syndrome - 1] = (double)(1 - (int)work->data[syndrome - 1]);
    }
    /* Extract message bits. */
    matlab_mat *msg = mat_alloc(k, 1);
    int msg_idx = 0;
    for (int pos = 1; pos <= n; ++pos) {
        if ((pos & (pos - 1)) == 0) continue;
        msg->data[msg_idx++] = work->data[pos - 1];
    }
    free(work->data); free(work);
    return msg;
}

/* ===== §5.5 Block interleavers ==========================================
 *
 * intrlv(data, perm) — reorder data per the permutation vector.  perm
 * is a length-N column of 1-based indices that says "row i of the
 * output is row perm(i) of the input".  deintrlv is the inverse.
 * Both data and perm are matlab_mat (real); the data type stays f64
 * so callers can use the same routine for bit / symbol streams. */
matlab_mat *matlab_comm_intrlv(matlab_mat *data, matlab_mat *perm) {
    if (!data || !perm) return mat_alloc(0, 0);
    int64_t N = data->rows * data->cols;
    int64_t Np = perm->rows * perm->cols;
    if (N != Np) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(data->rows, data->cols);
    for (int64_t i = 0; i < N; ++i) {
        int64_t idx = (int64_t)perm->data[i] - 1;
        if (idx < 0 || idx >= N) idx = 0;
        out->data[i] = data->data[idx];
    }
    return out;
}

matlab_mat *matlab_comm_deintrlv(matlab_mat *data, matlab_mat *perm) {
    if (!data || !perm) return mat_alloc(0, 0);
    int64_t N = data->rows * data->cols;
    int64_t Np = perm->rows * perm->cols;
    if (N != Np) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(data->rows, data->cols);
    for (int64_t i = 0; i < N; ++i) {
        int64_t idx = (int64_t)perm->data[i] - 1;
        if (idx < 0 || idx >= N) idx = 0;
        out->data[idx] = data->data[i];
    }
    return out;
}

}  /* extern "C" */
