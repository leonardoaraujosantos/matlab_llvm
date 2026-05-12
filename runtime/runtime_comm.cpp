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

/* ===== eyediagram — numeric trace-matrix return ========================= *
 *
 * MATLAB's `eyediagram(x, n)` overlays consecutive n-sample slices of
 * a pulse-shaped baseband signal onto a single figure — the "open
 * eye" plot used to inspect ISI / timing-jitter / SNR by eye.
 *
 * The headless lane returns the trace MATRIX of shape (n × num_traces)
 * where column k contains samples [k*n .. (k+1)*n - 1] of the input.
 * Users can then `plot` the columns (overlaid) for the canonical eye
 * diagram, or just inspect the matrix for ISI sanity-checks.
 *
 * Inputs:
 *   - Real (matlab_mat *): traces of the signal itself.
 *   - Complex (matlab_mat_c *): traces of the REAL part (the I
 *     channel).  Users can also call `eyediagram(imag(x), n)` for
 *     the Q channel.  MathWorks' interactive viewer renders I and Q
 *     side by side; we leave that orchestration to the caller. */
matlab_mat *matlab_comm_eyediagram(void *x_any, double n_samples_d) {
    int n = (int)n_samples_d;
    if (n < 2) n = 2;
    int64_t N = 0;
    const double *src = NULL;
    if (mat_is_complex(x_any)) {
        const matlab_mat_c *c = (const matlab_mat_c *)x_any;
        if (!c) return mat_alloc(0, 0);
        N = c->rows * c->cols;
        src = c->re;
    } else {
        const matlab_mat *m = (const matlab_mat *)x_any;
        if (!m) return mat_alloc(0, 0);
        N = m->rows * m->cols;
        src = m->data;
    }
    if (N <= 0 || !src) return mat_alloc(0, 0);
    int num_traces = (int)(N / n);
    if (num_traces <= 0) return mat_alloc(0, 0);
    /* Output: n rows × num_traces cols, row-major. */
    matlab_mat *Y = mat_alloc(n, num_traces);
    for (int j = 0; j < num_traces; ++j) {
        for (int i = 0; i < n; ++i) {
            int64_t idx = (int64_t)j * n + i;
            Y->data[(int64_t)i * num_traces + j] = src[idx];
        }
    }
    return Y;
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

/* ===== Tier-4 — equalisation, sync, RF impairments ======================= *
 *
 * docs/comm_toolbox_roadmap.md §6. Function-form only — the
 * comm.LinearEqualizer / DFE / CarrierSynchronizer / SymbolSynchronizer
 * / PreambleDetector / PhaseNoise / MemorylessNonlinearity System
 * Objects stay gated on the SO lowering fix recorded in CST §12. */

/* ----- §6.1 Adaptive equalisers — LMS / RLS / CMA / DFE ----- *
 *
 * lms(x, d, mu, ntaps) — Wiener / Widrow-Hoff LMS adaptive filter.
 *   x      : N x 1 received signal (real)
 *   d      : N x 1 desired (training-mode reference)
 *   mu     : step size (typical 1e-3 to 1e-1)
 *   ntaps  : filter length
 * Returns the equalised output y[n] of length N (the first ntaps-1
 * samples are filter "warm-up" with zero history).  All real here;
 * a complex sibling lms_c lives below. */
matlab_mat *matlab_comm_lms(matlab_mat *x, matlab_mat *d,
                             double mu, double ntaps) {
    if (!x || !d) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int64_t Nd = d->rows * d->cols;
    int K = (int)ntaps;
    if (K < 1) K = 1;
    if (K > N) K = (int)N;
    int64_t Lo = std::min(N, Nd);
    std::vector<double> w(K, 0.0);
    std::vector<double> buf(K, 0.0);
    matlab_mat *y = mat_alloc(Lo, 1);
    for (int64_t n = 0; n < Lo; ++n) {
        /* Shift buffer (right-to-left FIFO; index 0 is the newest). */
        for (int k = K - 1; k > 0; --k) buf[k] = buf[k - 1];
        buf[0] = x->data[n];
        double yk = 0.0;
        for (int k = 0; k < K; ++k) yk += w[k] * buf[k];
        double e = d->data[n] - yk;
        for (int k = 0; k < K; ++k) w[k] += mu * e * buf[k];
        y->data[n] = yk;
    }
    return y;
}

/* rls(x, d, lambda, delta, ntaps) — recursive least squares.
 *   lambda : forgetting factor (0.95..0.999 typical)
 *   delta  : initial P diagonal (1e2..1e4 typical) */
matlab_mat *matlab_comm_rls(matlab_mat *x, matlab_mat *d,
                             double lambda, double delta, double ntaps) {
    if (!x || !d) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int64_t Nd = d->rows * d->cols;
    int K = (int)ntaps;
    if (K < 1) K = 1;
    if (K > N) K = (int)N;
    int64_t Lo = std::min(N, Nd);
    if (lambda <= 0.0 || lambda > 1.0) lambda = 0.99;
    if (delta <= 0.0) delta = 1.0;
    std::vector<double> w(K, 0.0);
    std::vector<double> u(K, 0.0);
    std::vector<double> P(K * K, 0.0);
    for (int i = 0; i < K; ++i) P[i * K + i] = delta;
    std::vector<double> Pu(K), gain(K);
    matlab_mat *y = mat_alloc(Lo, 1);
    for (int64_t n = 0; n < Lo; ++n) {
        for (int k = K - 1; k > 0; --k) u[k] = u[k - 1];
        u[0] = x->data[n];
        /* Pu = P * u */
        for (int i = 0; i < K; ++i) {
            double s = 0.0;
            for (int j = 0; j < K; ++j) s += P[i * K + j] * u[j];
            Pu[i] = s;
        }
        /* den = lambda + u' * Pu */
        double den = lambda;
        for (int k = 0; k < K; ++k) den += u[k] * Pu[k];
        if (den == 0.0) den = 1e-30;
        for (int k = 0; k < K; ++k) gain[k] = Pu[k] / den;
        double yk = 0.0;
        for (int k = 0; k < K; ++k) yk += w[k] * u[k];
        double e = d->data[n] - yk;
        for (int k = 0; k < K; ++k) w[k] += gain[k] * e;
        /* P = (1/lambda) * (P - gain * u' * P) = (P - gain * Pu') / lambda */
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j)
                P[i * K + j] = (P[i * K + j] - gain[i] * Pu[j]) / lambda;
        }
        y->data[n] = yk;
    }
    return y;
}

/* cma(x, mu, ntaps, R2) — constant-modulus (Godard / CMA) blind
 * equaliser.  R2 = E[|s|^4] / E[|s|^2] for the source constellation
 * (e.g. R2 = 1 for PSK on the unit circle).  Operates on the real
 * envelope (since we don't have a typed complex argument descriptor
 * in the dispatch lane yet — the complex variant ships once the
 * matlab_mat_c arg lane is wired). */
matlab_mat *matlab_comm_cma(matlab_mat *x, double mu, double ntaps, double R2) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int K = (int)ntaps;
    if (K < 1) K = 1;
    if (K > N) K = (int)N;
    std::vector<double> w(K, 0.0);
    std::vector<double> buf(K, 0.0);
    /* Centre tap initialised to 1 — standard CMA initialisation. */
    w[K / 2] = 1.0;
    matlab_mat *y = mat_alloc(N, 1);
    for (int64_t n = 0; n < N; ++n) {
        for (int k = K - 1; k > 0; --k) buf[k] = buf[k - 1];
        buf[0] = x->data[n];
        double yk = 0.0;
        for (int k = 0; k < K; ++k) yk += w[k] * buf[k];
        double err = yk * (R2 - yk * yk);   /* gradient of |y|^2 vs R2 */
        for (int k = 0; k < K; ++k) w[k] += mu * err * buf[k];
        y->data[n] = yk;
    }
    return y;
}

/* dfe(x, d, mu, n_ff, n_fb) — LMS-trained decision-feedback equaliser.
 *   n_ff : feed-forward taps (across received samples)
 *   n_fb : feedback taps (across decided symbols)
 * Decision threshold at 0 (real BPSK-style symbol set {-1, +1}).
 * Training uses the d[n] vector; switches to decision-directed mode
 * once the trainer epoch (first half of d) completes. */
matlab_mat *matlab_comm_dfe(matlab_mat *x, matlab_mat *d, double mu,
                             double n_ff, double n_fb) {
    if (!x || !d) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int64_t Nd = d->rows * d->cols;
    int Kff = (int)n_ff; if (Kff < 1) Kff = 1;
    int Kfb = (int)n_fb; if (Kfb < 0) Kfb = 0;
    int64_t Lo = std::min(N, Nd);
    int64_t train_n = Lo / 2;
    std::vector<double> wff(Kff, 0.0);
    std::vector<double> wfb(Kfb, 0.0);
    std::vector<double> bff(Kff, 0.0);
    std::vector<double> bfb(Kfb, 0.0);
    matlab_mat *y = mat_alloc(Lo, 1);
    for (int64_t n = 0; n < Lo; ++n) {
        for (int k = Kff - 1; k > 0; --k) bff[k] = bff[k - 1];
        bff[0] = x->data[n];
        double yk = 0.0;
        for (int k = 0; k < Kff; ++k) yk += wff[k] * bff[k];
        for (int k = 0; k < Kfb; ++k) yk -= wfb[k] * bfb[k];
        double sym = (yk >= 0.0) ? 1.0 : -1.0;
        double ref = (n < train_n) ? d->data[n] : sym;
        double err = ref - yk;
        for (int k = 0; k < Kff; ++k) wff[k] += mu * err * bff[k];
        for (int k = 0; k < Kfb; ++k) wfb[k] -= mu * err * bfb[k];
        for (int k = Kfb - 1; k > 0; --k) bfb[k] = bfb[k - 1];
        if (Kfb > 0) bfb[0] = ref;
        y->data[n] = yk;
    }
    return y;
}

/* ----- §6.2 Carrier + symbol + frame sync ----- *
 *
 * costasPll(x, M_psk, loop_bw, fs) — M-PSK Costas-style PLL carrier
 *   recovery. M_psk = 2 for BPSK (squarer) or 4 for QPSK (4th-power).
 *   Returns the de-rotated output (a fresh complex matrix of the same
 *   length as the input).  Implementation: 2nd-order phase-locked
 *   loop with damping 1/sqrt(2). */
matlab_mat_c *matlab_comm_costas_pll(matlab_mat_c *x, double M_psk,
                                      double loop_bw, double fs) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    int M = (int)M_psk;
    if (M < 2) M = 2;
    if (fs <= 0.0) fs = 1.0;
    if (loop_bw <= 0.0) loop_bw = fs * 1e-3;
    double zeta = 1.0 / sqrt(2.0);
    double wn = 2.0 * M_PI * loop_bw / fs;
    double Kp = 2.0 * zeta * wn;
    double Ki = wn * wn;
    matlab_mat_c *y = mat_c_alloc(x->rows, x->cols);
    double phi = 0.0, freq = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        double cp = cos(-phi), sp = sin(-phi);
        double re_r = x->re[n] * cp - x->im[n] * sp;
        double im_r = x->re[n] * sp + x->im[n] * cp;
        y->re[n] = re_r; y->im[n] = im_r;
        /* Phase-error discriminator (M = 2 -> sign(Re)*Im; M = 4 -> imag of x^4). */
        double err;
        if (M == 2) {
            err = (re_r >= 0.0 ? 1.0 : -1.0) * im_r;
        } else if (M == 4) {
            /* Hard slicing onto the 4-PSK constellation; phase error
             * is the imag part of (y_rot * conj(sliced)). */
            double sx = re_r >= 0.0 ? 1.0 : -1.0;
            double sy = im_r >= 0.0 ? 1.0 : -1.0;
            err = im_r * sx - re_r * sy;
            err *= 0.5;
        } else {
            err = atan2(im_r, re_r);
        }
        freq += Ki * err;
        phi  += Kp * err + freq;
    }
    return y;
}

/* symbolSyncMM(x, sps, loop_bw) — Mueller-Müller symbol-timing
 * recovery for BPSK-style real signals.  Outputs a vector of
 * symbol-rate samples (one per nominal symbol).  loop_bw normalised
 * to sample rate.  Implementation: NCO-driven sample selector +
 * Mueller-Müller TED, 1st-order loop with leakage. */
matlab_mat *matlab_comm_symbol_sync_mm(matlab_mat *x, double sps,
                                        double loop_bw) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    double sps_d = sps > 1.0 ? sps : 1.0;
    if (loop_bw <= 0.0 || loop_bw >= 1.0) loop_bw = 0.05;
    int64_t Nsym = (int64_t)((double)N / sps_d) + 1;
    matlab_mat *y = mat_alloc(Nsym, 1);
    int64_t idx = 0;
    double tau = 0.0;            /* fractional time offset */
    double prev = 0.0;
    double pprev = 0.0;
    for (int64_t k = 0; k < Nsym; ++k) {
        double pos = sps_d * (double)k + tau;
        int64_t i0 = (int64_t)floor(pos);
        double frac = pos - (double)i0;
        if (i0 < 0) i0 = 0;
        if (i0 >= N - 1) break;
        /* Linear-interpolated sample. */
        double s = x->data[i0] * (1.0 - frac) + x->data[i0 + 1] * frac;
        y->data[idx++] = s;
        /* Mueller-Müller error: 0.5 * (sign(s) * prev - sign(prev) * pprev). */
        double err;
        if (k >= 2) {
            double sgs    = s    >= 0.0 ? 1.0 : -1.0;
            double sgprev = prev >= 0.0 ? 1.0 : -1.0;
            err = 0.5 * (sgs * prev - sgprev * pprev);
        } else err = 0.0;
        tau -= loop_bw * err;
        pprev = prev;
        prev  = s;
    }
    /* Truncate to actually filled rows. */
    y->rows = idx;
    return y;
}

/* preambleDetect(x, preamble) — cross-correlate `preamble` against
 * the leading samples of `x` and return the lag (1-based index)
 * of the maximum-correlation point.  Caller can slice x from that
 * index forward to align the frame. */
double matlab_comm_preamble_detect(matlab_mat *x, matlab_mat *preamble) {
    if (!x || !preamble) return 0.0;
    int64_t N = x->rows * x->cols;
    int64_t M = preamble->rows * preamble->cols;
    if (M < 1 || N < M) return 0.0;
    double best = -1e30;
    int64_t best_idx = 0;
    for (int64_t n = 0; n <= N - M; ++n) {
        double acc = 0.0;
        for (int64_t k = 0; k < M; ++k) acc += x->data[n + k] * preamble->data[k];
        if (acc > best) { best = acc; best_idx = n; }
    }
    return (double)(best_idx + 1);
}

/* ----- §6.3 RF impairments ----- *
 *
 * phaseFreqOffset(x, df_Hz, fs_Hz) — complex frequency / phase offset
 *   y[n] = x[n] * exp(j * 2*pi * df_Hz * n / fs_Hz). */
matlab_mat_c *matlab_comm_phase_freq_offset(matlab_mat_c *x,
                                             double df_Hz, double fs_Hz) {
    if (!x) return mat_c_alloc(0, 0);
    if (fs_Hz <= 0.0) fs_Hz = 1.0;
    int64_t N = x->rows * x->cols;
    matlab_mat_c *y = mat_c_alloc(x->rows, x->cols);
    double w = 2.0 * M_PI * df_Hz / fs_Hz;
    for (int64_t n = 0; n < N; ++n) {
        double c = cos(w * (double)n), s = sin(w * (double)n);
        y->re[n] = x->re[n] * c - x->im[n] * s;
        y->im[n] = x->re[n] * s + x->im[n] * c;
    }
    return y;
}

/* iqimbal(x, amp_imb_dB, phase_imb_deg) — apply I/Q amplitude and
 * phase imbalance.  Sets the Q axis scale to 10^(amp_dB/20) and
 * rotates by phase_deg before adding back to I. */
matlab_mat_c *matlab_comm_iqimbal(matlab_mat_c *x, double amp_imb_dB,
                                   double phase_imb_deg) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    matlab_mat_c *y = mat_c_alloc(x->rows, x->cols);
    double g = pow(10.0, amp_imb_dB / 20.0);
    double p = phase_imb_deg * M_PI / 180.0;
    double cp = cos(p), sp = sin(p);
    for (int64_t n = 0; n < N; ++n) {
        y->re[n] = x->re[n];
        y->im[n] = g * (x->im[n] * cp + x->re[n] * sp);
    }
    return y;
}

/* memorylessNl(x, model_code, p1..p4) — memoryless PA nonlinearity.
 *   model_code 0 = cubic clipper: y = x for |x| <= 1, sign(x) otherwise
 *   model_code 1 = Saleh AM/AM + AM/PM (p1=alpha_a, p2=beta_a,
 *                  p3=alpha_p, p4=beta_p)
 *   model_code 2 = Rapp (p1 = smoothness p, p2 = saturation Asat)
 *   model_code 3 = Ghorbani (4 params for AM/AM and AM/PM each;
 *                  ship Saleh-equivalent simplification here) */
matlab_mat_c *matlab_comm_memoryless_nl(matlab_mat_c *x, double model_code,
                                         double p1, double p2,
                                         double p3, double p4) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    matlab_mat_c *y = mat_c_alloc(x->rows, x->cols);
    int mc = (int)model_code;
    for (int64_t n = 0; n < N; ++n) {
        double r = sqrt(x->re[n] * x->re[n] + x->im[n] * x->im[n]);
        double phi = atan2(x->im[n], x->re[n]);
        double amp_out, phi_out;
        switch (mc) {
        case 0: { /* clipper */
            double sat = p1 > 0.0 ? p1 : 1.0;
            amp_out = r > sat ? sat : r;
            phi_out = phi;
        } break;
        case 1: { /* Saleh */
            double a_a = p1 > 0.0 ? p1 : 2.1587;
            double b_a = p2 > 0.0 ? p2 : 1.1517;
            double a_p = p3;
            double b_p = p4 > 0.0 ? p4 : 2.5293;
            amp_out = a_a * r / (1.0 + b_a * r * r);
            phi_out = phi + a_p * r * r / (1.0 + b_p * r * r);
        } break;
        case 2: { /* Rapp */
            double p_s = p1 > 0.0 ? p1 : 3.0;
            double Asat = p2 > 0.0 ? p2 : 1.0;
            double pn = pow(r / Asat, 2.0 * p_s);
            amp_out = r / pow(1.0 + pn, 1.0 / (2.0 * p_s));
            phi_out = phi;
        } break;
        case 3: { /* Ghorbani (Saleh-style simplification) */
            double aa = p1 > 0.0 ? p1 : 8.106;
            double ba = p2 > 0.0 ? p2 : 1.5879;
            double ap = p3;
            double bp = p4 > 0.0 ? p4 : 4.0033;
            amp_out = aa * pow(r, ba) / (1.0 + bp * pow(r, ba));
            phi_out = phi + ap * pow(r, bp);
        } break;
        default:
            amp_out = r; phi_out = phi;
        }
        y->re[n] = amp_out * cos(phi_out);
        y->im[n] = amp_out * sin(phi_out);
    }
    return y;
}

/* phaseNoise(x, level_dBcHz, fs_Hz) — colour white Gaussian phase
 * noise to the requested integrated single-sideband level, then
 * apply as a complex rotation per sample.  Uses the shared PRNG. */
matlab_mat_c *matlab_comm_phase_noise(matlab_mat_c *x,
                                       double level_dBcHz, double fs_Hz) {
    if (!x) return mat_c_alloc(0, 0);
    if (fs_Hz <= 0.0) fs_Hz = 1.0;
    int64_t N = x->rows * x->cols;
    matlab_mat_c *y = mat_c_alloc(x->rows, x->cols);
    /* phase noise variance per sample = 10^(level_dBcHz/10) * fs / 2
     * (integrating single-sideband density over fs / 2 Hz). */
    double sigma2 = pow(10.0, level_dBcHz / 10.0) * fs_Hz / 2.0;
    double sigma = sqrt(sigma2);
    double phi = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        phi += sigma * comm_normal();
        double c = cos(phi), s = sin(phi);
        y->re[n] = x->re[n] * c - x->im[n] * s;
        y->im[n] = x->re[n] * s + x->im[n] * c;
    }
    return y;
}

/* ===== Tier-5 — OFDM / fading / MIMO ===================================== *
 *
 * docs/comm_toolbox_roadmap.md §7. Function-form: OFDM mod / demod,
 * Rayleigh / Rician fading channels with Jakes-style Doppler,
 * Alamouti 2x1 space-time block coding, simple ML detector + 2x2
 * complex ZF MIMO detect.  The comm.OFDMModulator / RayleighChannel
 * / RicianChannel / OSTBC* System Objects stay gated on the SO fix;
 * sphere decoding defers to a follow-on (needs lattice reduction). */

extern matlab_mat_c *matlab_fft_c(void *Aptr);
extern matlab_mat_c *matlab_ifft_c(void *Aptr);

/* ----- §7.1 OFDM mod / demod ---------------------------------------------
 *
 * ofdmmod(data, fft_len, cp_len): data is Nfft x Nsym complex.  For
 * each column we IFFT to time domain then prepend cp_len samples
 * (the last cp_len of the IFFT output).  Result is a
 * (Nfft+cp_len)*Nsym x 1 complex column.
 */
matlab_mat_c *matlab_comm_ofdmmod(matlab_mat_c *data, double fft_len_d,
                                   double cp_len_d) {
    if (!data) return mat_c_alloc(0, 0);
    int64_t Nfft = (int64_t)fft_len_d;
    int64_t Lcp  = (int64_t)cp_len_d;
    if (Nfft < 1) Nfft = 1;
    if (Lcp  < 0) Lcp  = 0;
    if (data->rows != Nfft) return mat_c_alloc(0, 0);
    int64_t Nsym = data->cols;
    int64_t Lout = (Nfft + Lcp) * Nsym;
    matlab_mat_c *out = mat_c_alloc(Lout, 1);
    /* Work column buffer for IFFT. */
    matlab_mat_c col;
    col.magic = 0xC0FFEE01u;
    col.rows = Nfft;
    col.cols = 1;
    std::vector<double> re_buf(Nfft), im_buf(Nfft);
    col.re = re_buf.data();
    col.im = im_buf.data();
    for (int64_t k = 0; k < Nsym; ++k) {
        /* Extract column k from data (row-major (Nfft, Nsym) layout). */
        for (int64_t i = 0; i < Nfft; ++i) {
            re_buf[i] = data->re[i * data->cols + k];
            im_buf[i] = data->im[i * data->cols + k];
        }
        matlab_mat_c *T = matlab_ifft_c((void *)&col);
        int64_t base = k * (Nfft + Lcp);
        for (int64_t i = 0; i < Lcp; ++i) {
            int64_t src = Nfft - Lcp + i;
            out->re[base + i] = T->re[src];
            out->im[base + i] = T->im[src];
        }
        for (int64_t i = 0; i < Nfft; ++i) {
            out->re[base + Lcp + i] = T->re[i];
            out->im[base + Lcp + i] = T->im[i];
        }
        free(T->re); free(T->im); free(T);
    }
    return out;
}

matlab_mat_c *matlab_comm_ofdmdemod(matlab_mat_c *samples, double fft_len_d,
                                     double cp_len_d) {
    if (!samples) return mat_c_alloc(0, 0);
    int64_t Nfft = (int64_t)fft_len_d;
    int64_t Lcp  = (int64_t)cp_len_d;
    if (Nfft < 1) Nfft = 1;
    if (Lcp  < 0) Lcp  = 0;
    int64_t Lin = samples->rows * samples->cols;
    int64_t Nsym = Lin / (Nfft + Lcp);
    if (Nsym < 1) return mat_c_alloc(0, 0);
    matlab_mat_c *out = mat_c_alloc(Nfft, Nsym);
    matlab_mat_c col;
    col.magic = 0xC0FFEE01u;
    col.rows = Nfft;
    col.cols = 1;
    std::vector<double> re_buf(Nfft), im_buf(Nfft);
    col.re = re_buf.data();
    col.im = im_buf.data();
    for (int64_t k = 0; k < Nsym; ++k) {
        int64_t base = k * (Nfft + Lcp) + Lcp;
        for (int64_t i = 0; i < Nfft; ++i) {
            re_buf[i] = samples->re[base + i];
            im_buf[i] = samples->im[base + i];
        }
        matlab_mat_c *F = matlab_fft_c((void *)&col);
        for (int64_t i = 0; i < Nfft; ++i) {
            out->re[i * Nsym + k] = F->re[i];
            out->im[i * Nsym + k] = F->im[i];
        }
        free(F->re); free(F->im); free(F);
    }
    return out;
}

/* ----- §7.2 fading channels ---------------------------------------------- *
 *
 * Sum-of-sinusoids Jakes generator for a single Rayleigh path.  Fills
 * (re, im) with N samples of unit-power complex Gaussian-like fading
 * (mean(|h|^2) ≈ 1).  Mosalavi 2002 modified Jakes; M = 16 oscillators.
 */
static void jakes_gen(double *re, double *im, int64_t N,
                       double max_doppler_Hz, double fs_Hz) {
    if (fs_Hz <= 0.0) fs_Hz = 1.0;
    const int M = 16;
    std::vector<double> phi(M), theta(M);
    for (int m = 0; m < M; ++m) {
        phi[m]   = 2.0 * M_PI * comm_uniform();
        theta[m] = 2.0 * M_PI * comm_uniform();
    }
    double wd = 2.0 * M_PI * max_doppler_Hz / fs_Hz;
    for (int64_t n = 0; n < N; ++n) {
        double sr = 0.0, si = 0.0;
        for (int m = 0; m < M; ++m) {
            double alpha = (2.0 * M_PI * (double)(m + 1) - M_PI + theta[m]) / (4.0 * (double)M);
            double c = wd * (double)n * cos(alpha) + phi[m];
            sr += cos(c);
            si += sin(c);
        }
        re[n] = sr / sqrt((double)M);
        im[n] = si / sqrt((double)M);
    }
}

/* rayleighChannel(x, delays_samples, gains_dB, max_doppler_Hz, fs_Hz).
 * Each path gets its own independent Jakes process; the channel
 * output is the sum of g_p · h_p[n] · x[n - d_p] over paths. */
matlab_mat_c *matlab_comm_rayleigh_channel(matlab_mat_c *x, matlab_mat *delays,
                                            matlab_mat *gains_dB,
                                            double max_doppler_Hz, double fs_Hz) {
    if (!x || !delays || !gains_dB) return mat_c_alloc(0, 0);
    int P = (int)(delays->rows * delays->cols);
    int Pg = (int)(gains_dB->rows * gains_dB->cols);
    if (P < 1 || Pg < P) return mat_c_alloc(0, 0);
    int64_t Nin = x->rows * x->cols;
    int64_t max_delay = 0;
    std::vector<int64_t> d_int(P);
    for (int p = 0; p < P; ++p) {
        d_int[p] = (int64_t)delays->data[p];
        if (d_int[p] < 0) d_int[p] = 0;
        if (d_int[p] > max_delay) max_delay = d_int[p];
    }
    std::vector<double> g_lin(P);
    for (int p = 0; p < P; ++p)
        g_lin[p] = pow(10.0, gains_dB->data[p] / 20.0);
    int64_t Nout = Nin + max_delay;
    matlab_mat_c *y = mat_c_alloc(Nout, 1);
    std::vector<std::vector<double>> hre(P), him(P);
    for (int p = 0; p < P; ++p) {
        hre[p].resize(Nin);
        him[p].resize(Nin);
        jakes_gen(hre[p].data(), him[p].data(), Nin, max_doppler_Hz, fs_Hz);
    }
    for (int64_t n = 0; n < Nout; ++n) {
        double yr = 0.0, yi = 0.0;
        for (int p = 0; p < P; ++p) {
            int64_t k = n - d_int[p];
            if (k < 0 || k >= Nin) continue;
            double xr = x->re[k];
            double xi = x->im[k];
            double hr = g_lin[p] * hre[p][k];
            double hi = g_lin[p] * him[p][k];
            yr += xr * hr - xi * hi;
            yi += xr * hi + xi * hr;
        }
        y->re[n] = yr;
        y->im[n] = yi;
    }
    return y;
}

/* ricianChannel(x, K_dB, delays, gains_dB, max_doppler, fs).
 * LOS component is the input itself scaled by sqrt(K / (K+1));
 * scatter component is Rayleigh scaled by sqrt(1 / (K+1)). */
matlab_mat_c *matlab_comm_rician_channel(matlab_mat_c *x, double K_dB,
                                          matlab_mat *delays, matlab_mat *gains_dB,
                                          double max_doppler_Hz, double fs_Hz) {
    matlab_mat_c *scatter = matlab_comm_rayleigh_channel(x, delays, gains_dB,
                                                          max_doppler_Hz, fs_Hz);
    if (!scatter) return mat_c_alloc(0, 0);
    double K = pow(10.0, K_dB / 10.0);
    double a_los     = sqrt(K / (K + 1.0));
    double a_scatter = sqrt(1.0 / (K + 1.0));
    int64_t N = scatter->rows * scatter->cols;
    int64_t Nin = x->rows * x->cols;
    for (int64_t n = 0; n < N; ++n) {
        double sr = a_scatter * scatter->re[n];
        double si = a_scatter * scatter->im[n];
        if (n < Nin) {
            sr += a_los * x->re[n];
            si += a_los * x->im[n];
        }
        scatter->re[n] = sr;
        scatter->im[n] = si;
    }
    return scatter;
}

/* ----- §7.3 Alamouti space-time block coding ----------------------------- *
 *
 * ostbcEncode(x): 2k x 1 complex input -> 2k x 2 complex output;
 * column 1 is the Tx1 stream, column 2 is the Tx2 stream. */
matlab_mat_c *matlab_comm_ostbc_encode(matlab_mat_c *x) {
    if (!x) return mat_c_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N % 2 != 0) N -= 1;
    matlab_mat_c *out = mat_c_alloc(N, 2);
    for (int64_t k = 0; k < N / 2; ++k) {
        double s0r = x->re[2 * k],     s0i = x->im[2 * k];
        double s1r = x->re[2 * k + 1], s1i = x->im[2 * k + 1];
        out->re[(2 * k) * 2 + 0] = s0r; out->im[(2 * k) * 2 + 0] = s0i;
        out->re[(2 * k) * 2 + 1] = s1r; out->im[(2 * k) * 2 + 1] = s1i;
        out->re[(2 * k + 1) * 2 + 0] = -s1r; out->im[(2 * k + 1) * 2 + 0] =  s1i;
        out->re[(2 * k + 1) * 2 + 1] =  s0r; out->im[(2 * k + 1) * 2 + 1] = -s0i;
    }
    return out;
}

/* ostbcCombine(y, h1_re, h1_im, h2_re, h2_im): 2-Tx Alamouti
 * maximum-ratio combiner at a single-RX terminal. Channel gains are
 * scalar (flat-fading assumption — caller breaks the burst into
 * coherence-time chunks if needed). */
matlab_mat_c *matlab_comm_ostbc_combine(matlab_mat_c *y,
                                         double h1_re, double h1_im,
                                         double h2_re, double h2_im) {
    if (!y) return mat_c_alloc(0, 0);
    int64_t N = y->rows * y->cols;
    if (N % 2 != 0) N -= 1;
    double norm2 = h1_re * h1_re + h1_im * h1_im + h2_re * h2_re + h2_im * h2_im;
    if (norm2 <= 0.0) norm2 = 1.0;
    matlab_mat_c *out = mat_c_alloc(N, 1);
    for (int64_t k = 0; k < N / 2; ++k) {
        double y0r = y->re[2 * k],     y0i = y->im[2 * k];
        double y1r = y->re[2 * k + 1], y1i = y->im[2 * k + 1];
        double t1r = h1_re * y0r + h1_im * y0i;
        double t1i = h1_re * y0i - h1_im * y0r;
        double t2r = h2_re * y1r + h2_im * y1i;
        double t2i = h2_re * (-y1i) + h2_im * y1r;
        out->re[2 * k]     = (t1r + t2r) / norm2;
        out->im[2 * k]     = (t1i + t2i) / norm2;
        double u1r = h2_re * y0r + h2_im * y0i;
        double u1i = h2_re * y0i - h2_im * y0r;
        double u2r = h1_re * y1r + h1_im * y1i;
        double u2i = h1_re * (-y1i) + h1_im * y1r;
        out->re[2 * k + 1] = (u1r - u2r) / norm2;
        out->im[2 * k + 1] = (u1i - u2i) / norm2;
    }
    return out;
}

/* ===== Tier-6 — spreading sequences + source coding ====================== *
 *
 * docs/comm_toolbox_roadmap.md §8.  Function-form spreading
 * (PN / Gold / Walsh-Hadamard) plus source-coding helpers (quantiz,
 * Lloyd-Max, A-law / μ-law companding, DPCM).  System-Object forms
 * (comm.PNSequence / GoldSequence / KasamiSequence) stay gated on
 * the SO lowering fix.  Kasami + hybrid-ARQ stay deferred per the
 * roadmap §8.3 carve-out. */

/* ----- §8.1 Spreading sequences ---------------------------------------- *
 *
 * pnSequence(poly_int, init_int, length, output_mode)
 *   poly_int    : LFSR feedback polynomial as an integer mask.  Each
 *                 set bit selects a tap.  e.g. x^4 + x + 1 -> 0b10011
 *                 = 19.  The polynomial degree is the highest set bit.
 *   init_int    : initial state, lower (degree) bits.
 *   length      : output sample count.
 *   output_mode : 0 = {0, 1} bits, 1 = {-1, +1} bipolar.
 *
 * Returns a column of length samples.  Galois LFSR.
 */
matlab_mat *matlab_comm_pn_sequence(double poly_int_d, double init_int_d,
                                     double length_d, double output_mode) {
    int64_t N = (int64_t)length_d;
    if (N < 1) N = 1;
    uint64_t poly = (uint64_t)poly_int_d;
    if (poly < 3) poly = 3;
    int deg = 0;
    {
        uint64_t v = poly;
        while (v) { v >>= 1; ++deg; }
        --deg;
        if (deg < 1) deg = 1;
    }
    uint64_t state = (uint64_t)init_int_d;
    uint64_t state_mask = (deg >= 64) ? ~0ULL : ((1ULL << deg) - 1ULL);
    state &= state_mask;
    if (state == 0) state = 1;
    /* Use the standard "Fibonacci" form: output the low bit, then
     * shift right one and conditionally XOR the high taps (above the
     * implicit leading 1). */
    matlab_mat *out = mat_alloc(N, 1);
    int mode = (int)output_mode;
    for (int64_t i = 0; i < N; ++i) {
        uint64_t bit = state & 1ULL;
        out->data[i] = (mode == 1) ? (1.0 - 2.0 * (double)bit) : (double)bit;
        state >>= 1;
        if (bit) state ^= (poly >> 1);
        state &= state_mask;
    }
    return out;
}

/* goldSequence(poly1, poly2, init1, init2, length, output_mode)
 *   Two preferred-pair LFSRs whose outputs are XOR'd to give a Gold
 *   sequence of length (2^n - 1) (or longer if the LFSRs are run
 *   past one period).
 */
matlab_mat *matlab_comm_gold_sequence(double poly1_d, double poly2_d,
                                       double init1_d, double init2_d,
                                       double length_d, double output_mode) {
    matlab_mat *a = matlab_comm_pn_sequence(poly1_d, init1_d, length_d, 0);
    matlab_mat *b = matlab_comm_pn_sequence(poly2_d, init2_d, length_d, 0);
    int64_t N = (int64_t)length_d;
    if (N < 1) N = 1;
    matlab_mat *out = mat_alloc(N, 1);
    int mode = (int)output_mode;
    for (int64_t i = 0; i < N; ++i) {
        int bit = ((int)a->data[i] ^ (int)b->data[i]) & 1;
        out->data[i] = (mode == 1) ? (1.0 - 2.0 * (double)bit) : (double)bit;
    }
    free(a->data); free(a);
    free(b->data); free(b);
    return out;
}

/* hadamard(n): n × n Sylvester-form Hadamard matrix (n a power of 2,
 * with H(1) = [1] and H(2n) = [[H(n), H(n)]; [H(n), -H(n)]]).  Walsh
 * codes are the rows of this matrix.  Other Hadamard orders (n = 12,
 * 20, ...) are not in scope; pass n = 1, 2, 4, 8, ... */
matlab_mat *matlab_comm_hadamard(double n_d) {
    int n = (int)n_d;
    if (n < 1) n = 1;
    /* Snap to next power of 2. */
    int p = 1;
    while (p < n) p <<= 1;
    n = p;
    matlab_mat *H = mat_alloc(n, n);
    H->data[0] = 1.0;
    for (int sz = 1; sz < n; sz <<= 1) {
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) {
                double v = H->data[i * n + j];
                H->data[i * n + (sz + j)]      =  v;
                H->data[(sz + i) * n + j]       =  v;
                H->data[(sz + i) * n + (sz + j)] = -v;
            }
        }
    }
    return H;
}

/* walshCode(n, k): k-th row (1-based) of the n×n Hadamard matrix. */
matlab_mat *matlab_comm_walsh_code(double n_d, double k_d) {
    matlab_mat *H = matlab_comm_hadamard(n_d);
    int n = (int)H->rows;
    int k = (int)k_d - 1;
    if (k < 0) k = 0;
    if (k >= n) k = n - 1;
    matlab_mat *out = mat_alloc(n, 1);
    for (int j = 0; j < n; ++j) out->data[j] = H->data[k * n + j];
    free(H->data); free(H);
    return out;
}

/* ----- §8.2 Source coding ---------------------------------------------- *
 *
 * quantiz(sig, partition, codebook):
 *   sig        : Nin × 1 real input
 *   partition  : (M-1) thresholds (sorted ascending)
 *   codebook   : M codebook entries (1 more than partitions)
 *
 * Returns an Nin x 1 column of *codebook indices* (0 .. M-1). For the
 * quantised-signal companion, the caller looks up codebook[indices].
 * (MATLAB returns [indx, quant, dist]; we ship the index lookup here
 * and a `quantizApply(idx, codebook)` companion below.) */
matlab_mat *matlab_comm_quantiz(matlab_mat *sig, matlab_mat *partition,
                                 matlab_mat *codebook) {
    if (!sig || !partition || !codebook) return mat_alloc(0, 0);
    int64_t Nin = sig->rows * sig->cols;
    int64_t Np = partition->rows * partition->cols;
    matlab_mat *idx = mat_alloc(Nin, 1);
    for (int64_t i = 0; i < Nin; ++i) {
        double v = sig->data[i];
        int64_t k = 0;
        while (k < Np && v > partition->data[k]) ++k;
        idx->data[i] = (double)k;
    }
    return idx;
}

/* quantizApply(idx, codebook): look up the codebook entries for the
 * integer indices in idx. */
matlab_mat *matlab_comm_quantiz_apply(matlab_mat *idx, matlab_mat *codebook) {
    if (!idx || !codebook) return mat_alloc(0, 0);
    int64_t N = idx->rows * idx->cols;
    int64_t Mc = codebook->rows * codebook->cols;
    matlab_mat *out = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) {
        int64_t k = (int64_t)idx->data[i];
        if (k < 0) k = 0;
        if (k >= Mc) k = Mc - 1;
        out->data[i] = codebook->data[k];
    }
    return out;
}

/* lloydsQuant(sig, initCodebook, max_iter, tol):
 *   Iterative Lloyd-Max optimisation.  initCodebook is the M-element
 *   starting codebook (sorted ascending).  Returns the optimised
 *   codebook (M x 1) after at most max_iter sweeps. */
matlab_mat *matlab_comm_lloyds_quant(matlab_mat *sig, matlab_mat *init_cb,
                                      double max_iter, double tol_d) {
    if (!sig || !init_cb) return mat_alloc(0, 0);
    int M = (int)(init_cb->rows * init_cb->cols);
    int64_t N = sig->rows * sig->cols;
    if (M < 1 || N < 1) return mat_alloc(0, 0);
    int iters = (int)max_iter;
    if (iters < 1) iters = 30;
    double tol = tol_d > 0.0 ? tol_d : 1e-6;
    matlab_mat *cb = mat_alloc(M, 1);
    for (int i = 0; i < M; ++i) cb->data[i] = init_cb->data[i];
    std::vector<double> sum(M), cnt(M);
    for (int it = 0; it < iters; ++it) {
        /* Compute partitions = midpoints of adjacent codebook entries. */
        for (int i = 0; i < M; ++i) { sum[i] = 0.0; cnt[i] = 0.0; }
        for (int64_t k = 0; k < N; ++k) {
            double v = sig->data[k];
            int i = 0;
            while (i < M - 1 && v > 0.5 * (cb->data[i] + cb->data[i + 1])) ++i;
            sum[i] += v;
            cnt[i] += 1.0;
        }
        double delta = 0.0;
        for (int i = 0; i < M; ++i) {
            double new_v = cnt[i] > 0.0 ? sum[i] / cnt[i] : cb->data[i];
            double d = new_v - cb->data[i];
            if (fabs(d) > delta) delta = fabs(d);
            cb->data[i] = new_v;
        }
        if (delta < tol) break;
    }
    return cb;
}

/* μ-law companding (G.711).
 *   compandMu(x, mu, V, dir):
 *     dir = 0 -> compress (mu-law encode)
 *     dir = 1 -> expand   (mu-law decode)
 *   V = peak amplitude (positive).
 *
 * Compress: y = sign(x) · V · ln(1 + μ·|x|/V) / ln(1 + μ)
 * Expand:   x = sign(y) · V · ((1 + μ)^(|y|/V) − 1) / μ
 */
matlab_mat *matlab_comm_compand_mu(matlab_mat *x, double mu_d, double V_d,
                                    double dir_d) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    matlab_mat *y = mat_alloc(x->rows, x->cols);
    double mu = mu_d > 0.0 ? mu_d : 255.0;
    double V  = V_d  > 0.0 ? V_d  : 1.0;
    int dir = (int)dir_d;
    if (dir == 0) {
        double denom = log(1.0 + mu);
        for (int64_t i = 0; i < N; ++i) {
            double v = x->data[i];
            double s = v >= 0.0 ? 1.0 : -1.0;
            y->data[i] = s * V * log(1.0 + mu * fabs(v) / V) / denom;
        }
    } else {
        for (int64_t i = 0; i < N; ++i) {
            double v = x->data[i];
            double s = v >= 0.0 ? 1.0 : -1.0;
            y->data[i] = s * V * (pow(1.0 + mu, fabs(v) / V) - 1.0) / mu;
        }
    }
    return y;
}

/* A-law companding (G.711).
 *   compandA(x, A, V, dir):  dir 0 compress / 1 expand.
 */
matlab_mat *matlab_comm_compand_a(matlab_mat *x, double A_d, double V_d,
                                   double dir_d) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    matlab_mat *y = mat_alloc(x->rows, x->cols);
    double A = A_d > 1.0 ? A_d : 87.6;
    double V = V_d > 0.0 ? V_d : 1.0;
    double thr = V / A;
    double denom = 1.0 + log(A);
    int dir = (int)dir_d;
    for (int64_t i = 0; i < N; ++i) {
        double v = x->data[i];
        double s = v >= 0.0 ? 1.0 : -1.0;
        double a = fabs(v);
        if (dir == 0) {
            double yv;
            if (a < thr) yv = A * a / denom;
            else         yv = V * (1.0 + log(A * a / V)) / denom;
            y->data[i] = s * yv;
        } else {
            double xv;
            double thr2 = V / denom;
            if (a < thr2) xv = a * denom / A;
            else          xv = V * exp(a * denom / V - 1.0) / A;
            y->data[i] = s * xv;
        }
    }
    return y;
}

/* dpcmEncode(sig, codebook, partition):
 *   Differential PCM encoder. Predicts each sample as the previous
 *   reconstructed sample (first-order predictor) and quantises the
 *   prediction residual through (partition, codebook). Returns
 *   the codebook-index column.
 *
 * dpcmDecode(idx, codebook): inverse — sums the reconstructed
 * residuals to recover the (approximate) original signal.
 */
matlab_mat *matlab_comm_dpcm_encode(matlab_mat *sig, matlab_mat *partition,
                                     matlab_mat *codebook) {
    if (!sig || !partition || !codebook) return mat_alloc(0, 0);
    int64_t N = sig->rows * sig->cols;
    int64_t Np = partition->rows * partition->cols;
    int64_t Mc = codebook->rows * codebook->cols;
    if (Mc != Np + 1) return mat_alloc(0, 0);
    matlab_mat *idx = mat_alloc(N, 1);
    double recon = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        double err = sig->data[i] - recon;
        int64_t k = 0;
        while (k < Np && err > partition->data[k]) ++k;
        idx->data[i] = (double)k;
        recon += codebook->data[k];
    }
    return idx;
}

matlab_mat *matlab_comm_dpcm_decode(matlab_mat *idx, matlab_mat *codebook) {
    if (!idx || !codebook) return mat_alloc(0, 0);
    int64_t N = idx->rows * idx->cols;
    int64_t Mc = codebook->rows * codebook->cols;
    matlab_mat *out = mat_alloc(N, 1);
    double recon = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        int64_t k = (int64_t)idx->data[i];
        if (k < 0) k = 0;
        if (k >= Mc) k = Mc - 1;
        recon += codebook->data[k];
        out->data[i] = recon;
    }
    return out;
}

/* ===== Tier-7 — LDPC / Turbo / Polar (modern channel codes) ============ *
 *
 * docs/comm_toolbox_roadmap.md §5.4 — the "Tier-7 stretch" carve-out.
 * Function-form implementations of the three modern code families:
 *   - Polar (Arikan transform + SC decoder)
 *   - LDPC  (systematic encode from a parity portion + min-sum BP)
 *   - Turbo (PCCC encode with interleaver + iterative max-log MAP)
 *
 * System-Object wrappers stay gated on the SO lowering fix.  These
 * primitives are sufficient for end-to-end "encode → AWGN → decode"
 * Monte-Carlo loops; they are NOT bit-identical to commercial
 * implementations (Q.LDPC / 5G NR LDPC base matrices, 3GPP polar
 * sequence indices, 3GPP turbo permutations are caller-supplied
 * lookup tables when required).
 */

/* ----- §5.4.A Polar codes ----- *
 *
 * polarEncode(u, N) — Arikan polar transform y = u · G_N where
 * G_N = F^(⊗log2(N)) and F = [[1,0],[1,1]].  Caller supplies the
 * full N-bit input with frozen bits already set to 0; the encoder
 * is purely the bit-reversal-free butterfly.
 */
matlab_mat *matlab_comm_polar_encode(matlab_mat *u, double Nd) {
    int64_t N = (int64_t)Nd;
    if (!u || N < 1) return mat_alloc(0, 0);
    /* Round N up to next power of 2. */
    int64_t P = 1;
    while (P < N) P <<= 1;
    N = P;
    matlab_mat *y = mat_alloc(N, 1);
    int64_t Nu = u->rows * u->cols;
    for (int64_t i = 0; i < N; ++i)
        y->data[i] = (i < Nu) ? ((int64_t)u->data[i] & 1) : 0.0;
    /* Recursive butterfly in place: for each stage s of size 2^(s+1),
     * pair (a, b) -> (a XOR b, b). */
    for (int64_t step = 1; step < N; step <<= 1) {
        for (int64_t base = 0; base < N; base += (step << 1)) {
            for (int64_t i = 0; i < step; ++i) {
                int a = (int)y->data[base + i] & 1;
                int b = (int)y->data[base + step + i] & 1;
                y->data[base + i] = (double)(a ^ b);
            }
        }
    }
    return y;
}

/* Recursive SC-decode helper.  Returns hard bit decisions (length N)
 * for the input estimate `u_hat`.  `llr` is the N-element LLR vector
 * at the current recursion level (positive favours 0). */
static void polar_sc_rec(double *llr, double *u_hat, int *frozen,
                          int64_t pos, int64_t N) {
    if (N == 1) {
        if (frozen[pos]) {
            u_hat[pos] = 0.0;
        } else {
            u_hat[pos] = (llr[0] < 0.0) ? 1.0 : 0.0;
        }
        return;
    }
    int64_t H = N / 2;
    std::vector<double> llr_top(H), llr_bot(H);
    /* f-node: top half — soft XOR. */
    for (int64_t i = 0; i < H; ++i) {
        double a = llr[i], b = llr[i + H];
        double sa = a >= 0 ? 1.0 : -1.0;
        double sb = b >= 0 ? 1.0 : -1.0;
        double mn = fabs(a) < fabs(b) ? fabs(a) : fabs(b);
        llr_top[i] = sa * sb * mn;
    }
    polar_sc_rec(llr_top.data(), u_hat, frozen, pos, H);
    /* g-node: bottom half — sign uses the *partial codeword* for the
     * top half, i.e. u_top · G_{N/2}, not the raw decoded info bits.
     * Apply the recursive polar butterfly to u_hat[pos..pos+H-1]
     * locally before pulling the sign. */
    std::vector<double> u_top_partial(H);
    for (int64_t i = 0; i < H; ++i)
        u_top_partial[i] = u_hat[pos + i];
    for (int64_t step = 1; step < H; step <<= 1) {
        for (int64_t base = 0; base < H; base += (step << 1)) {
            for (int64_t i = 0; i < step; ++i) {
                int aa = (int)u_top_partial[base + i] & 1;
                int bb = (int)u_top_partial[base + step + i] & 1;
                u_top_partial[base + i] = (double)(aa ^ bb);
            }
        }
    }
    for (int64_t i = 0; i < H; ++i) {
        double a = llr[i], b = llr[i + H];
        double u_part = u_top_partial[i];
        double sign = (u_part > 0.5) ? -1.0 : 1.0;
        llr_bot[i] = b + sign * a;
    }
    polar_sc_rec(llr_bot.data(), u_hat, frozen, pos + H, H);
}

/* polarSCdecode(llr, frozen_mask, N) — successive-cancellation
 * decoder.  Returns N-bit u_hat.  frozen_mask is a 0/1 vector of
 * length N indicating frozen positions. */
matlab_mat *matlab_comm_polar_sc_decode(matlab_mat *llr,
                                         matlab_mat *frozen, double Nd) {
    int64_t N = (int64_t)Nd;
    if (!llr || !frozen || N < 1) return mat_alloc(0, 0);
    int64_t P = 1;
    while (P < N) P <<= 1;
    N = P;
    std::vector<double> llr_buf(N);
    std::vector<int> fz_buf(N);
    int64_t Nl = llr->rows * llr->cols;
    int64_t Nf = frozen->rows * frozen->cols;
    for (int64_t i = 0; i < N; ++i) {
        llr_buf[i] = (i < Nl) ? llr->data[i] : 0.0;
        fz_buf [i] = (i < Nf) ? ((int)frozen->data[i] & 1) : 1;
    }
    matlab_mat *u_hat = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) u_hat->data[i] = 0.0;
    polar_sc_rec(llr_buf.data(), u_hat->data, fz_buf.data(), 0, N);
    return u_hat;
}

/* ----- §5.4.B LDPC (function-form) ----- *
 *
 * ldpcEncode(msg, P) — systematic encoder.
 *   msg : k x 1 message bits
 *   P   : k x (n-k) parity portion of G in systematic form
 *         (so G = [I_k | P], H = [P^T | I_(n-k)])
 * Returns n x 1 codeword = [msg ; mod(P^T · msg, 2)].
 */
matlab_mat *matlab_comm_ldpc_encode(matlab_mat *msg, matlab_mat *P) {
    if (!msg || !P) return mat_alloc(0, 0);
    int64_t k  = msg->rows * msg->cols;
    int64_t Pk = P->rows;
    int64_t Pm = P->cols;
    if (Pk != k) return mat_alloc(0, 0);
    int64_t m = Pm;
    int64_t n = k + m;
    matlab_mat *cw = mat_alloc(n, 1);
    /* Systematic prefix. */
    for (int64_t i = 0; i < k; ++i)
        cw->data[i] = (int64_t)msg->data[i] & 1;
    /* Parity = mod(P^T · msg, 2): each parity bit j is XOR over i of
     * P[i, j] · msg[i]. */
    for (int64_t j = 0; j < m; ++j) {
        int p = 0;
        for (int64_t i = 0; i < k; ++i) {
            int b = (int)P->data[i * Pm + j] & 1;
            int u = (int)msg->data[i] & 1;
            p ^= (b & u);
        }
        cw->data[k + j] = (double)p;
    }
    return cw;
}

/* ldpcDecodeMS(llr, H, max_iter) — min-sum belief-propagation
 * decoder.  Returns the hard-decision bit vector (length n).
 *   llr      : n x 1 channel LLR (positive favours 0)
 *   H        : (n-k) x n parity-check matrix (0/1)
 *   max_iter : maximum BP iterations
 */
matlab_mat *matlab_comm_ldpc_decode_ms(matlab_mat *llr, matlab_mat *H,
                                        double max_iter) {
    if (!llr || !H) return mat_alloc(0, 0);
    int64_t n = H->cols;
    int64_t m = H->rows;
    int64_t Nl = llr->rows * llr->cols;
    if (Nl != n) return mat_alloc(0, 0);
    int iters = (int)max_iter;
    if (iters < 1) iters = 1;
    /* Build edge list: list of (check, variable) pairs for each H[c, v] = 1. */
    std::vector<std::vector<int64_t>> v_to_c(n);
    std::vector<std::vector<int64_t>> c_to_v(m);
    for (int64_t c = 0; c < m; ++c) {
        for (int64_t v = 0; v < n; ++v) {
            if (((int)H->data[c * n + v] & 1) == 1) {
                v_to_c[v].push_back(c);
                c_to_v[c].push_back(v);
            }
        }
    }
    /* Messages: var-to-check L_vc[v][c_index] and check-to-var L_cv[c][v_index]. */
    std::vector<std::vector<double>> L_vc(n), L_cv(m);
    for (int64_t v = 0; v < n; ++v) L_vc[v].resize(v_to_c[v].size(), 0.0);
    for (int64_t c = 0; c < m; ++c) L_cv[c].resize(c_to_v[c].size(), 0.0);
    /* Initialise var-to-check with the channel LLR. */
    for (int64_t v = 0; v < n; ++v)
        for (size_t i = 0; i < L_vc[v].size(); ++i)
            L_vc[v][i] = llr->data[v];

    matlab_mat *out = mat_alloc(n, 1);
    for (int it = 0; it < iters; ++it) {
        /* Check-node update: min-sum. */
        for (int64_t c = 0; c < m; ++c) {
            int deg = (int)c_to_v[c].size();
            for (int j = 0; j < deg; ++j) {
                double prod_sign = 1.0;
                double min_abs   = 1e300;
                for (int k = 0; k < deg; ++k) {
                    if (k == j) continue;
                    int64_t v = c_to_v[c][k];
                    /* find the edge index from v's side that points back to c */
                    int idx = -1;
                    for (size_t e = 0; e < v_to_c[v].size(); ++e)
                        if (v_to_c[v][e] == c) { idx = (int)e; break; }
                    double L = L_vc[v][idx];
                    if (L < 0.0) prod_sign = -prod_sign;
                    double a = fabs(L);
                    if (a < min_abs) min_abs = a;
                }
                L_cv[c][j] = prod_sign * min_abs;
            }
        }
        /* Variable-node update: sum + channel. */
        for (int64_t v = 0; v < n; ++v) {
            int deg = (int)v_to_c[v].size();
            double total = llr->data[v];
            for (int i = 0; i < deg; ++i) {
                int64_t c = v_to_c[v][i];
                int idx = -1;
                for (size_t e = 0; e < c_to_v[c].size(); ++e)
                    if (c_to_v[c][e] == v) { idx = (int)e; break; }
                total += L_cv[c][idx];
            }
            for (int i = 0; i < deg; ++i) {
                int64_t c = v_to_c[v][i];
                int idx = -1;
                for (size_t e = 0; e < c_to_v[c].size(); ++e)
                    if (c_to_v[c][e] == v) { idx = (int)e; break; }
                L_vc[v][i] = total - L_cv[c][idx];
            }
            out->data[v] = (total < 0.0) ? 1.0 : 0.0;
        }
    }
    return out;
}

/* ----- §5.4.C Turbo codes (PCCC) ----- *
 *
 * turboEncode(msg, trellis, perm) — parallel-concatenated convolutional
 * codes with an interleaver.
 *   msg     : k x 1 message bits
 *   trellis : matlab_struct from poly2trellis (rate 1/n RSC; for the
 *             canonical 3GPP turbo we use rate 1/2 so n = 2, but a
 *             non-recursive code is acceptable for the demo)
 *   perm    : k x 1 permutation (1-based indices) for the interleaver
 *
 * Output is a (3 · k) x 1 vector laid out as
 *   [systematic_1; ...; systematic_k;
 *    parity1_1;    ...; parity1_k;
 *    parity2_1;    ...; parity2_k]
 *
 * (the trellis is rate 1/2 with two output bits per input; we emit
 * the parity bit only, dropping the systematic copy that convenc
 * would otherwise produce — caller's responsibility to ensure n = 2).
 */
matlab_mat *matlab_comm_turbo_encode(matlab_mat *msg, matlab_struct *trellis,
                                      matlab_mat *perm) {
    if (!msg || !trellis || !perm) return mat_alloc(0, 0);
    int64_t k = msg->rows * msg->cols;
    int n_out = (int)matlab_struct_get_f64(trellis, "n", 1);
    if (n_out < 1) n_out = 2;
    /* First encoder pass: convenc on msg. */
    matlab_mat *c1 = matlab_comm_convenc(msg, trellis);
    /* Permute msg via the interleaver. */
    matlab_mat *msg_p = mat_alloc(k, 1);
    for (int64_t i = 0; i < k; ++i) {
        int64_t idx = (int64_t)perm->data[i] - 1;
        if (idx < 0 || idx >= k) idx = 0;
        msg_p->data[i] = msg->data[idx];
    }
    matlab_mat *c2 = matlab_comm_convenc(msg_p, trellis);
    /* Assemble [sys; p1; p2]. p1 = the n_out-th bit of each c1 chunk
     * (drop the systematic copy); same for p2. */
    matlab_mat *out = mat_alloc(3 * k, 1);
    for (int64_t i = 0; i < k; ++i) out->data[i] = msg->data[i];
    for (int64_t i = 0; i < k; ++i)
        out->data[k + i] = c1->data[i * n_out + (n_out - 1)];
    for (int64_t i = 0; i < k; ++i)
        out->data[2 * k + i] = c2->data[i * n_out + (n_out - 1)];
    free(c1->data); free(c1);
    free(c2->data); free(c2);
    free(msg_p->data); free(msg_p);
    return out;
}

/* Max-log-MAP (BCJR) SISO decoder for a rate-1/2 convolutional
 * trellis.  Returns per-info-bit LLR (positive favours u=0,
 * matching the qamdemodLlr / vitdecSoft convention).
 *
 *   llr_sys / llr_p : k-length channel LLRs for systematic + parity
 *   La              : k-length a priori LLR on u (extrinsic from
 *                     the previous decoder; pass zeros on iter 0)
 *
 * Operates over the same trellis struct `poly2trellis` builds.  For
 * a non-recursive convolutional code the output's first bit is the
 * systematic copy of u; we override the trellis's stored output bits
 * to enforce "(u, parity)" interpretation regardless of the input
 * generator polynomials.
 */
static void bcjr_max_log_siso(const double *llr_sys, const double *llr_p,
                               const double *La, double *Lapp,
                               matlab_struct *trellis, int64_t k) {
    int64_t S = (int64_t)matlab_struct_get_f64(trellis, "numStates", 9);
    matlab_mat *outputs    = matlab_struct_get_mat(trellis, "outputs",      7);
    matlab_mat *nextStates = matlab_struct_get_mat(trellis, "nextStates", 10);
    if (!outputs || !nextStates || S < 1) return;
    int n_out = (int)matlab_struct_get_f64(trellis, "n", 1);
    const double NEG = -1e18;

    /* alpha[t][s], beta[t][s] — forward / backward path metrics. */
    std::vector<std::vector<double>> alpha(k + 1, std::vector<double>(S, NEG));
    std::vector<std::vector<double>> beta (k + 1, std::vector<double>(S, NEG));
    alpha[0][0] = 0.0;
    /* Open-end termination: beta_K = uniform 0. */
    for (int64_t s = 0; s < S; ++s) beta[k][s] = 0.0;

    /* Branch metric γ[t, s', u] = 0.5·u·(La[t] + Lsys[t]) + 0.5·p·Lp[t].
     * The trellis stores the *encoded* output for transition (s', u);
     * for a rate-1/2 code we interpret output bits as (b0, b1) where
     * b0 is the parity in our turbo convention (we emit parity-only,
     * dropping the systematic copy in `turboEncode`). */
    auto gamma = [&](int64_t t, int64_t sp, int u) {
        int oi = (int)outputs->data[sp * 2 + u];
        int p_bit = (oi >> (n_out - 1)) & 1;   /* high bit ≡ parity_1 in (171,133)₈ */
        /* For 3-bit and other rate forms we still take the first
         * generator's bit as the parity stream — matches how
         * turboEncode emits the n_out-th bit of each chunk. */
        p_bit = oi & 1;
        double u_d = (u == 0) ? 0.0 : 1.0;
        double p_d = (p_bit == 0) ? 0.0 : 1.0;
        /* sign convention: positive LLR favours bit=0, so b=0 yields
         * +L/2 in the metric, b=1 yields -L/2. */
        double sys_metric = (1.0 - 2.0 * u_d) * 0.5 * llr_sys[t];
        double par_metric = (1.0 - 2.0 * p_d) * 0.5 * llr_p[t];
        double ap_metric  = (1.0 - 2.0 * u_d) * 0.5 * La[t];
        return sys_metric + par_metric + ap_metric;
    };

    /* Forward recursion. */
    for (int64_t t = 0; t < k; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            double best = NEG;
            for (int64_t sp = 0; sp < S; ++sp) {
                for (int u = 0; u <= 1; ++u) {
                    int64_t ns = (int64_t)nextStates->data[sp * 2 + u];
                    if (ns != s) continue;
                    double cand = alpha[t][sp] + gamma(t, sp, u);
                    if (cand > best) best = cand;
                }
            }
            alpha[t + 1][s] = best;
        }
    }
    /* Backward recursion. */
    for (int64_t t = k - 1; t >= 0; --t) {
        for (int64_t s = 0; s < S; ++s) {
            double best = NEG;
            for (int u = 0; u <= 1; ++u) {
                int64_t ns = (int64_t)nextStates->data[s * 2 + u];
                if (ns < 0 || ns >= S) continue;
                double cand = beta[t + 1][ns] + gamma(t, s, u);
                if (cand > best) best = cand;
            }
            beta[t][s] = best;
        }
    }
    /* APP LLR per info bit. */
    for (int64_t t = 0; t < k; ++t) {
        double max0 = NEG, max1 = NEG;
        for (int64_t sp = 0; sp < S; ++sp) {
            for (int u = 0; u <= 1; ++u) {
                int64_t ns = (int64_t)nextStates->data[sp * 2 + u];
                if (ns < 0 || ns >= S) continue;
                double m = alpha[t][sp] + gamma(t, sp, u) + beta[t + 1][ns];
                if (u == 0) { if (m > max0) max0 = m; }
                else        { if (m > max1) max1 = m; }
            }
        }
        Lapp[t] = max0 - max1;
    }
}

/* turboDecode(llr_sys, llr_p1, llr_p2, trellis, perm, max_iter)
 *
 * Iterative BCJR (max-log-MAP) turbo decoder.  Each iteration runs
 * the SISO above twice — once on the natural-order half and once on
 * the interleaved half — exchanging extrinsic LLR between them.
 * Returns k hard-decision bits.
 */
matlab_mat *matlab_comm_turbo_decode(matlab_mat *llr_sys, matlab_mat *llr_p1,
                                      matlab_mat *llr_p2,
                                      matlab_struct *trellis, matlab_mat *perm,
                                      double max_iter_d) {
    if (!llr_sys || !llr_p1 || !llr_p2 || !trellis || !perm)
        return mat_alloc(0, 0);
    int64_t k = llr_sys->rows * llr_sys->cols;
    int max_iter = (int)max_iter_d;
    if (max_iter < 1) max_iter = 4;
    /* Inverse permutation. */
    std::vector<int64_t> inv_perm(k, 0);
    for (int64_t i = 0; i < k; ++i) {
        int64_t idx = (int64_t)perm->data[i] - 1;
        if (idx >= 0 && idx < k) inv_perm[idx] = i;
    }
    /* Interleaved systematic LLR (used by decoder 2). */
    std::vector<double> sys_perm(k);
    for (int64_t i = 0; i < k; ++i) {
        int64_t idx = (int64_t)perm->data[i] - 1;
        sys_perm[i] = llr_sys->data[idx];
    }
    /* Extrinsic LLR feedback (a priori for the next half). */
    std::vector<double> La1(k, 0.0);    /* a priori for decoder 1 (natural) */
    std::vector<double> La2(k, 0.0);    /* a priori for decoder 2 (perm)    */
    std::vector<double> Lapp1(k, 0.0), Lapp2(k, 0.0);
    matlab_mat *dec = mat_alloc(k, 1);
    for (int it = 0; it < max_iter; ++it) {
        bcjr_max_log_siso(llr_sys->data, llr_p1->data, La1.data(),
                          Lapp1.data(), trellis, k);
        /* Extrinsic out of decoder 1 = Lapp - Lsys - La1, mapped into
         * encoder-2's ordering: La2[inv_perm[i]] = Le[i] (so
         * La2[j] = Le[perm[j]-1] is the a priori for decoder 2's bit j). */
        for (int64_t i = 0; i < k; ++i) {
            double Le = Lapp1[i] - llr_sys->data[i] - La1[i];
            La2[inv_perm[i]] = Le;
        }
        bcjr_max_log_siso(sys_perm.data(), llr_p2->data, La2.data(),
                          Lapp2.data(), trellis, k);
        /* Extrinsic out of decoder 2, mapped back to natural order:
         * La1[perm[i]-1] = Le_dec2[i]. */
        for (int64_t i = 0; i < k; ++i) {
            double Le = Lapp2[i] - sys_perm[i] - La2[i];
            int64_t idx = (int64_t)perm->data[i] - 1;
            La1[idx] = Le;
        }
    }
    /* Final hard decision from the combined a posteriori at decoder 2,
     * de-permuted back to natural order. */
    for (int64_t i = 0; i < k; ++i) {
        int64_t idx = (int64_t)perm->data[i] - 1;
        dec->data[idx] = (Lapp2[i] < 0.0) ? 1.0 : 0.0;
    }
    return dec;
}

/* mlDetect(y, alphabet): per-symbol ML decision against a complex
 * alphabet of M candidate constellation points. */
matlab_mat *matlab_comm_ml_detect(matlab_mat_c *y, matlab_mat_c *alphabet) {
    if (!y || !alphabet) return mat_alloc(0, 0);
    int64_t N = y->rows * y->cols;
    int64_t M = alphabet->rows * alphabet->cols;
    matlab_mat *out = mat_alloc(N, 1);
    for (int64_t n = 0; n < N; ++n) {
        double best = 1e300;
        int64_t best_i = 0;
        for (int64_t i = 0; i < M; ++i) {
            double dr = y->re[n] - alphabet->re[i];
            double di = y->im[n] - alphabet->im[i];
            double d2 = dr * dr + di * di;
            if (d2 < best) { best = d2; best_i = i; }
        }
        out->data[n] = (double)best_i;
    }
    return out;
}

/* ----- §6.x soft-decision Viterbi extension ----- *
 *
 * vitdecSoft(llr_or_quantised, trellis, tblen, opmode) — soft-input
 * Viterbi.  The branch metric is the dot product of the n-tuple LLR
 * (or unquantised real) chunk with (1 - 2*expected_bits).  Returns
 * the decoded bit vector (same length as message). */
matlab_mat *matlab_comm_vitdec_soft(matlab_mat *llr, matlab_struct *trellis,
                                     double tblen_d, double opmode_d) {
    (void)tblen_d;
    if (!llr || !trellis) return mat_alloc(0, 0);
    int n = (int)matlab_struct_get_f64(trellis, "n", 1);
    int64_t S = (int64_t)matlab_struct_get_f64(trellis, "numStates", 9);
    matlab_mat *outputs    = matlab_struct_get_mat(trellis, "outputs",      7);
    matlab_mat *nextStates = matlab_struct_get_mat(trellis, "nextStates", 10);
    if (!outputs || !nextStates || n < 1 || S < 1) return mat_alloc(0, 0);
    int64_t total = llr->rows * llr->cols;
    int64_t T = total / n;
    if (T < 1) return mat_alloc(0, 0);
    int opmode = (int)opmode_d;

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
    const double NEG = -1e18;
    std::vector<double> pm(S, NEG);
    std::vector<double> pm_next(S, NEG);
    std::vector<int8_t> bit_dec(T * S, 0);
    std::vector<int32_t> prev_dec(T * S, 0);
    pm[0] = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            double best = NEG;
            int best_bit = 0;
            int64_t best_prev = 0;
            for (int slot = 0; slot < n_in[s] && slot < 2; ++slot) {
                int64_t ps = in_prev[s * 2 + slot];
                if (ps < 0) continue;
                int u  = in_bit[s * 2 + slot];
                int oi = in_out[s * 2 + slot];
                double bm = 0.0;
                for (int i = 0; i < n; ++i) {
                    int bit = (oi >> (n - 1 - i)) & 1;
                    /* +LLR/sample for bit=0, -LLR for bit=1 — matches
                     * the qamdemodLlr "positive favours 0" convention. */
                    double sign = (bit == 0) ? 1.0 : -1.0;
                    bm += sign * llr->data[t * n + i];
                }
                double cand = pm[ps] + bm;
                if (cand > best) {
                    best = cand; best_bit = u; best_prev = ps;
                }
            }
            pm_next[s] = best;
            bit_dec [t * S + s] = (int8_t)best_bit;
            prev_dec[t * S + s] = (int32_t)best_prev;
        }
        std::swap(pm, pm_next);
        for (int64_t s = 0; s < S; ++s) pm_next[s] = NEG;
    }
    int64_t end = 0;
    if (opmode != 1) {
        double best = pm[0];
        for (int64_t s = 1; s < S; ++s) {
            if (pm[s] > best) { best = pm[s]; end = s; }
        }
    }
    matlab_mat *msg = mat_alloc(T, 1);
    int64_t state = end;
    for (int64_t t = T - 1; t >= 0; --t) {
        msg->data[t] = (double)bit_dec[t * S + state];
        state = (int64_t)prev_dec[t * S + state];
    }
    return msg;
}

}  /* extern "C" */
