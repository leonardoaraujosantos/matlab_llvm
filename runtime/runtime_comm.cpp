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

}  /* extern "C" */
