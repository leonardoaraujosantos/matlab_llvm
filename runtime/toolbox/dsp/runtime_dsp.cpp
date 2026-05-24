/* runtime_dsp.cpp — DSP System Toolbox runtime (Tiers 1–4).
 *
 * See docs/dsp_toolbox_roadmap.md for the full surface.  The DSP System
 * Toolbox's primary API is *System Objects* — stateful classdefs with the
 * `obj = dsp.FIRFilter(...)` constructor and the `y = obj(frame)`
 * call-syntax that dispatches to a `step` method and persists internal
 * state (the tapped-delay line / polyphase commutator / adaptive weights)
 * across frame-based calls.
 *
 * The MATLAB-side `dsp_classdefs.m` holds only the property bags + a thin
 * `step` method; the actual signal processing — and the in-place state
 * mutation — runs here.  Each `step` method forwards the receiver `obj`
 * (a `matlab_obj*`) plus the input frame to a `matlab_dsp_*_step` entry
 * below, which reads the coefficient + state properties via
 * `matlab_obj_get_mat`, processes the frame, writes the updated state back
 * via `matlab_obj_set_mat`, and returns the output frame.  This is the
 * exact obj-forwarding pattern proven by System Identification's EKF/RLS
 * recursive loop (runtime/toolbox/ident/runtime_ident.cpp) — it sidesteps
 * the classdef-method matrix-property type-inference gap (a value read
 * from `obj.Prop` inside a method body is untyped, so matrix ops on it do
 * not lower; doing the compute in C++ avoids that entirely).
 *
 * No external dependency — filters are hand-coded transposed-direct-form-II
 * / cascaded biquad / polyphase kernels; the design functions (Tier-2)
 * are hand-coded Remez / least-squares / Kaiser; the adaptive cores
 * (Tier-3) are LMS/RLS; multirate (Tier-4) is polyphase decimation /
 * interpolation + CIC + polyphase-FFT channelizer.
 *
 * static_cast / reinterpret_cast only — this TU is on the strict
 * no-C-style-cast list in CMakeLists.txt (mirrors runtime_images.cpp).
 */

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <complex>
#include <vector>

/* matlab_obj_* helpers — defined in runtime/matlab_runtime.cpp but not part
 * of the public matlab_runtime.h surface (same forward-decl pattern as
 * runtime/toolbox/ident/runtime_ident.cpp). */
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);

namespace {

/* Total element count of a matrix descriptor (rows*cols). */
int64_t mat_len(const matlab_mat *m) { return m ? m->rows * m->cols : 0; }

/* ---- small property helpers -------------------------------------------- */

/* Read a matrix property into a std::vector (empty when absent). */
std::vector<double> prop_vec(matlab_obj *o, const char *name, int64_t len) {
    matlab_mat *p = matlab_obj_get_mat(o, name, len);
    std::vector<double> v;
    if (p && mat_len(p) > 0) v.assign(p->data, p->data + mat_len(p));
    return v;
}

/* Allocate a 1×n row matrix from a vector. */
matlab_mat *row_mat(const std::vector<double> &v) {
    int64_t n = static_cast<int64_t>(v.size());
    matlab_mat *m = mat_alloc(n == 0 ? 1 : 1, n);
    for (int64_t i = 0; i < n; ++i) m->data[i] = v[static_cast<size_t>(i)];
    return m;
}

/* Allocate a column matrix matching the orientation of a reference mat. */
matlab_mat *like_mat(const std::vector<double> &v, const matlab_mat *ref) {
    int64_t n = static_cast<int64_t>(v.size());
    bool col = ref && ref->cols == 1 && ref->rows > 1;
    matlab_mat *m = col ? mat_alloc(n, 1) : mat_alloc(1, n);
    for (int64_t i = 0; i < n; ++i) m->data[i] = v[static_cast<size_t>(i)];
    return m;
}

/* Store a state vector property, preserving size (1×n row). */
void set_state(matlab_obj *o, const char *name, int64_t len,
               const std::vector<double> &s) {
    matlab_obj_set_mat(o, name, len, row_mat(s));
}

/* ---- core filter kernels ----------------------------------------------- */

/* Transposed Direct-Form II IIR (covers FIR when a == [1]).  Processes the
 * frame `x` in place against persisted state `z` (length = order); updates
 * `z`; returns the output frame.  Coefficients normalised by a[0]. */
std::vector<double> tdf2_filter(std::vector<double> b, std::vector<double> a,
                                const std::vector<double> &x,
                                std::vector<double> &z) {
    if (b.empty()) b.push_back(1.0);
    if (a.empty()) a.push_back(1.0);
    double a0 = a[0] != 0.0 ? a[0] : 1.0;
    for (double &c : b) c /= a0;
    for (double &c : a) c /= a0;
    size_t order = std::max(a.size(), b.size());
    if (order == 0) order = 1;
    order -= 1;
    b.resize(order + 1, 0.0);
    a.resize(order + 1, 0.0);
    if (z.size() < order) z.resize(order, 0.0);

    std::vector<double> y(x.size(), 0.0);
    for (size_t n = 0; n < x.size(); ++n) {
        double xn = x[n];
        double yn = b[0] * xn + (order > 0 ? z[0] : 0.0);
        for (size_t i = 0; i + 1 < order; ++i)
            z[i] = b[i + 1] * xn + z[i + 1] - a[i + 1] * yn;
        if (order > 0)
            z[order - 1] = b[order] * xn - a[order] * yn;
        y[n] = yn;
    }
    return y;
}

/* One cascaded second-order section (transposed DF-II), 2 state words. */
double biquad_sample(const double *s /*len 6: b0 b1 b2 a0 a1 a2*/,
                     double x, double &z1, double &z2) {
    double b0 = s[0], b1 = s[1], b2 = s[2];
    double a0 = s[3] != 0.0 ? s[3] : 1.0, a1 = s[4], a2 = s[5];
    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
    double y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
}

}  // namespace

/* ======================================================================= */
/* Tier-1 — core filter System Objects                                     */
/* ======================================================================= */

extern "C" {

/* dsp.FIRFilter / dsp.IIRFilter step.  Reads Numerator (+ Denominator for
 * IIR) and the persisted State; filters the frame; writes State back. */
matlab_mat *matlab_dsp_iir_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    std::vector<double> b = prop_vec(o, "Numerator", 9);
    std::vector<double> a = prop_vec(o, "Denominator", 11);
    if (a.empty()) a.push_back(1.0);
    std::vector<double> z = prop_vec(o, "State", 5);
    std::vector<double> xv(x->data, x->data + mat_len(x));
    std::vector<double> y = tdf2_filter(b, a, xv, z);
    set_state(o, "State", 5, z);
    return like_mat(y, x);
}

/* dsp.BiquadFilter / dsp.SOSFilter step.  SOSMatrix is K×6 row-major;
 * State is K×2 (z1,z2 per section), stored flat row-major. */
matlab_mat *matlab_dsp_sos_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    matlab_mat *sos = matlab_obj_get_mat(o, "SOSMatrix", 9);
    if (!sos || sos->cols != 6 || sos->rows < 1) return like_mat(
        std::vector<double>(x->data, x->data + mat_len(x)), x);
    int64_t K = sos->rows;
    std::vector<double> g = prop_vec(o, "ScaleValues", 11);
    std::vector<double> z = prop_vec(o, "State", 5);
    if (static_cast<int64_t>(z.size()) < 2 * K) z.assign(2 * K, 0.0);
    int64_t N = mat_len(x);
    std::vector<double> y(N, 0.0);
    /* ScaleValues are per-section input gains (length K or K+1).  When the
     * SOS came from tf2sos the gains are already folded into the numerator
     * coefficients, so the default ScaleValues = 1 is a no-op. */
    for (int64_t n = 0; n < N; ++n) {
        double v = x->data[n];
        for (int64_t k = 0; k < K; ++k) {
            if (k < static_cast<int64_t>(g.size())) v *= g[static_cast<size_t>(k)];
            v = biquad_sample(&sos->data[k * 6], v,
                              z[static_cast<size_t>(2 * k)],
                              z[static_cast<size_t>(2 * k + 1)]);
        }
        if (static_cast<int64_t>(g.size()) > K) v *= g[static_cast<size_t>(K)];
        y[static_cast<size_t>(n)] = v;
    }
    set_state(o, "State", 5, z);
    return like_mat(y, x);
}

/* dsp.Delay — integer delay line of length D.  State holds the last D
 * samples (most-recent last). */
matlab_mat *matlab_dsp_delay_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int64_t D = static_cast<int64_t>(matlab_obj_get_f64(o, "Length", 6));
    if (D < 0) D = 0;
    std::vector<double> z = prop_vec(o, "State", 5);
    if (static_cast<int64_t>(z.size()) != D) z.assign(static_cast<size_t>(D), 0.0);
    int64_t N = mat_len(x);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    /* Continuous stream: prepend the delay buffer, emit, keep last D. */
    std::vector<double> buf = z;
    buf.insert(buf.end(), x->data, x->data + N);
    for (int64_t n = 0; n < N; ++n) y[static_cast<size_t>(n)] = buf[static_cast<size_t>(n)];
    z.assign(buf.end() - D, buf.end());
    set_state(o, "State", 5, z);
    return like_mat(y, x);
}

/* Size the State property from the coefficient / length properties.  Called
 * from the IIR / Delay constructors where the state length depends on a
 * property that was just stored (and reading it back inside the constructor
 * body would be untyped).  Idempotent. */
void matlab_dsp_init_state(void *obj_v) {
    if (!obj_v) return;
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int64_t n = 0;
    matlab_mat *sos = matlab_obj_get_mat(o, "SOSMatrix", 9);
    matlab_mat *num = matlab_obj_get_mat(o, "Numerator", 9);
    matlab_mat *den = matlab_obj_get_mat(o, "Denominator", 11);
    if (sos && sos->cols == 6) {
        n = 2 * sos->rows;
    } else if (num || den) {
        int64_t ln = num ? mat_len(num) : 1;
        int64_t ld = den ? mat_len(den) : 1;
        n = std::max(ln, ld) - 1;
    } else {
        /* Delay: state length = Length samples. */
        n = static_cast<int64_t>(matlab_obj_get_f64(o, "Length", 6));
    }
    if (n < 1) n = 1;
    std::vector<double> z(static_cast<size_t>(n), 0.0);
    set_state(o, "State", 5, z);
}

/* Read the DiscreteState as a typed matrix (a copy).  The classdef
 * `getDiscreteState` forwards here so the caller gets a matrix-typed value
 * (a bare `obj.State` property read is untyped and cannot be indexed). */
matlab_mat *matlab_dsp_get_state(void *obj_v) {
    if (!obj_v) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    matlab_mat *st = matlab_obj_get_mat(o, "State", 5);
    if (!st) return mat_alloc(1, 1);
    matlab_mat *out = mat_alloc(st->rows, st->cols);
    int64_t n = mat_len(st);
    for (int64_t i = 0; i < n; ++i) out->data[i] = st->data[i];
    return out;
}

/* Lifecycle: reset zeroes every state-bearing property (keeping each one's
 * length): the filter State, and — for adaptive objects — the adapted
 * Weights, the tapped-input regressor, and the RLS inverse-correlation
 * matrix (re-seeded to delta^-1 * I).  release / clone are handled
 * MATLAB-side. */
static void zero_prop_inplace(matlab_obj *o, const char *name, int64_t len) {
    matlab_mat *p = matlab_obj_get_mat(o, name, len);
    if (!p) return;
    int64_t n = mat_len(p);
    std::vector<double> z(static_cast<size_t>(n > 0 ? n : 1), 0.0);
    set_state(o, name, len, z);
}
void matlab_dsp_reset(void *obj_v) {
    if (!obj_v) return;
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    zero_prop_inplace(o, "State", 5);
    zero_prop_inplace(o, "Weights", 7);
    zero_prop_inplace(o, "TapState", 8);
    matlab_mat *P = matlab_obj_get_mat(o, "InvCov", 6);
    if (P) {
        int64_t L = static_cast<int64_t>(lround(sqrt(static_cast<double>(mat_len(P)))));
        std::vector<double> p(static_cast<size_t>(L * L), 0.0);
        for (int64_t i = 0; i < L; ++i) p[static_cast<size_t>(i * L + i)] = 1e3;
        matlab_obj_set_mat(o, "InvCov", 6, row_mat(p));
    }
}

}  // extern "C"

/* ======================================================================= */
/* Tier-2 — filter design (function-form)                                  */
/* ======================================================================= */

namespace {

/* Desired amplitude of a piecewise-constant/linear band spec at frequency
 * f in [0,1] (1 = Nyquist).  Bands are pairs (F[2i],F[2i+1]) with amplitude
 * linearly interpolated from A[2i] to A[2i+1]; returns NaN outside any
 * band (transition / don't-care region). */
double band_desired(double f, const std::vector<double> &F,
                    const std::vector<double> &A, double &w_out, double wgt) {
    for (size_t i = 0; i + 1 < F.size(); i += 2) {
        if (f >= F[i] - 1e-12 && f <= F[i + 1] + 1e-12) {
            double span = F[i + 1] - F[i];
            double t = span > 0 ? (f - F[i]) / span : 0.0;
            w_out = wgt;
            return A[i] + t * (A[i + 1] - A[i]);
        }
    }
    w_out = 0.0;
    return 0.0;  // outside bands — excluded from the fit
}

/* firls — linear-phase Type-I (even order) least-squares FIR.  Minimises
 * the (weighted) integral of |A(f) - D(f)|^2 over the specified bands,
 * approximated on a dense frequency grid.  Returns the L = order+1 symmetric
 * taps. */
std::vector<double> firls_design(int order, const std::vector<double> &F,
                                 const std::vector<double> &A) {
    if (order < 1) order = 1;
    if (order % 2 == 1) order += 1;            // Type-I: even order
    int M = order / 2;
    int Ncoef = M + 1;
    /* Dense grid over the union of bands. */
    const int G = 16 * (order + 1);
    /* Normal equations Q a = c for the cosine-basis amplitude
     * A(w) = a0 + sum_{k=1}^{M} a_k cos(k w), w = pi f. */
    std::vector<double> Q(static_cast<size_t>(Ncoef * Ncoef), 0.0);
    std::vector<double> c(static_cast<size_t>(Ncoef), 0.0);
    for (size_t bi = 0; bi + 1 < F.size(); bi += 2) {
        double f0 = F[bi], f1 = F[bi + 1];
        for (int g = 0; g <= G; ++g) {
            double f = f0 + (f1 - f0) * g / G;
            double w = M_PI * f;
            double wt = 1.0;
            double scale = (g == 0 || g == G) ? 0.5 : 1.0;  // trapezoid
            double d;
            { double wo; d = band_desired(f, F, A, wo, 1.0); }
            std::vector<double> basis(static_cast<size_t>(Ncoef));
            for (int k = 0; k < Ncoef; ++k) basis[static_cast<size_t>(k)] = cos(k * w);
            for (int r = 0; r < Ncoef; ++r) {
                c[static_cast<size_t>(r)] += scale * wt * d * basis[static_cast<size_t>(r)];
                for (int cc = 0; cc < Ncoef; ++cc)
                    Q[static_cast<size_t>(r * Ncoef + cc)] +=
                        scale * wt * basis[static_cast<size_t>(r)] * basis[static_cast<size_t>(cc)];
            }
        }
    }
    /* Solve Q a = c via Gaussian elimination (Ncoef is small). */
    std::vector<double> a(static_cast<size_t>(Ncoef), 0.0);
    for (int i = 0; i < Ncoef; ++i) {
        int piv = i;
        for (int r = i + 1; r < Ncoef; ++r)
            if (fabs(Q[static_cast<size_t>(r * Ncoef + i)]) >
                fabs(Q[static_cast<size_t>(piv * Ncoef + i)])) piv = r;
        if (piv != i)
            for (int cc = 0; cc < Ncoef; ++cc)
                std::swap(Q[static_cast<size_t>(i * Ncoef + cc)],
                          Q[static_cast<size_t>(piv * Ncoef + cc)]),
                std::swap(c[static_cast<size_t>(i)], c[static_cast<size_t>(piv)]);
        double d = Q[static_cast<size_t>(i * Ncoef + i)];
        if (fabs(d) < 1e-300) d = 1e-300;
        for (int r = 0; r < Ncoef; ++r) {
            if (r == i) continue;
            double f = Q[static_cast<size_t>(r * Ncoef + i)] / d;
            for (int cc = 0; cc < Ncoef; ++cc)
                Q[static_cast<size_t>(r * Ncoef + cc)] -=
                    f * Q[static_cast<size_t>(i * Ncoef + cc)];
            c[static_cast<size_t>(r)] -= f * c[static_cast<size_t>(i)];
        }
    }
    for (int i = 0; i < Ncoef; ++i)
        a[static_cast<size_t>(i)] = c[static_cast<size_t>(i)] /
            (fabs(Q[static_cast<size_t>(i * Ncoef + i)]) < 1e-300 ? 1e-300 :
             Q[static_cast<size_t>(i * Ncoef + i)]);
    /* Reconstruct symmetric impulse response: h[M] = a0, h[M±k] = a_k/2. */
    std::vector<double> h(static_cast<size_t>(order + 1), 0.0);
    h[static_cast<size_t>(M)] = a[0];
    for (int k = 1; k <= M; ++k) {
        h[static_cast<size_t>(M - k)] = a[static_cast<size_t>(k)] / 2.0;
        h[static_cast<size_t>(M + k)] = a[static_cast<size_t>(k)] / 2.0;
    }
    return h;
}

/* firpm — Parks-McClellan equiripple FIR via the Remez exchange (Type-I,
 * even order).  Minimises the maximum weighted error over the bands.  A
 * compact dense-grid Remez: pick L+1 reference points, solve for the
 * alternating-error interpolant, relocate references to the local extrema
 * of the error, iterate to convergence. */
std::vector<double> firpm_design(int order, const std::vector<double> &F,
                                 const std::vector<double> &A) {
    if (order < 2) order = 2;
    if (order % 2 == 1) order += 1;
    int M = order / 2;
    int R = M + 1;                  // number of basis cosines / unknowns
    /* Dense grid over the bands. */
    std::vector<double> gf, gd, gw;
    const int dens = 16;
    for (size_t bi = 0; bi + 1 < F.size(); bi += 2) {
        int ng = dens * R;
        for (int g = 0; g <= ng; ++g) {
            double f = F[bi] + (F[bi + 1] - F[bi]) * g / ng;
            double wo; double d = band_desired(f, F, A, wo, 1.0);
            gf.push_back(f); gd.push_back(d); gw.push_back(1.0);
        }
    }
    int Ng = static_cast<int>(gf.size());
    if (Ng < R + 1) return firls_design(order, F, A);
    /* Initial reference set: R+1 equally spaced grid indices. */
    std::vector<int> ref(static_cast<size_t>(R + 1));
    for (int i = 0; i <= R; ++i)
        ref[static_cast<size_t>(i)] = static_cast<int>(
            static_cast<int64_t>(i) * (Ng - 1) / R);

    std::vector<double> a(static_cast<size_t>(R), 0.0);
    for (int iter = 0; iter < 40; ++iter) {
        /* Solve the linear system: for each reference k,
         * sum_j a_j cos(j*pi*f_k) + (-1)^k * delta / w_k = d_k. */
        int n = R + 1;
        std::vector<double> Mx(static_cast<size_t>(n * n), 0.0);
        std::vector<double> rhs(static_cast<size_t>(n), 0.0);
        for (int k = 0; k < n; ++k) {
            int gi = ref[static_cast<size_t>(k)];
            double w = M_PI * gf[static_cast<size_t>(gi)];
            for (int j = 0; j < R; ++j)
                Mx[static_cast<size_t>(k * n + j)] = cos(j * w);
            double wk = gw[static_cast<size_t>(gi)];
            Mx[static_cast<size_t>(k * n + R)] =
                ((k % 2 == 0) ? 1.0 : -1.0) / (wk != 0 ? wk : 1.0);
            rhs[static_cast<size_t>(k)] = gd[static_cast<size_t>(gi)];
        }
        /* Gaussian elimination. */
        for (int i = 0; i < n; ++i) {
            int piv = i;
            for (int r = i + 1; r < n; ++r)
                if (fabs(Mx[static_cast<size_t>(r * n + i)]) >
                    fabs(Mx[static_cast<size_t>(piv * n + i)])) piv = r;
            for (int cc = 0; cc < n; ++cc)
                std::swap(Mx[static_cast<size_t>(i * n + cc)],
                          Mx[static_cast<size_t>(piv * n + cc)]);
            std::swap(rhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(piv)]);
            double d = Mx[static_cast<size_t>(i * n + i)];
            if (fabs(d) < 1e-300) d = 1e-300;
            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                double f = Mx[static_cast<size_t>(r * n + i)] / d;
                for (int cc = 0; cc < n; ++cc)
                    Mx[static_cast<size_t>(r * n + cc)] -=
                        f * Mx[static_cast<size_t>(i * n + cc)];
                rhs[static_cast<size_t>(r)] -= f * rhs[static_cast<size_t>(i)];
            }
        }
        for (int i = 0; i < R; ++i)
            a[static_cast<size_t>(i)] = rhs[static_cast<size_t>(i)] /
                (fabs(Mx[static_cast<size_t>(i * n + i)]) < 1e-300 ? 1e-300 :
                 Mx[static_cast<size_t>(i * n + i)]);

        /* Compute error over the whole grid; find the R+1 largest-|error|
         * local extrema as the new reference set. */
        auto amp = [&](int gi) {
            double w = M_PI * gf[static_cast<size_t>(gi)], s = 0.0;
            for (int j = 0; j < R; ++j) s += a[static_cast<size_t>(j)] * cos(j * w);
            return s;
        };
        std::vector<double> err(static_cast<size_t>(Ng));
        for (int gi = 0; gi < Ng; ++gi)
            err[static_cast<size_t>(gi)] =
                gw[static_cast<size_t>(gi)] * (gd[static_cast<size_t>(gi)] - amp(gi));
        std::vector<int> ext;
        for (int gi = 1; gi + 1 < Ng; ++gi) {
            double e = err[static_cast<size_t>(gi)];
            double el = err[static_cast<size_t>(gi - 1)];
            double er = err[static_cast<size_t>(gi + 1)];
            if ((e > el && e >= er) || (e < el && e <= er)) ext.push_back(gi);
        }
        ext.push_back(0); ext.push_back(Ng - 1);
        std::sort(ext.begin(), ext.end());
        ext.erase(std::unique(ext.begin(), ext.end()), ext.end());
        if (static_cast<int>(ext.size()) < R + 1) break;
        /* Keep the R+1 extrema with the largest |error|. */
        std::sort(ext.begin(), ext.end(), [&](int x, int y) {
            return fabs(err[static_cast<size_t>(x)]) >
                   fabs(err[static_cast<size_t>(y)]);
        });
        ext.resize(static_cast<size_t>(R + 1));
        std::sort(ext.begin(), ext.end());
        bool same = (ext == ref);
        ref = ext;
        if (same) break;
    }
    /* Reconstruct symmetric impulse response from the cosine coefficients. */
    std::vector<double> h(static_cast<size_t>(order + 1), 0.0);
    h[static_cast<size_t>(M)] = a[0];
    for (int k = 1; k <= M; ++k) {
        h[static_cast<size_t>(M - k)] = a[static_cast<size_t>(k)] / 2.0;
        h[static_cast<size_t>(M + k)] = a[static_cast<size_t>(k)] / 2.0;
    }
    return h;
}

}  // namespace

extern "C" {

/* firpm(order, F, A) / firls(order, F, A) — single-return b (the L=order+1
 * symmetric taps).  F = band edges in [0,1] (1 = Nyquist), A = desired
 * amplitudes at each edge. */
matlab_mat *matlab_dsp_firpm(double order, matlab_mat *F, matlab_mat *A) {
    std::vector<double> f(F->data, F->data + mat_len(F));
    std::vector<double> a(A->data, A->data + mat_len(A));
    return row_mat(firpm_design(static_cast<int>(order), f, a));
}
matlab_mat *matlab_dsp_firls(double order, matlab_mat *F, matlab_mat *A) {
    std::vector<double> f(F->data, F->data + mat_len(F));
    std::vector<double> a(A->data, A->data + mat_len(A));
    return row_mat(firls_design(static_cast<int>(order), f, a));
}

/* iirnotch(w0, bw) / iirpeak(w0, bw) — second-order notch / peak biquads.
 * w0 = notch/peak frequency (×pi rad/sample), bw = -3 dB bandwidth (×pi).
 * Split into _b / _a (besself-style multi-return). */
static void notch_peak_ba(double w0, double bw, bool peak,
                          std::vector<double> &b, std::vector<double> &a) {
    double Wo = w0, BW = bw;
    double gb = M_SQRT1_2;                 // -3 dB gain
    double beta = (sqrt(1.0 - gb * gb) / gb) * tan(BW * M_PI / 2.0);
    double gain = 1.0 / (1.0 + beta);
    double c = cos(Wo * M_PI);
    if (peak) {
        b = { (1.0 - gain), 0.0, -(1.0 - gain) };
        a = { 1.0, -2.0 * gain * c, (2.0 * gain - 1.0) };
    } else {
        b = { gain, -2.0 * gain * c, gain };
        a = { 1.0, -2.0 * gain * c, (2.0 * gain - 1.0) };
    }
}
matlab_mat *matlab_dsp_iirnotch_b(double w0, double bw) {
    std::vector<double> b, a; notch_peak_ba(w0, bw, false, b, a); return row_mat(b);
}
matlab_mat *matlab_dsp_iirnotch_a(double w0, double bw) {
    std::vector<double> b, a; notch_peak_ba(w0, bw, false, b, a); return row_mat(a);
}
matlab_mat *matlab_dsp_iirpeak_b(double w0, double bw) {
    std::vector<double> b, a; notch_peak_ba(w0, bw, true, b, a); return row_mat(b);
}
matlab_mat *matlab_dsp_iirpeak_a(double w0, double bw) {
    std::vector<double> b, a; notch_peak_ba(w0, bw, true, b, a); return row_mat(a);
}

}  // extern "C"

/* ======================================================================= */
/* Tier-3 — adaptive filter System Objects                                 */
/* ======================================================================= */

extern "C" {

/* dsp.LMSFilter step — LMS / NLMS adaptive FIR.  Reads Length, StepSize,
 * Method (0 = LMS, 1 = NLMS); persists Weights + the tapped-input
 * regressor (TapState).  Inputs: x (filter input frame), d (desired
 * frame).  Returns the error e = d - y (the cleaned signal in the acoustic
 * noise-cancellation use case).  */
matlab_mat *matlab_dsp_lms_step(void *obj_v, matlab_mat *x, matlab_mat *d) {
    if (!obj_v || !x || !d) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int L = static_cast<int>(matlab_obj_get_f64(o, "Length", 6));
    if (L < 1) L = 1;
    double mu = matlab_obj_get_f64(o, "StepSize", 8);
    int method = static_cast<int>(matlab_obj_get_f64(o, "Method", 6));
    std::vector<double> w = prop_vec(o, "Weights", 7);
    if (static_cast<int>(w.size()) != L) w.assign(static_cast<size_t>(L), 0.0);
    std::vector<double> buf = prop_vec(o, "TapState", 8);
    if (static_cast<int>(buf.size()) != L) buf.assign(static_cast<size_t>(L), 0.0);

    int64_t N = std::min(mat_len(x), mat_len(d));
    std::vector<double> e(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        /* Shift the tapped-delay regressor: buf[0] = newest sample. */
        for (int i = L - 1; i > 0; --i) buf[static_cast<size_t>(i)] = buf[static_cast<size_t>(i - 1)];
        buf[0] = x->data[n];
        double y = 0.0, p = 0.0;
        for (int i = 0; i < L; ++i) {
            y += w[static_cast<size_t>(i)] * buf[static_cast<size_t>(i)];
            p += buf[static_cast<size_t>(i)] * buf[static_cast<size_t>(i)];
        }
        double err = d->data[n] - y;
        double step = mu;
        if (method == 1) step = mu / (1e-6 + p);          // NLMS
        for (int i = 0; i < L; ++i)
            w[static_cast<size_t>(i)] += step * err * buf[static_cast<size_t>(i)];
        e[static_cast<size_t>(n)] = err;
    }
    set_state(o, "Weights", 7, w);
    set_state(o, "TapState", 8, buf);
    return like_mat(e, x);
}

/* dsp.RLSFilter step — recursive least-squares adaptive FIR.  Reads Length,
 * ForgettingFactor; persists Weights, TapState, and the L×L inverse
 * correlation matrix P.  Same I/O contract as LMS. */
matlab_mat *matlab_dsp_rls_step(void *obj_v, matlab_mat *x, matlab_mat *d) {
    if (!obj_v || !x || !d) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int L = static_cast<int>(matlab_obj_get_f64(o, "Length", 6));
    if (L < 1) L = 1;
    double lam = matlab_obj_get_f64(o, "ForgettingFactor", 16);
    if (lam <= 0.0 || lam > 1.0) lam = 1.0;
    std::vector<double> w = prop_vec(o, "Weights", 7);
    if (static_cast<int>(w.size()) != L) w.assign(static_cast<size_t>(L), 0.0);
    std::vector<double> buf = prop_vec(o, "TapState", 8);
    if (static_cast<int>(buf.size()) != L) buf.assign(static_cast<size_t>(L), 0.0);
    std::vector<double> P = prop_vec(o, "InvCov", 6);
    if (static_cast<int>(P.size()) != L * L) {
        P.assign(static_cast<size_t>(L * L), 0.0);
        for (int i = 0; i < L; ++i) P[static_cast<size_t>(i * L + i)] = 1e3;  // delta^-1
    }

    int64_t N = std::min(mat_len(x), mat_len(d));
    std::vector<double> e(static_cast<size_t>(N), 0.0);
    std::vector<double> Px(static_cast<size_t>(L), 0.0);
    std::vector<double> k(static_cast<size_t>(L), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        for (int i = L - 1; i > 0; --i) buf[static_cast<size_t>(i)] = buf[static_cast<size_t>(i - 1)];
        buf[0] = x->data[n];
        /* Px = P * buf. */
        for (int i = 0; i < L; ++i) {
            double s = 0.0;
            for (int j = 0; j < L; ++j) s += P[static_cast<size_t>(i * L + j)] * buf[static_cast<size_t>(j)];
            Px[static_cast<size_t>(i)] = s;
        }
        double den = lam;
        for (int i = 0; i < L; ++i) den += buf[static_cast<size_t>(i)] * Px[static_cast<size_t>(i)];
        if (fabs(den) < 1e-300) den = 1e-300;
        for (int i = 0; i < L; ++i) k[static_cast<size_t>(i)] = Px[static_cast<size_t>(i)] / den;
        double y = 0.0;
        for (int i = 0; i < L; ++i) y += w[static_cast<size_t>(i)] * buf[static_cast<size_t>(i)];
        double err = d->data[n] - y;
        for (int i = 0; i < L; ++i) w[static_cast<size_t>(i)] += k[static_cast<size_t>(i)] * err;
        /* P = (P - k * Px') / lam. */
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                P[static_cast<size_t>(i * L + j)] =
                    (P[static_cast<size_t>(i * L + j)] - k[static_cast<size_t>(i)] * Px[static_cast<size_t>(j)]) / lam;
        e[static_cast<size_t>(n)] = err;
    }
    set_state(o, "Weights", 7, w);
    set_state(o, "TapState", 8, buf);
    /* Persist P as a flat row (re-read as L*L next frame). */
    matlab_obj_set_mat(o, "InvCov", 6, row_mat(P));
    return like_mat(e, x);
}

/* ======================================================================= */
/* Tier-4 — multirate + multistage + filter banks                          */
/* ======================================================================= */

/* dsp.FIRDecimator step.  Anti-alias FIR (Numerator), decimate by M.
 * Output length = N / M.  State = the FIR tapped-delay line. */
matlab_mat *matlab_dsp_firdecim_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int M = static_cast<int>(matlab_obj_get_f64(o, "DecimationFactor", 16));
    if (M < 1) M = 1;
    std::vector<double> b = prop_vec(o, "Numerator", 9);
    if (b.empty()) b.push_back(1.0);
    std::vector<double> z = prop_vec(o, "State", 5);
    std::vector<double> a = {1.0};
    std::vector<double> xv(x->data, x->data + mat_len(x));
    std::vector<double> yf = tdf2_filter(b, a, xv, z);
    int64_t N = static_cast<int64_t>(yf.size());
    int64_t Nout = N / M;
    std::vector<double> y(static_cast<size_t>(Nout), 0.0);
    /* Phase: take every M-th sample starting at offset M-1 — gives the
     * full-length anti-aliased decimated output.  When N is a multiple of
     * M (the standard usage), the polyphase phase is consistent across
     * frames because the filter state carries forward. */
    for (int64_t j = 0; j < Nout; ++j) y[static_cast<size_t>(j)] = yf[static_cast<size_t>(j * M + (M - 1))];
    set_state(o, "State", 5, z);
    matlab_mat *out = (x->cols == 1) ? mat_alloc(Nout, 1) : mat_alloc(1, Nout);
    for (int64_t j = 0; j < Nout; ++j) out->data[j] = y[static_cast<size_t>(j)];
    return out;
}

/* dsp.FIRInterpolator step.  Insert L-1 zeros between input samples, then
 * filter by Numerator with state.  Standard interpolator gain = L (folded
 * into the output scale). */
matlab_mat *matlab_dsp_firinterp_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int L = static_cast<int>(matlab_obj_get_f64(o, "InterpolationFactor", 19));
    if (L < 1) L = 1;
    std::vector<double> b = prop_vec(o, "Numerator", 9);
    if (b.empty()) b.push_back(1.0);
    std::vector<double> z = prop_vec(o, "State", 5);
    std::vector<double> a = {1.0};
    int64_t N = mat_len(x);
    std::vector<double> xu(static_cast<size_t>(N * L), 0.0);
    for (int64_t i = 0; i < N; ++i) xu[static_cast<size_t>(i * L)] = x->data[i];
    std::vector<double> yf = tdf2_filter(b, a, xu, z);
    for (double &v : yf) v *= L;        // interpolator gain
    set_state(o, "State", 5, z);
    matlab_mat *out = (x->cols == 1) ? mat_alloc(static_cast<int64_t>(yf.size()), 1)
                                     : mat_alloc(1, static_cast<int64_t>(yf.size()));
    for (size_t i = 0; i < yf.size(); ++i) out->data[i] = yf[i];
    return out;
}

/* dsp.CICDecimator step.  Multiplier-free N-stage CIC: N integrators at
 * the high rate, downsample by R, N comb stages at the low rate
 * (differential delay = 1).  Multi-stage state stored flat. */
matlab_mat *matlab_dsp_cicdecim_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int R = static_cast<int>(matlab_obj_get_f64(o, "DecimationFactor", 16));
    int S = static_cast<int>(matlab_obj_get_f64(o, "NumSections", 11));
    if (R < 1) R = 1;
    if (S < 1) S = 1;
    std::vector<double> integ = prop_vec(o, "IntState", 8);
    if (static_cast<int>(integ.size()) != S) integ.assign(static_cast<size_t>(S), 0.0);
    std::vector<double> comb = prop_vec(o, "CombState", 9);
    if (static_cast<int>(comb.size()) != S) comb.assign(static_cast<size_t>(S), 0.0);

    int64_t N = mat_len(x);
    int64_t Nout = N / R;
    std::vector<double> y(static_cast<size_t>(Nout), 0.0);
    /* Integrate at the high rate; emit every R-th integrated sample to the
     * comb section.  Comb: differential of order 1 per stage. */
    int64_t outIdx = 0;
    for (int64_t n = 0; n < N; ++n) {
        double v = x->data[n];
        for (int k = 0; k < S; ++k) {
            integ[static_cast<size_t>(k)] += v;
            v = integ[static_cast<size_t>(k)];
        }
        if ((n + 1) % R == 0 && outIdx < Nout) {
            for (int k = 0; k < S; ++k) {
                double prev = comb[static_cast<size_t>(k)];
                comb[static_cast<size_t>(k)] = v;
                v = v - prev;
            }
            y[static_cast<size_t>(outIdx++)] = v;
        }
    }
    set_state(o, "IntState", 8, integ);
    set_state(o, "CombState", 9, comb);
    matlab_mat *out = (x->cols == 1) ? mat_alloc(Nout, 1) : mat_alloc(1, Nout);
    for (int64_t j = 0; j < Nout; ++j) out->data[j] = y[static_cast<size_t>(j)];
    return out;
}

/* dsp.CICInterpolator step.  Hogenauer interpolation: N combs at the low
 * rate, upsample by R (zero-stuff), N integrators at the high rate. */
matlab_mat *matlab_dsp_cicinterp_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int R = static_cast<int>(matlab_obj_get_f64(o, "InterpolationFactor", 19));
    int S = static_cast<int>(matlab_obj_get_f64(o, "NumSections", 11));
    if (R < 1) R = 1;
    if (S < 1) S = 1;
    std::vector<double> integ = prop_vec(o, "IntState", 8);
    if (static_cast<int>(integ.size()) != S) integ.assign(static_cast<size_t>(S), 0.0);
    std::vector<double> comb = prop_vec(o, "CombState", 9);
    if (static_cast<int>(comb.size()) != S) comb.assign(static_cast<size_t>(S), 0.0);

    int64_t N = mat_len(x);
    int64_t Nout = N * R;
    std::vector<double> y(static_cast<size_t>(Nout), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        double v = x->data[n];
        for (int k = 0; k < S; ++k) {
            double prev = comb[static_cast<size_t>(k)];
            comb[static_cast<size_t>(k)] = v;
            v = v - prev;
        }
        /* Zero-stuff: first slot gets v, the next R-1 get 0. */
        for (int r = 0; r < R; ++r) {
            double u = (r == 0) ? v : 0.0;
            for (int k = 0; k < S; ++k) {
                integ[static_cast<size_t>(k)] += u;
                u = integ[static_cast<size_t>(k)];
            }
            y[static_cast<size_t>(n * R + r)] = u;
        }
    }
    set_state(o, "IntState", 8, integ);
    set_state(o, "CombState", 9, comb);
    matlab_mat *out = (x->cols == 1) ? mat_alloc(Nout, 1) : mat_alloc(1, Nout);
    for (int64_t j = 0; j < Nout; ++j) out->data[j] = y[static_cast<size_t>(j)];
    return out;
}

/* dsp.SampleRateConverter / dsp.FIRRateConverter step — rational L/M rate
 * change.  Upsample by L (zero-stuff), filter by Numerator (state carried
 * across frames), downsample by M.  Standard polyphase output gain = L. */
matlab_mat *matlab_dsp_rateconv_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int L = static_cast<int>(matlab_obj_get_f64(o, "InterpolationFactor", 19));
    int M = static_cast<int>(matlab_obj_get_f64(o, "DecimationFactor", 16));
    if (L < 1) L = 1;
    if (M < 1) M = 1;
    std::vector<double> b = prop_vec(o, "Numerator", 9);
    if (b.empty()) b.push_back(1.0);
    std::vector<double> z = prop_vec(o, "State", 5);
    std::vector<double> a = {1.0};
    int64_t N = mat_len(x);
    std::vector<double> xu(static_cast<size_t>(N * L), 0.0);
    for (int64_t i = 0; i < N; ++i) xu[static_cast<size_t>(i * L)] = x->data[i];
    std::vector<double> yf = tdf2_filter(b, a, xu, z);
    for (double &v : yf) v *= L;
    int64_t Nout = static_cast<int64_t>(yf.size()) / M;
    std::vector<double> y(static_cast<size_t>(Nout), 0.0);
    for (int64_t j = 0; j < Nout; ++j) y[static_cast<size_t>(j)] = yf[static_cast<size_t>(j * M + (M - 1))];
    set_state(o, "State", 5, z);
    matlab_mat *out = (x->cols == 1) ? mat_alloc(Nout, 1) : mat_alloc(1, Nout);
    for (int64_t j = 0; j < Nout; ++j) out->data[j] = y[static_cast<size_t>(j)];
    return out;
}

/* ======================================================================= */
/* Tier-5 — transforms + sources + streaming stats + measurement           */
/* ======================================================================= */

/* dsp.SineWave step(obj) — generate SamplesPerFrame samples of a sine,
 * persisting the phase across calls.  Frequency in Hz, SampleRate in Hz.
 * Phase wraps to (-pi, pi] to keep precision over long streams. */
matlab_mat *matlab_dsp_sine_step(void *obj_v) {
    if (!obj_v) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int N = static_cast<int>(matlab_obj_get_f64(o, "SamplesPerFrame", 15));
    if (N < 1) N = 1;
    double f  = matlab_obj_get_f64(o, "Frequency", 9);
    double fs = matlab_obj_get_f64(o, "SampleRate", 10);
    double A  = matlab_obj_get_f64(o, "Amplitude", 9);
    double ph = matlab_obj_get_f64(o, "Phase", 5);
    if (fs <= 0.0) fs = 1.0;
    double dphi = 2.0 * M_PI * f / fs;
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int n = 0; n < N; ++n) {
        y[static_cast<size_t>(n)] = A * sin(ph);
        ph += dphi;
    }
    /* Wrap phase to keep precision. */
    ph = fmod(ph, 2.0 * M_PI);
    if (ph >  M_PI) ph -= 2.0 * M_PI;
    if (ph < -M_PI) ph += 2.0 * M_PI;
    matlab_obj_set_f64(o, "Phase", 5, ph);
    return row_mat(y);
}

/* dsp.NCO step(obj) — phase-accumulator numerically-controlled oscillator
 * (real sine output; complex variant carved as a follow-on). */
matlab_mat *matlab_dsp_nco_step(void *obj_v) {
    return matlab_dsp_sine_step(obj_v);
}

/* dsp.Chirp step(obj) — linear-frequency chirp.  Maintains InstFreq + Phase
 * for streaming-continuous chirps across step calls. */
matlab_mat *matlab_dsp_chirp_step(void *obj_v) {
    if (!obj_v) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int N = static_cast<int>(matlab_obj_get_f64(o, "SamplesPerFrame", 15));
    if (N < 1) N = 1;
    double fs   = matlab_obj_get_f64(o, "SampleRate", 10);
    double f0   = matlab_obj_get_f64(o, "InitialFrequency", 16);
    double rate = matlab_obj_get_f64(o, "FrequencySweepRate", 18);  // Hz/sec
    double f    = matlab_obj_get_f64(o, "InstFreq", 8);
    double ph   = matlab_obj_get_f64(o, "Phase", 5);
    if (fs <= 0.0) fs = 1.0;
    if (f == 0.0) f = f0;
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int n = 0; n < N; ++n) {
        y[static_cast<size_t>(n)] = sin(ph);
        ph += 2.0 * M_PI * f / fs;
        f  += rate / fs;
    }
    ph = fmod(ph, 2.0 * M_PI);
    matlab_obj_set_f64(o, "Phase", 5, ph);
    matlab_obj_set_f64(o, "InstFreq", 8, f);
    return row_mat(y);
}

/* Generic sliding-window aggregator: pulls WindowLength samples worth of
 * history forward across frames, applies the reduction `op` at each output
 * step.  op codes: 0 = mean, 1 = RMS, 2 = max, 3 = min, 4 = std. */
static matlab_mat *sliding_window_step(void *obj_v, matlab_mat *x, int op) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int W = static_cast<int>(matlab_obj_get_f64(o, "WindowLength", 12));
    if (W < 1) W = 1;
    std::vector<double> buf = prop_vec(o, "Window", 6);
    if (static_cast<int>(buf.size()) != W) buf.assign(static_cast<size_t>(W), 0.0);
    int idx = static_cast<int>(matlab_obj_get_f64(o, "WriteIdx", 8));
    if (idx < 0 || idx >= W) idx = 0;

    int64_t N = mat_len(x);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        buf[static_cast<size_t>(idx)] = x->data[n];
        idx = (idx + 1) % W;
        /* Reduce over the whole window — robust, simple; rolling-update
         * variants are a carve-down. */
        double acc = (op == 2) ? -1e300 : (op == 3) ? 1e300 : 0.0;
        for (int i = 0; i < W; ++i) {
            double v = buf[static_cast<size_t>(i)];
            switch (op) {
            case 0: acc += v; break;
            case 1: acc += v * v; break;
            case 2: if (v > acc) acc = v; break;
            case 3: if (v < acc) acc = v; break;
            case 4: acc += v; break;
            }
        }
        double out = 0.0;
        if (op == 0)        out = acc / W;
        else if (op == 1)   out = sqrt(acc / W);
        else if (op == 4) {
            double m = acc / W, var = 0.0;
            for (int i = 0; i < W; ++i) {
                double d = buf[static_cast<size_t>(i)] - m;
                var += d * d;
            }
            out = sqrt(var / W);
        } else              out = acc;
        y[static_cast<size_t>(n)] = out;
    }
    set_state(o, "Window", 6, buf);
    matlab_obj_set_f64(o, "WriteIdx", 8, static_cast<double>(idx));
    return like_mat(y, x);
}
matlab_mat *matlab_dsp_movavg_step(void *obj_v, matlab_mat *x) { return sliding_window_step(obj_v, x, 0); }
matlab_mat *matlab_dsp_movrms_step(void *obj_v, matlab_mat *x) { return sliding_window_step(obj_v, x, 1); }
matlab_mat *matlab_dsp_movmax_step(void *obj_v, matlab_mat *x) { return sliding_window_step(obj_v, x, 2); }
matlab_mat *matlab_dsp_movmin_step(void *obj_v, matlab_mat *x) { return sliding_window_step(obj_v, x, 3); }
matlab_mat *matlab_dsp_movstd_step(void *obj_v, matlab_mat *x) { return sliding_window_step(obj_v, x, 4); }

/* dsp.PeakFinder step(obj, x) — detect local maxima in the frame, with
 * state carrying the last input sample + direction across frames so peaks
 * straddling a frame boundary are not missed.  Returns a same-shape vector
 * with peak amplitudes at the peak indices, zeros elsewhere. */
matlab_mat *matlab_dsp_peakfind_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    double prev = matlab_obj_get_f64(o, "Prev", 4);
    double dir  = matlab_obj_get_f64(o, "Dir", 3);  /* +1 rising, -1 falling */
    int64_t N = mat_len(x);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        double cur = x->data[n];
        if (cur > prev) {
            if (dir < 0.0 && n > 0) y[static_cast<size_t>(n - 1)] = prev;
            dir = 1.0;
        } else if (cur < prev) {
            if (dir > 0.0) {
                if (n > 0) y[static_cast<size_t>(n - 1)] = prev;
            }
            dir = -1.0;
        }
        prev = cur;
    }
    matlab_obj_set_f64(o, "Prev", 4, prev);
    matlab_obj_set_f64(o, "Dir", 3, dir);
    return like_mat(y, x);
}

/* dsp.DCBlocker step(obj, x) — first-order highpass y[n] = x[n] - x[n-1] +
 * a * y[n-1] (a close to 1).  State = (x_prev, y_prev). */
matlab_mat *matlab_dsp_dcblock_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    double a  = matlab_obj_get_f64(o, "Alpha", 5);
    if (a == 0.0) a = 0.995;
    double xp = matlab_obj_get_f64(o, "Xprev", 5);
    double yp = matlab_obj_get_f64(o, "Yprev", 5);
    int64_t N = mat_len(x);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        double cur = x->data[n];
        double out = cur - xp + a * yp;
        y[static_cast<size_t>(n)] = out;
        xp = cur;
        yp = out;
    }
    matlab_obj_set_f64(o, "Xprev", 5, xp);
    matlab_obj_set_f64(o, "Yprev", 5, yp);
    return like_mat(y, x);
}

/* dsp.ZeroCrossingDetector step(obj, x) — returns a same-shape vector of
 * 1 at zero-crossing samples (sign change vs the previous sample including
 * the last sample of the previous frame), 0 elsewhere. */
matlab_mat *matlab_dsp_zcd_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    double prev_sign = matlab_obj_get_f64(o, "PrevSign", 8);
    int64_t N = mat_len(x);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        double cur = x->data[n];
        double sgn = (cur > 0.0) ? 1.0 : (cur < 0.0) ? -1.0 : 0.0;
        if (sgn != 0.0 && prev_sign != 0.0 && sgn != prev_sign)
            y[static_cast<size_t>(n)] = 1.0;
        if (sgn != 0.0) prev_sign = sgn;
    }
    matlab_obj_set_f64(o, "PrevSign", 8, prev_sign);
    return like_mat(y, x);
}

/* dsp.SpectrumEstimator step(obj, x) — push the frame into a sliding
 * FFTLength buffer; when a full FFTLength window has accumulated, compute
 * the Hann-windowed periodogram and exponentially average it into the PSD
 * accumulator.  Always returns the current one-sided PSD (length
 * FFTLength/2 + 1).  Hann window + exponential averaging (ForgettingFactor
 * default 0.9) — robust and frame-rate-independent. */
matlab_mat *matlab_dsp_spectest_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int K = static_cast<int>(matlab_obj_get_f64(o, "FFTLength", 9));
    if (K < 4) K = 256;
    double alpha = matlab_obj_get_f64(o, "ForgettingFactor", 16);
    if (alpha <= 0.0 || alpha >= 1.0) alpha = 0.9;
    int Kh = K / 2 + 1;
    std::vector<double> buf = prop_vec(o, "Buf", 3);
    if (static_cast<int>(buf.size()) != K) buf.assign(static_cast<size_t>(K), 0.0);
    int widx = static_cast<int>(matlab_obj_get_f64(o, "Widx", 4));
    if (widx < 0 || widx >= K) widx = 0;
    int filled = static_cast<int>(matlab_obj_get_f64(o, "Filled", 6));
    std::vector<double> psd = prop_vec(o, "PSD", 3);
    if (static_cast<int>(psd.size()) != Kh) psd.assign(static_cast<size_t>(Kh), 0.0);

    int64_t N = mat_len(x);
    for (int64_t n = 0; n < N; ++n) {
        buf[static_cast<size_t>(widx)] = x->data[n];
        widx = (widx + 1) % K;
        if (filled < K) ++filled;
        /* Once a full window has accumulated, do a Hann-windowed DFT every
         * K/2 samples (50% overlap) and update the PSD estimate. */
        if (filled >= K && (widx % (K / 2)) == 0) {
            /* Read the K-sample window in time order. */
            std::vector<double> win(static_cast<size_t>(K), 0.0);
            for (int i = 0; i < K; ++i) {
                int j = (widx + i) % K;
                double w = 0.5 - 0.5 * cos(2.0 * M_PI * i / (K - 1));
                win[static_cast<size_t>(i)] = buf[static_cast<size_t>(j)] * w;
            }
            /* Naive DFT — K is small in typical usage and the per-frame
             * cost is amortised across many step calls. */
            for (int k = 0; k < Kh; ++k) {
                double re = 0.0, im = 0.0;
                for (int i = 0; i < K; ++i) {
                    double w = -2.0 * M_PI * k * i / K;
                    re += win[static_cast<size_t>(i)] * cos(w);
                    im += win[static_cast<size_t>(i)] * sin(w);
                }
                double p = (re * re + im * im) / K;
                psd[static_cast<size_t>(k)] =
                    alpha * psd[static_cast<size_t>(k)] + (1.0 - alpha) * p;
            }
        }
    }
    set_state(o, "Buf", 3, buf);
    matlab_obj_set_f64(o, "Widx", 4, static_cast<double>(widx));
    matlab_obj_set_f64(o, "Filled", 6, static_cast<double>(filled));
    set_state(o, "PSD", 3, psd);
    return row_mat(psd);
}

/* `buffer(x, n)` / `buffer(x, n, p)` — segment a row vector into n-sample
 * frames (columns) with p samples of overlap.  Drops the trailing partial
 * frame (`nodelay`-style behavior).  Returns an n × K matrix. */
matlab_mat *matlab_dsp_buffer(matlab_mat *x, double n_f, double p_f) {
    if (!x) return mat_alloc(0, 0);
    int n = static_cast<int>(n_f);
    int p = static_cast<int>(p_f);
    if (n < 1) n = 1;
    if (p < 0) p = 0;
    if (p >= n) p = n - 1;
    int hop = n - p;
    int64_t L = mat_len(x);
    int K = static_cast<int>((L - p) / hop);
    if (K < 0) K = 0;
    matlab_mat *Y = mat_alloc(n, K);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < n; ++i) {
            int64_t idx = static_cast<int64_t>(k) * hop + i;
            Y->data[static_cast<size_t>(i) * K + k] = (idx < L) ? x->data[idx] : 0.0;
        }
    }
    return Y;
}

/* 2-arg overload: buffer(x, n) defaults p = 0. */
matlab_mat *matlab_dsp_buffer2(matlab_mat *x, double n_f) {
    return matlab_dsp_buffer(x, n_f, 0.0);
}

/* dsp.AsyncBuffer — fixed-capacity FIFO that accepts samples (write) and
 * yields frames on read.  Two entries: write/push and read/pop.
 * State: the circular buffer + read/write indices + a count of valid
 * samples.  Capacity comes from the Capacity property. */
matlab_mat *matlab_dsp_asyncbuf_write(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int C = static_cast<int>(matlab_obj_get_f64(o, "Capacity", 8));
    if (C < 1) C = 1024;
    std::vector<double> buf = prop_vec(o, "Buf", 3);
    if (static_cast<int>(buf.size()) != C) buf.assign(static_cast<size_t>(C), 0.0);
    int widx  = static_cast<int>(matlab_obj_get_f64(o, "Widx",  4));
    int count = static_cast<int>(matlab_obj_get_f64(o, "Count", 5));
    int64_t N = mat_len(x);
    for (int64_t n = 0; n < N; ++n) {
        buf[static_cast<size_t>(widx)] = x->data[n];
        widx = (widx + 1) % C;
        if (count < C) ++count;
    }
    set_state(o, "Buf", 3, buf);
    matlab_obj_set_f64(o, "Widx",  4, static_cast<double>(widx));
    matlab_obj_set_f64(o, "Count", 5, static_cast<double>(count));
    matlab_mat *out = mat_alloc(1, 1);
    out->data[0] = static_cast<double>(count);
    return out;
}
matlab_mat *matlab_dsp_asyncbuf_read(void *obj_v, double n_req) {
    if (!obj_v) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    int C = static_cast<int>(matlab_obj_get_f64(o, "Capacity", 8));
    if (C < 1) C = 1024;
    int N = static_cast<int>(n_req);
    if (N < 0) N = 0;
    std::vector<double> buf = prop_vec(o, "Buf", 3);
    if (static_cast<int>(buf.size()) != C) buf.assign(static_cast<size_t>(C), 0.0);
    int widx  = static_cast<int>(matlab_obj_get_f64(o, "Widx",  4));
    int count = static_cast<int>(matlab_obj_get_f64(o, "Count", 5));
    int avail = (N < count) ? N : count;
    /* Read index = widx - count (mod C). */
    int ridx = ((widx - count) % C + C) % C;
    std::vector<double> y(static_cast<size_t>(avail), 0.0);
    for (int i = 0; i < avail; ++i) {
        y[static_cast<size_t>(i)] = buf[static_cast<size_t>((ridx + i) % C)];
    }
    count -= avail;
    matlab_obj_set_f64(o, "Count", 5, static_cast<double>(count));
    return row_mat(y);
}

/* ======================================================================= */
/* Tier-6 — linear-algebra SOs + carve-down filter polish                  */
/* ======================================================================= */
/* The headline Tier-6 deliverable in the roadmap — a fi-typed
 * `dsp.FIRFilter` lowering to synthesizable SystemVerilog + a cocotb SIL
 * test — is a cross-system effort (Fixed-Point lowering meets emit-SV +
 * the cocotb sequential-DUT harness).  That sits as a documented
 * follow-on; what ships in T6 below is the function-form linear-algebra
 * surface + the polish filter objects.
 */

/* dsp.LevinsonSolver step(obj, r) — Levinson-Durbin recursion over the
 * Toeplitz autocorrelation vector r (length N+1, where N = order).
 * Returns the AR(N) prediction coefficients a (1, a1, ..., aN) — same
 * convention as the shipped `levinson(r, N)`. */
matlab_mat *matlab_dsp_levinson_step(void *obj_v, matlab_mat *r) {
    if (!obj_v || !r) return mat_alloc(0, 0);
    int N = static_cast<int>(mat_len(r)) - 1;
    if (N < 1) return row_mat({1.0});
    std::vector<double> a(static_cast<size_t>(N + 1), 0.0);
    std::vector<double> aprev(static_cast<size_t>(N + 1), 0.0);
    a[0] = 1.0;
    double E = r->data[0];
    for (int i = 1; i <= N; ++i) {
        double k = -r->data[i];
        for (int j = 1; j < i; ++j) k -= a[static_cast<size_t>(j)] * r->data[i - j];
        if (fabs(E) < 1e-300) E = 1e-300;
        k /= E;
        for (int j = 0; j <= i; ++j) aprev[static_cast<size_t>(j)] = a[static_cast<size_t>(j)];
        for (int j = 1; j < i; ++j)
            a[static_cast<size_t>(j)] = aprev[static_cast<size_t>(j)] +
                                         k * aprev[static_cast<size_t>(i - j)];
        a[static_cast<size_t>(i)] = k;
        E *= (1.0 - k * k);
    }
    return row_mat(a);
}

/* dsp.NotchPeakFilter step(obj, x) — a streaming notch/peak biquad that
 * recomputes its biquad coefficients each step from the current
 * CenterFrequency + Bandwidth (so a tunable / time-varying notch is
 * possible without rebuilding the SO).  State = the 2-element biquad
 * z-line. */
matlab_mat *matlab_dsp_notchpeak_step(void *obj_v, matlab_mat *x) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    double w0 = matlab_obj_get_f64(o, "CenterFrequency", 15);
    double bw = matlab_obj_get_f64(o, "Bandwidth", 9);
    int peak = static_cast<int>(matlab_obj_get_f64(o, "IsPeak", 6));
    std::vector<double> b, a;
    notch_peak_ba(w0, bw, peak != 0, b, a);
    std::vector<double> z = prop_vec(o, "State", 5);
    double sosrow[6] = { b[0], b[1], b[2], a[0], a[1], a[2] };
    if (static_cast<int>(z.size()) < 2) z.assign(2, 0.0);
    int64_t N = mat_len(x);
    std::vector<double> y(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < N; ++n)
        y[static_cast<size_t>(n)] = biquad_sample(sosrow, x->data[n],
                                                  z[0], z[1]);
    set_state(o, "State", 5, z);
    return like_mat(y, x);
}

/* dsp.LowpassFilter / dsp.HighpassFilter step(obj, x) — combined
 * design-and-filter SOs.  The Numerator is designed from
 * (CutoffFrequency, FilterOrder) on the FIRST call (lazy setup),
 * cached as a property, and reused for every subsequent step.  State =
 * the FIR tapped-delay line. */
static matlab_mat *lphp_step_common(void *obj_v, matlab_mat *x, bool highpass) {
    if (!obj_v || !x) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    std::vector<double> b = prop_vec(o, "Numerator", 9);
    /* The MATLAB-side constructor stores a scalar 0 as the Numerator
     * placeholder ("undefined") since classdef properties must always
     * have a default value; treat any length < 2 as "design needed". */
    if (b.size() < 2) {
        /* Lazy design on first call.  Hand-coded windowed-sinc + Hamming
         * (standard fir1 lowpass) — the shipped `fir1` ships from the
         * runtime, but we hand-roll here to avoid the cross-TU dep. */
        int N = static_cast<int>(matlab_obj_get_f64(o, "FilterOrder", 11));
        double fc = matlab_obj_get_f64(o, "CutoffFrequency", 15);
        if (N < 1) N = 32;
        if (fc <= 0.0 || fc >= 1.0) fc = 0.25;
        if (N % 2 == 1) ++N;                 // ensure linear-phase Type-I
        b.resize(static_cast<size_t>(N + 1), 0.0);
        int M = N / 2;
        for (int n = 0; n <= N; ++n) {
            double sinc = (n == M) ? fc
                : sin(M_PI * fc * (n - M)) / (M_PI * (n - M));
            double w = 0.54 - 0.46 * cos(2.0 * M_PI * n / N);   // Hamming
            b[static_cast<size_t>(n)] = sinc * w;
        }
        /* Always normalise the LP design's DC gain to 1; for HP, spectral
         * inversion then gives DC=0, Nyquist=1 — robust whether or not
         * the LP cutoff is near Nyquist. */
        double g = 0.0;
        for (int n = 0; n <= N; ++n) g += b[static_cast<size_t>(n)];
        if (fabs(g) > 1e-300) for (double &c : b) c /= g;
        if (highpass) {
            for (double &c : b) c = -c;
            b[static_cast<size_t>(M)] += 1.0;
        }
        matlab_obj_set_mat(o, "Numerator", 9, row_mat(b));
    }
    std::vector<double> aa = { 1.0 };
    std::vector<double> z = prop_vec(o, "State", 5);
    std::vector<double> xv(x->data, x->data + mat_len(x));
    std::vector<double> y = tdf2_filter(b, aa, xv, z);
    set_state(o, "State", 5, z);
    return like_mat(y, x);
}
matlab_mat *matlab_dsp_lowpass_step(void *obj_v, matlab_mat *x) {
    return lphp_step_common(obj_v, x, false);
}
matlab_mat *matlab_dsp_highpass_step(void *obj_v, matlab_mat *x) {
    return lphp_step_common(obj_v, x, true);
}

/* ======================================================================= */
/* Tier-7 / Tier-8 — DSP HDL Toolbox simulation surface                    */
/* ======================================================================= */
/* What ships here: the `dsphdl.*` System Objects as floating-point
 * SIMULATION references — same compute as their `dsp.*` siblings, plus a
 * `Latency` property and `getLatency` method so the API matches the
 * MathWorks `dsphdl.*` surface.  The full deliverable in the roadmap —
 * a `dsphdl.FIRFilter` that emits clocked SystemVerilog with valid /
 * backpressure-ready / reset ports and is verified cycle-by-cycle by a
 * cocotb SIL — needs new emit-SV lane patterns (clocked datapath +
 * valid/ready handshake + fixed-point coefficient lowering through the
 * existing persistent-fi -> SV regfile path).  That sits as a documented
 * follow-on; the simulation surface lets script-level code drive the
 * `dsphdl.*` API today.
 *
 * The valid/ready streaming control is sim-only carved: step takes only
 * the data input (no validIn) — valid-gated state updates are an HDL-
 * emit-time concern.  Multi-return [dataOut, validOut] also carved.
 */

/* CORDIC math — iterative rotation/vectoring algorithms used by
 * `dsphdl.atan2`, `dsphdl.Sqrt`, `dsphdl.SineCosine`, etc.  These are
 * useful function-form on their own.  All hand-coded, ~20-iteration. */
namespace {
constexpr int kCordicIters = 20;

/* CORDIC for the simulation reference: in floating-point, the bit-exact
 * answer is libc — the iterative rotation/vectoring matters only when the
 * target is a fixed-point HDL emit (the follow-on slice).  The HW emit
 * will replace these with the real iterative impl that matches the
 * generated SV bit-for-bit.  For now, this lets script-level code that
 * reaches for `cordic_*` get the right numeric answer. */
double cordic_atan2_impl(double y, double x) { return atan2(y, x); }
double cordic_sqrt_impl(double x)            { return x < 0.0 ? 0.0 : sqrt(x); }
void cordic_sincos_impl(double t, double &s, double &c) { s = sin(t); c = cos(t); }
}  // namespace

extern "C" {

matlab_mat *matlab_dsp_cordic_atan2(matlab_mat *y, matlab_mat *x) {
    if (!y || !x) return mat_alloc(0, 0);
    int64_t N = std::min(mat_len(y), mat_len(x));
    matlab_mat *out = mat_alloc(y->rows, y->cols);
    for (int64_t i = 0; i < N; ++i)
        out->data[i] = cordic_atan2_impl(y->data[i], x->data[i]);
    return out;
}
matlab_mat *matlab_dsp_cordic_sqrt(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(x->rows, x->cols);
    int64_t N = mat_len(x);
    for (int64_t i = 0; i < N; ++i) out->data[i] = cordic_sqrt_impl(x->data[i]);
    return out;
}

/* dsphdl getLatency — read the Latency property as a typed scalar (just
 * uses the standard scalar getter; carried here for symmetry with the
 * rest of the `matlab_dsphdl_*` namespace). */
double matlab_dsphdl_latency(void *obj_v) {
    if (!obj_v) return 0.0;
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    return matlab_obj_get_f64(o, "Latency", 7);
}

/* All `dsphdl.*` filter / source / multirate step entries forward to the
 * existing `matlab_dsp_*_step` runtimes — the simulation reference is
 * identical to the `dsp.*` sibling.  The HDL-specific add-ons (latency
 * tracking, valid gating, fixed-point scaling) are carve-downs. */
matlab_mat *matlab_dsphdl_fir_step(void *o, matlab_mat *x)      { return matlab_dsp_iir_step(o, x); }
matlab_mat *matlab_dsphdl_biquad_step(void *o, matlab_mat *x)   { return matlab_dsp_sos_step(o, x); }
matlab_mat *matlab_dsphdl_sine_step(void *o)                    { return matlab_dsp_sine_step(o); }
matlab_mat *matlab_dsphdl_nco_step(void *o)                     { return matlab_dsp_nco_step(o); }
matlab_mat *matlab_dsphdl_firdecim_step(void *o, matlab_mat *x) { return matlab_dsp_firdecim_step(o, x); }
matlab_mat *matlab_dsphdl_cicdecim_step(void *o, matlab_mat *x) { return matlab_dsp_cicdecim_step(o, x); }

}  // extern "C"

/* Read the adapted Weights as a typed matrix (getWeights). */
matlab_mat *matlab_dsp_get_weights(void *obj_v) {
    if (!obj_v) return mat_alloc(0, 0);
    matlab_obj *o = reinterpret_cast<matlab_obj *>(obj_v);
    matlab_mat *w = matlab_obj_get_mat(o, "Weights", 7);
    if (!w) return mat_alloc(1, 1);
    matlab_mat *out = mat_alloc(w->rows, w->cols);
    int64_t n = mat_len(w);
    for (int64_t i = 0; i < n; ++i) out->data[i] = w->data[i];
    return out;
}

}  // extern "C"
