/* runtime_gads.cpp — Global Optimization Toolbox runtime, Tier-1.
 *
 * See docs/global_optim_toolbox_roadmap.md for the full surface.  Tier-1
 * is the three derivative-free / stochastic global solvers on a
 * box-bounded objective, each with an optional `fmincon` hybrid-polish
 * step:
 *
 *   matlab_gads_ga              — real-coded genetic algorithm
 *   matlab_gads_particleswarm   — particle swarm optimization
 *   matlab_gads_simulannealbnd  — bounded simulated annealing
 *
 * Every solver takes the objective as a single-arg scalar-returning
 * handle `double(*)(matlab_mat *)` — the exact ABI the shipped
 * `fminunc` / `fmincon` use.  The stochastic loops run over the shared
 * PRNG (`matlab_rng_state`), so `rng(seed)` makes every run
 * byte-reproducible.  The hybrid-polish step calls the shipped
 * `matlab_optim_fmincon` (no external dependency).
 *
 * Tier-1 carve-downs (deferred to Tier-6):
 *   - integer constraints (IntCon), nonlinear-constraint handles
 *   - optimoptions surface (PopulationSize / MaxGenerations / …)
 *   - exitflag / output multi-return
 *   - vectorized / parallel objective evaluation
 */

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <vector>

/* Shared global PRNG state (xorshift64) — defined in matlab_runtime.cpp;
 * shared with rand / randn / rng so seeded runs are reproducible. */
extern "C" uint64_t matlab_rng_state;

/* Shipped local NLP solver for the hybrid-polish step. */
extern "C" matlab_mat *matlab_optim_fmincon(void *obj_p, matlab_mat *x0,
                                            matlab_mat *A, matlab_mat *b,
                                            matlab_mat *Aeq, matlab_mat *beq,
                                            matlab_mat *lb, matlab_mat *ub,
                                            void *nonlcon_p);

/* matlab_obj_* accessors (Tier-6 optimoptions read) — defined in
 * runtime/matlab_runtime.cpp; same forward-decl pattern as the other
 * toolbox TUs. */
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);

/* Runtime class introspection (Tier-2 `run` dispatch) — `class_id` is the
 * compiler-assigned tag; `matlab_dbg_class_name` resolves it to the
 * classdef name through the per-program registry matlabc emits. */
extern "C" int         matlab_obj_is_known(const void *p);
extern "C" double      matlab_obj_class_id(matlab_obj *o);
extern "C" const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out);

extern "C" {

/* Objective handle ABI — identical to runtime_optim's vector objective. */
typedef double (*gads_obj_fn)(matlab_mat *);

/* ---------------------------------------------------------------- */
/* File-local helpers                                               */
/* ---------------------------------------------------------------- */

static inline double gads_uniform(void) {
    uint64_t x = matlab_rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    matlab_rng_state = x;
    return static_cast<double>(x >> 11) / static_cast<double>(1ULL << 53);
}

static inline double gads_normal(void) {
    double u1 = gads_uniform();
    double u2 = gads_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Evaluate the objective handle on a parameter vector. */
static double gads_eval(gads_obj_fn f, const std::vector<double> &v) {
    int n = static_cast<int>(v.size());
    matlab_mat *m = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) m->data[i] = v[i];
    double r = f(m);
    free(m->data);
    free(m);
    return r;
}

/* Read lb/ub into vectors of length n; defaults ±1e6 when absent. */
static void gads_bounds(matlab_mat *lb, matlab_mat *ub, int n,
                        std::vector<double> &lo, std::vector<double> &hi) {
    lo.assign(static_cast<size_t>(n), -1e6);
    hi.assign(static_cast<size_t>(n), +1e6);
    if (lb && lb->rows * lb->cols >= n)
        for (int i = 0; i < n; ++i) lo[static_cast<size_t>(i)] = lb->data[i];
    if (ub && ub->rows * ub->cols >= n)
        for (int i = 0; i < n; ++i) hi[static_cast<size_t>(i)] = ub->data[i];
}

static inline double gads_clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Hybrid polish: refine `x` with the shipped fmincon under the same
 * bounds.  Returns the polished point if it improves the objective. */
static std::vector<double> gads_hybrid(void *obj_p, const std::vector<double> &x,
                                       matlab_mat *lb, matlab_mat *ub,
                                       gads_obj_fn f) {
    int n = static_cast<int>(x.size());
    matlab_mat *x0 = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) x0->data[i] = x[i];
    matlab_mat *xp = matlab_optim_fmincon(obj_p, x0, nullptr, nullptr,
                                          nullptr, nullptr, lb, ub, nullptr);
    if (!xp || xp->rows * xp->cols < n) return x;
    std::vector<double> cand(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) cand[static_cast<size_t>(i)] = xp->data[i];
    return (gads_eval(f, cand) < gads_eval(f, x)) ? cand : x;
}

static matlab_mat *gads_col(const std::vector<double> &v) {
    matlab_mat *m = mat_alloc(static_cast<int64_t>(v.size()), 1);
    for (size_t i = 0; i < v.size(); ++i) m->data[i] = v[i];
    return m;
}

/* Run the shipped fmincon from a start point under [lb,ub]; returns the
 * resulting point as a vector (falls back to the start if it fails). */
static std::vector<double> gads_fmincon_from(void *fn_p,
                                             const std::vector<double> &x0v,
                                             matlab_mat *lb, matlab_mat *ub) {
    int n = static_cast<int>(x0v.size());
    matlab_mat *x0 = gads_col(x0v);
    matlab_mat *xp = matlab_optim_fmincon(fn_p, x0, nullptr, nullptr,
                                          nullptr, nullptr, lb, ub, nullptr);
    if (!xp || xp->rows * xp->cols < n) return x0v;
    std::vector<double> r(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) r[static_cast<size_t>(i)] = xp->data[i];
    return r;
}

/* ================================================================ */
/* Tier-5 — multiobjective optimization (gamultiobj / paretosearch)  */
/*                                                                  */
/* The objective returns a vector of `nobj` values (the vector-out   */
/* handle ABI, like lsqnonlin).  Both solvers return the Pareto set  */
/* — the non-dominated decision points — as a k×nvars matrix.        */
/* gamultiobj is NSGA-II (non-dominated sort + crowding distance);   */
/* paretosearch maintains a non-dominated archive refined by a       */
/* GPS-style poll.  Shared dominance + crowding helpers below.       */
/* ---------------------------------------------------------------- */
typedef matlab_mat *(*gads_vec_fn)(matlab_mat *);

/* Evaluate the vector objective at x → nobj-vector. */
static std::vector<double> gads_eval_vec(gads_vec_fn f, const std::vector<double> &x) {
    int n = static_cast<int>(x.size());
    matlab_mat *m = mat_alloc(n, 1);
    for (int i = 0; i < n; ++i) m->data[i] = x[i];
    matlab_mat *r = f(m);
    free(m->data); free(m);
    std::vector<double> out;
    if (r) {
        int k = static_cast<int>(r->rows * r->cols);
        out.assign(r->data, r->data + k);
        free(r->data); free(r);
    }
    return out;
}

/* Minimization dominance: a dominates b iff a ≤ b in all objectives and
 * a < b in at least one. */
static bool gads_dominates(const std::vector<double> &a, const std::vector<double> &b) {
    bool any_better = false;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] > b[i]) return false;
        if (a[i] < b[i]) any_better = true;
    }
    return any_better;
}

/* Fast non-dominated sort → rank[i] (0 = first front). */
static std::vector<int> gads_nondom_sort(const std::vector<std::vector<double>> &F) {
    int N = static_cast<int>(F.size());
    std::vector<int> rank(static_cast<size_t>(N), 0);
    std::vector<int> ndom(static_cast<size_t>(N), 0);
    std::vector<std::vector<int>> dominated(static_cast<size_t>(N));
    std::vector<int> front;
    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
            if (p == q) continue;
            if (gads_dominates(F[static_cast<size_t>(p)], F[static_cast<size_t>(q)]))
                dominated[static_cast<size_t>(p)].push_back(q);
            else if (gads_dominates(F[static_cast<size_t>(q)], F[static_cast<size_t>(p)]))
                ndom[static_cast<size_t>(p)]++;
        }
        if (ndom[static_cast<size_t>(p)] == 0) { rank[static_cast<size_t>(p)] = 0; front.push_back(p); }
    }
    int fi = 0;
    while (!front.empty()) {
        std::vector<int> next;
        for (int p : front)
            for (int q : dominated[static_cast<size_t>(p)])
                if (--ndom[static_cast<size_t>(q)] == 0) { rank[static_cast<size_t>(q)] = fi + 1; next.push_back(q); }
        fi++;
        front.swap(next);
    }
    return rank;
}

/* Crowding distance within an index set (NSGA-II): boundary points get
 * +inf, interior points the sum of normalised neighbour gaps per objective. */
static std::vector<double> gads_crowding(const std::vector<std::vector<double>> &F,
                                         const std::vector<int> &idx, int nobj) {
    int m = static_cast<int>(idx.size());
    std::vector<double> cd(static_cast<size_t>(m), 0.0);
    if (m <= 2) { for (auto &v : cd) v = 1e300; return cd; }
    for (int o = 0; o < nobj; ++o) {
        std::vector<int> order(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) order[static_cast<size_t>(i)] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return F[static_cast<size_t>(idx[static_cast<size_t>(a)])][static_cast<size_t>(o)] <
                   F[static_cast<size_t>(idx[static_cast<size_t>(b)])][static_cast<size_t>(o)];
        });
        double fmin = F[static_cast<size_t>(idx[static_cast<size_t>(order[0])])][static_cast<size_t>(o)];
        double fmax = F[static_cast<size_t>(idx[static_cast<size_t>(order[static_cast<size_t>(m - 1)])])][static_cast<size_t>(o)];
        double rng = (fmax > fmin) ? (fmax - fmin) : 1.0;
        cd[static_cast<size_t>(order[0])] = 1e300;
        cd[static_cast<size_t>(order[static_cast<size_t>(m - 1)])] = 1e300;
        for (int i = 1; i < m - 1; ++i)
            cd[static_cast<size_t>(order[static_cast<size_t>(i)])] +=
                (F[static_cast<size_t>(idx[static_cast<size_t>(order[static_cast<size_t>(i + 1)])])][static_cast<size_t>(o)] -
                 F[static_cast<size_t>(idx[static_cast<size_t>(order[static_cast<size_t>(i - 1)])])][static_cast<size_t>(o)]) / rng;
    }
    return cd;
}

/* Pack a list of decision vectors into a k×nvars row-major matrix. */
static matlab_mat *gads_pareto_mat(const std::vector<std::vector<double>> &X, int n) {
    int k = static_cast<int>(X.size());
    if (k == 0) return mat_alloc(0, 0);
    matlab_mat *M = mat_alloc(k, n);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j) M->data[i * n + j] = X[static_cast<size_t>(i)][static_cast<size_t>(j)];
    return M;
}

matlab_mat *matlab_gads_gamultiobj(void *fn_p, double nvars_d,
                                   matlab_mat *lb, matlab_mat *ub) {
    if (!fn_p) return mat_alloc(0, 0);
    gads_vec_fn f = reinterpret_cast<gads_vec_fn>(fn_p);
    int n = static_cast<int>(nvars_d);
    if (n < 1) return mat_alloc(0, 0);
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);
    auto span = [&](int i) { double s = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
                             return (s < 1e12 && s > 0) ? s : 10.0; };
    auto lobnd = [&](int i) { return (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)] < 1e12)
                                     ? lo[static_cast<size_t>(i)] : -5.0; };
    auto rand_pt = [&]() { std::vector<double> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = lobnd(i) + gads_uniform() * span(i);
        return x; };

    int pop = std::min(120, std::max(40, 25 * n));
    int gens = 100;
    std::vector<std::vector<double>> X(static_cast<size_t>(pop));
    std::vector<std::vector<double>> F(static_cast<size_t>(pop));
    for (int p = 0; p < pop; ++p) { X[static_cast<size_t>(p)] = rand_pt();
                                    F[static_cast<size_t>(p)] = gads_eval_vec(f, X[static_cast<size_t>(p)]); }
    int nobj = F[0].empty() ? 1 : static_cast<int>(F[0].size());

    auto better = [&](int a, int b, const std::vector<int>& rank, const std::vector<double>& cd) {
        if (rank[static_cast<size_t>(a)] != rank[static_cast<size_t>(b)])
            return rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)];
        return cd[static_cast<size_t>(a)] > cd[static_cast<size_t>(b)];
    };
    for (int g = 0; g < gens; ++g) {
        /* Rank + crowding of current population for selection. */
        std::vector<int> rank = gads_nondom_sort(F);
        int maxr = 0; for (int r : rank) maxr = std::max(maxr, r);
        std::vector<double> cd(static_cast<size_t>(pop), 0.0);
        for (int r = 0; r <= maxr; ++r) {
            std::vector<int> idx; for (int i = 0; i < pop; ++i) if (rank[static_cast<size_t>(i)] == r) idx.push_back(i);
            std::vector<double> c = gads_crowding(F, idx, nobj);
            for (size_t t = 0; t < idx.size(); ++t) cd[static_cast<size_t>(idx[t])] = c[t];
        }
        /* Offspring via crowded tournament + BLX-α crossover + mutation. */
        std::vector<std::vector<double>> Q(static_cast<size_t>(pop));
        std::vector<std::vector<double>> QF(static_cast<size_t>(pop));
        auto pick = [&]() -> int { int a = static_cast<int>(gads_uniform()*pop);
            int b = static_cast<int>(gads_uniform()*pop); if (a>=pop)a=pop-1; if(b>=pop)b=pop-1;
            return better(a,b,rank,cd) ? a : b; };
        for (int c = 0; c < pop; ++c) {
            const std::vector<double> &pa = X[static_cast<size_t>(pick())];
            const std::vector<double> &pb = X[static_cast<size_t>(pick())];
            std::vector<double> ch(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                double xa = pa[static_cast<size_t>(i)], xb = pb[static_cast<size_t>(i)];
                double cmin = std::min(xa,xb), cmax = std::max(xa,xb), d = cmax - cmin;
                double v = (cmin - 0.5*d) + gads_uniform()*(d*2.0);
                if (gads_uniform() < 0.1) v += 0.1 * span(i) * gads_normal();
                ch[static_cast<size_t>(i)] = gads_clamp(v, lo[static_cast<size_t>(i)], hi[static_cast<size_t>(i)]);
            }
            Q[static_cast<size_t>(c)] = ch; QF[static_cast<size_t>(c)] = gads_eval_vec(f, ch);
        }
        /* Elitist combine R = P ∪ Q, sort, take best `pop` by (rank, crowding). */
        std::vector<std::vector<double>> RX = X; RX.insert(RX.end(), Q.begin(), Q.end());
        std::vector<std::vector<double>> RF = F; RF.insert(RF.end(), QF.begin(), QF.end());
        std::vector<int> rrank = gads_nondom_sort(RF);
        int rmax = 0; for (int r : rrank) rmax = std::max(rmax, r);
        std::vector<std::vector<double>> nX, nF;
        for (int r = 0; r <= rmax && static_cast<int>(nX.size()) < pop; ++r) {
            std::vector<int> idx; for (int i = 0; i < static_cast<int>(RX.size()); ++i) if (rrank[static_cast<size_t>(i)]==r) idx.push_back(i);
            if (static_cast<int>(nX.size()) + static_cast<int>(idx.size()) <= pop) {
                for (int i : idx) { nX.push_back(RX[static_cast<size_t>(i)]); nF.push_back(RF[static_cast<size_t>(i)]); }
            } else {
                std::vector<double> c = gads_crowding(RF, idx, nobj);
                std::vector<int> ord(idx.size()); for (size_t t=0;t<idx.size();++t) ord[t]=static_cast<int>(t);
                std::sort(ord.begin(), ord.end(), [&](int a,int b){ return c[static_cast<size_t>(a)] > c[static_cast<size_t>(b)]; });
                for (int t = 0; static_cast<int>(nX.size()) < pop && t < static_cast<int>(ord.size()); ++t) {
                    int i = idx[static_cast<size_t>(ord[static_cast<size_t>(t)])];
                    nX.push_back(RX[static_cast<size_t>(i)]); nF.push_back(RF[static_cast<size_t>(i)]);
                }
            }
        }
        X.swap(nX); F.swap(nF);
    }
    /* Return the first non-dominated front. */
    std::vector<int> rank = gads_nondom_sort(F);
    std::vector<std::vector<double>> front;
    for (int i = 0; i < static_cast<int>(X.size()); ++i)
        if (rank[static_cast<size_t>(i)] == 0) front.push_back(X[static_cast<size_t>(i)]);
    return gads_pareto_mat(front, n);
}

matlab_mat *matlab_gads_paretosearch(void *fn_p, double nvars_d,
                                     matlab_mat *lb, matlab_mat *ub) {
    if (!fn_p) return mat_alloc(0, 0);
    gads_vec_fn f = reinterpret_cast<gads_vec_fn>(fn_p);
    int n = static_cast<int>(nvars_d);
    if (n < 1) return mat_alloc(0, 0);
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);
    auto span = [&](int i) { double s = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
                             return (s < 1e12 && s > 0) ? s : 10.0; };
    auto lobnd = [&](int i) { return (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)] < 1e12)
                                     ? lo[static_cast<size_t>(i)] : -5.0; };
    auto rand_pt = [&]() { std::vector<double> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = lobnd(i) + gads_uniform() * span(i);
        return x; };

    /* Archive of non-dominated points. */
    std::vector<std::vector<double>> AX, AF;
    auto try_add = [&](const std::vector<double> &x) {
        std::vector<double> fx = gads_eval_vec(f, x);
        for (size_t i = 0; i < AF.size(); ++i)
            if (gads_dominates(AF[i], fx)) return;            /* dominated → reject */
        /* Remove archive points this one dominates. */
        std::vector<std::vector<double>> nX, nF;
        for (size_t i = 0; i < AF.size(); ++i)
            if (!gads_dominates(fx, AF[i])) { nX.push_back(AX[i]); nF.push_back(AF[i]); }
        AX.swap(nX); AF.swap(nF);
        AX.push_back(x); AF.push_back(fx);
    };

    /* Prune the archive to `cap` best-spread points by crowding distance.
     * For a continuous Pareto front the non-dominated set is unbounded, so
     * this MUST run during the loop (not just at the end) — otherwise
     * try_add's per-call O(archive) cost blows up to O(archive²). */
    const int cap = 80;
    auto prune = [&]() {
        if (static_cast<int>(AX.size()) <= cap) return;
        int nobj = AF.empty() ? 1 : static_cast<int>(AF[0].size());
        std::vector<int> idx(AX.size()); for (size_t t=0;t<AX.size();++t) idx[t]=static_cast<int>(t);
        std::vector<double> c = gads_crowding(AF, idx, nobj);
        std::vector<int> ord(idx.size()); for (size_t t=0;t<idx.size();++t) ord[t]=static_cast<int>(t);
        std::sort(ord.begin(), ord.end(), [&](int a,int b){ return c[static_cast<size_t>(a)] > c[static_cast<size_t>(b)]; });
        std::vector<std::vector<double>> kX; std::vector<std::vector<double>> kF;
        for (int t = 0; t < cap; ++t) { kX.push_back(AX[static_cast<size_t>(ord[static_cast<size_t>(t)])]);
                                        kF.push_back(AF[static_cast<size_t>(ord[static_cast<size_t>(t)])]); }
        AX.swap(kX); AF.swap(kF);
    };

    int nseed = std::min(200, std::max(60, 40 * n));
    for (int s = 0; s < nseed; ++s) try_add(rand_pt());
    prune();
    /* Refine: GPS-style poll around each archive point at shrinking radius. */
    double delta = 0.2;
    for (int it = 0; it < 30; ++it) {
        std::vector<std::vector<double>> base = AX;
        for (const auto &x : base) {
            for (int i = 0; i < n; ++i) {
                for (int sgn = -1; sgn <= 1; sgn += 2) {
                    std::vector<double> xt = x;
                    xt[static_cast<size_t>(i)] = gads_clamp(
                        x[static_cast<size_t>(i)] + sgn * delta * span(i),
                        lo[static_cast<size_t>(i)], hi[static_cast<size_t>(i)]);
                    try_add(xt);
                }
            }
        }
        prune();                 /* keep the archive bounded each iteration */
        delta *= 0.8;
    }
    return gads_pareto_mat(AX, n);
}

/* ================================================================ */
/* Tier-4 — surrogate optimization (surrogateopt)                   */
/*                                                                  */
/* The sample-efficient global solver for expensive objectives: fit a */
/* cubic radial-basis-function surrogate to the evaluated points,    */
/* then choose the next sample by a merit score that trades the      */
/* surrogate prediction against distance from existing samples       */
/* (exploration), cycling the weight.  RBF coefficients come from the */
/* shipped `matlab_mldivide_mm` on the (N+n+1)-square interpolation   */
/* system (cubic φ(r)=r³ + linear polynomial tail).  A final fmincon */
/* polish lands the exact optimum (Tier-4 simplification — the        */
/* no-polish minimal-evaluation mode is a Tier-6 option).            */
/* ---------------------------------------------------------------- */
static double gads_cube(double r) { return r * r * r; }

static double gads_dist(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; s += d * d; }
    return sqrt(s);
}

/* Solve the cubic-RBF + linear-tail interpolation system; returns the
 * (N+n+1) coefficient vector (or empty on failure). */
static std::vector<double> gads_rbf_fit(const std::vector<std::vector<double>> &X,
                                        const std::vector<double> &F, int n) {
    int N = static_cast<int>(X.size());
    int dim = N + n + 1;
    matlab_mat *A = mat_alloc(dim, dim);     /* calloc'd → zeros */
    matlab_mat *b = mat_alloc(dim, 1);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            A->data[i * dim + j] = gads_cube(gads_dist(X[static_cast<size_t>(i)],
                                                       X[static_cast<size_t>(j)]));
        A->data[i * dim + N] = 1.0;          /* polynomial tail: const + linear */
        A->data[N * dim + i] = 1.0;
        for (int k = 0; k < n; ++k) {
            A->data[i * dim + (N + 1 + k)] = X[static_cast<size_t>(i)][static_cast<size_t>(k)];
            A->data[(N + 1 + k) * dim + i] = X[static_cast<size_t>(i)][static_cast<size_t>(k)];
        }
        b->data[i] = F[static_cast<size_t>(i)];
    }
    /* Tiny ridge on the diagonal keeps the conditionally-PD system
     * solvable when sample points cluster. */
    for (int i = 0; i < dim; ++i) A->data[i * dim + i] += 1e-9;
    matlab_mat *c = matlab_mldivide_mm(A, b);
    std::vector<double> coef;
    if (c && c->rows * c->cols >= dim)
        coef.assign(c->data, c->data + dim);
    return coef;
}

static double gads_rbf_eval(const std::vector<double> &coef,
                            const std::vector<std::vector<double>> &X,
                            int n, const std::vector<double> &y) {
    int N = static_cast<int>(X.size());
    if (static_cast<int>(coef.size()) < N + n + 1) return 0.0;
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += coef[static_cast<size_t>(i)] * gads_cube(gads_dist(y, X[static_cast<size_t>(i)]));
    s += coef[static_cast<size_t>(N)];
    for (int k = 0; k < n; ++k) s += coef[static_cast<size_t>(N + 1 + k)] * y[static_cast<size_t>(k)];
    return s;
}

matlab_mat *matlab_gads_surrogateopt(void *fn_p, matlab_mat *lb, matlab_mat *ub,
                                     double hybrid) {
    if (!fn_p || !lb || !ub) return mat_alloc(0, 0);
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(fn_p);
    int n = static_cast<int>(lb->rows * lb->cols);
    if (n < 1 || ub->rows * ub->cols < n) return mat_alloc(0, 0);
    std::vector<double> lo(lb->data, lb->data + n);
    std::vector<double> hi(ub->data, ub->data + n);
    auto rand_pt = [&]() {
        std::vector<double> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = lo[static_cast<size_t>(i)] +
                gads_uniform() * (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)]);
        return x;
    };

    int budget = 60 + 30 * n;                 /* function-evaluation budget */
    int ninit = 2 * (n + 1);
    std::vector<std::vector<double>> X;
    std::vector<double> F;
    std::vector<double> xbest;
    double fbest = 1e300;
    for (int i = 0; i < ninit; ++i) {
        std::vector<double> x = rand_pt();
        double fx = gads_eval(f, x);
        X.push_back(x); F.push_back(fx);
        if (fx < fbest) { fbest = fx; xbest = x; }
    }

    const double wcycle[4] = {0.3, 0.5, 0.7, 0.95};
    double radius = 0.2;
    int fail = 0;
    int evals = ninit;
    int it = 0;
    while (evals < budget) {
        std::vector<double> coef = gads_rbf_fit(X, F, n);
        if (coef.empty()) break;
        /* Candidate pool: perturbations around the incumbent + global samples. */
        int M = 100 + 20 * n;
        std::vector<std::vector<double>> cand;
        cand.reserve(static_cast<size_t>(M));
        for (int c = 0; c < M; ++c) {
            std::vector<double> x;
            if (c % 2 == 0) {              /* local perturbation of best */
                x = xbest;
                for (int i = 0; i < n; ++i) {
                    double span = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
                    x[static_cast<size_t>(i)] = gads_clamp(
                        x[static_cast<size_t>(i)] + radius * span * gads_normal(),
                        lo[static_cast<size_t>(i)], hi[static_cast<size_t>(i)]);
                }
            } else {                       /* global exploration */
                x = rand_pt();
            }
            cand.push_back(std::move(x));
        }
        /* Score: surrogate value + distance-to-samples, scaled to [0,1]. */
        std::vector<double> sval(static_cast<size_t>(M)), dval(static_cast<size_t>(M));
        double smin = 1e300, smax = -1e300, dmin = 1e300, dmax = -1e300;
        for (int c = 0; c < M; ++c) {
            sval[static_cast<size_t>(c)] = gads_rbf_eval(coef, X, n, cand[static_cast<size_t>(c)]);
            double dmn = 1e300;
            for (const auto &xs : X) dmn = std::min(dmn, gads_dist(cand[static_cast<size_t>(c)], xs));
            dval[static_cast<size_t>(c)] = dmn;
            smin = std::min(smin, sval[static_cast<size_t>(c)]); smax = std::max(smax, sval[static_cast<size_t>(c)]);
            dmin = std::min(dmin, dmn); dmax = std::max(dmax, dmn);
        }
        double w = wcycle[it % 4];
        int bestc = 0; double bestmerit = 1e300;
        for (int c = 0; c < M; ++c) {
            double ss = (smax > smin) ? (sval[static_cast<size_t>(c)] - smin) / (smax - smin) : 0.0;
            double dd = (dmax > dmin) ? (dval[static_cast<size_t>(c)] - dmin) / (dmax - dmin) : 0.0;
            double merit = w * ss + (1.0 - w) * (1.0 - dd);   /* low surrogate + far = good */
            if (merit < bestmerit) { bestmerit = merit; bestc = c; }
        }
        std::vector<double> xc = cand[static_cast<size_t>(bestc)];
        double fc = gads_eval(f, xc);
        X.push_back(xc); F.push_back(fc); ++evals; ++it;
        if (fc < fbest - 1e-12) { fbest = fc; xbest = xc; fail = 0; radius = std::min(0.4, radius * 1.4); }
        else { if (++fail >= 3) { radius = std::max(0.01, radius * 0.5); fail = 0; } }
    }

    if (xbest.empty()) return mat_alloc(0, 0);
    std::vector<double> xr = xbest;
    if (hybrid != 0.0) xr = gads_hybrid(fn_p, xbest, lb, ub, f);
    return gads_col(xr);
}

/* ================================================================ */
/* Tier-3 — deterministic direct search (patternsearch, GPS)        */
/*                                                                  */
/* Generalized Pattern Search with the 2N positive-spanning basis   */
/* {±e_i}: at each iteration poll x ± Δ·e_i (complete poll, take the */
/* best), move on success and expand the mesh (Δ←2Δ), contract on    */
/* failure (Δ←Δ/2), until Δ < tol.  Derivative-free and fully        */
/* deterministic (no PRNG) — robust on nonsmooth / discontinuous     */
/* objectives where gradient solvers fail.  No hybrid polish: the    */
/* mesh refinement IS the convergence (a gradient `fmincon` polish   */
/* would be inappropriate on the nonsmooth objectives this targets). */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_gads_patternsearch(void *fn_p, matlab_mat *x0,
                                      matlab_mat *lb, matlab_mat *ub) {
    if (!fn_p || !x0) return mat_alloc(0, 0);
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(fn_p);
    int n = static_cast<int>(x0->rows * x0->cols);
    if (n < 1) return mat_alloc(0, 0);
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);

    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = gads_clamp(x0->data[i], lo[static_cast<size_t>(i)],
                                               hi[static_cast<size_t>(i)]);
    double fx = gads_eval(f, x);

    double delta = 1.0;                 /* initial mesh size */
    const double tol = 1e-9, delta_max = 1e6;
    const int max_iter = 2000;
    for (int it = 0; it < max_iter && delta > tol; ++it) {
        std::vector<double> best_x = x;
        double best_f = fx;
        bool improved = false;
        /* Complete poll over the 2N basis directions ±e_i. */
        for (int i = 0; i < n; ++i) {
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                std::vector<double> xt = x;
                xt[static_cast<size_t>(i)] = gads_clamp(
                    x[static_cast<size_t>(i)] + sgn * delta,
                    lo[static_cast<size_t>(i)], hi[static_cast<size_t>(i)]);
                if (xt[static_cast<size_t>(i)] == x[static_cast<size_t>(i)]) continue;  /* clamped to no-op */
                double ft = gads_eval(f, xt);
                if (ft < best_f) { best_f = ft; best_x = xt; improved = true; }
            }
        }
        if (improved) {
            x = best_x; fx = best_f;
            delta *= 2.0; if (delta > delta_max) delta = delta_max;   /* expand */
        } else {
            delta *= 0.5;                                             /* contract */
        }
    }
    return gads_col(x);
}

/* ================================================================ */
/* Tier-2 — multi-start meta-solvers (MultiStart / GlobalSearch)    */
/*                                                                  */
/* createOptimProblem stashes the objective handle + x0 / lb / ub   */
/* into a thread-local context (function handles can't round-trip   */
/* through the obj property bag — the nlmpc precedent); run() then   */
/* reads it back.  Single active problem at a time (Tier-2 simpl.).  */
/* ---------------------------------------------------------------- */
struct GadsProblem {
    void *fn = nullptr;
    std::vector<double> x0, lb, ub;
    bool has_bounds = false;
};
static thread_local GadsProblem g_gads_prob;

matlab_mat *matlab_gads_make_problem(void *fn_p, matlab_mat *x0,
                                     matlab_mat *lb, matlab_mat *ub) {
    GadsProblem &p = g_gads_prob;
    p.fn = fn_p;
    p.x0.clear(); p.lb.clear(); p.ub.clear();
    if (x0) for (int64_t i = 0; i < x0->rows * x0->cols; ++i) p.x0.push_back(x0->data[i]);
    int n = static_cast<int>(p.x0.size());
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);
    p.lb = lo; p.ub = hi;
    p.has_bounds = (lb && lb->rows * lb->cols >= n) ||
                   (ub && ub->rows * ub->cols >= n);
    return mat_alloc(1, 1);   /* problem marker (run reads the thread-local) */
}

/* Best-of-k fmincon restarts: start 0 = x0, starts 1..k-1 random in
 * [lb,ub].  Returns the lowest-objective local solution found. */
matlab_mat *matlab_gads_multistart(double k_d) {
    GadsProblem &p = g_gads_prob;
    if (!p.fn || p.x0.empty()) return mat_alloc(0, 0);
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(p.fn);
    int n = static_cast<int>(p.x0.size());
    int k = static_cast<int>(k_d);
    if (k < 1) k = 1;
    matlab_mat *lbm = gads_col(p.lb);
    matlab_mat *ubm = gads_col(p.ub);
    std::vector<double> xbest = gads_fmincon_from(p.fn, p.x0, lbm, ubm);
    double fbest = gads_eval(f, xbest);
    for (int s = 1; s < k; ++s) {
        std::vector<double> x0s(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            double lo = p.lb[static_cast<size_t>(i)];
            double hi = p.ub[static_cast<size_t>(i)];
            /* Finite-span fallback when bounds were not supplied. */
            if (hi - lo >= 1e12) { lo = -5.0; hi = 5.0; }
            x0s[static_cast<size_t>(i)] = lo + gads_uniform() * (hi - lo);
        }
        std::vector<double> xs = gads_fmincon_from(p.fn, x0s, lbm, ubm);
        double fs = gads_eval(f, xs);
        if (fs < fbest) { fbest = fs; xbest = xs; }
    }
    return gads_col(xbest);
}

/* GlobalSearch: scatter a sample of trial points, score them, then run
 * fmincon from x0 + the most promising trials (a pragmatic OQNLP). */
matlab_mat *matlab_gads_globalsearch(void) {
    GadsProblem &p = g_gads_prob;
    if (!p.fn || p.x0.empty()) return mat_alloc(0, 0);
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(p.fn);
    int n = static_cast<int>(p.x0.size());
    matlab_mat *lbm = gads_col(p.lb);
    matlab_mat *ubm = gads_col(p.ub);

    /* Stage 1: scatter-sample + score. */
    const int nsample = 200, nrefine = 8;
    std::vector<std::pair<double, std::vector<double>>> trials;
    trials.reserve(static_cast<size_t>(nsample));
    auto span = [&](int i) {
        double s = p.ub[static_cast<size_t>(i)] - p.lb[static_cast<size_t>(i)];
        return (s < 1e12 && s > 0) ? s : 10.0;
    };
    auto lobnd = [&](int i) {
        return (p.ub[static_cast<size_t>(i)] - p.lb[static_cast<size_t>(i)] < 1e12)
                   ? p.lb[static_cast<size_t>(i)] : -5.0;
    };
    for (int t = 0; t < nsample; ++t) {
        std::vector<double> xt(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) xt[static_cast<size_t>(i)] = lobnd(i) + gads_uniform() * span(i);
        trials.emplace_back(gads_eval(f, xt), std::move(xt));
    }
    std::sort(trials.begin(), trials.end(),
              [](const std::pair<double, std::vector<double>> &a,
                 const std::pair<double, std::vector<double>> &b) {
                  return a.first < b.first;
              });
    /* Stage 2: local solve from x0 + the top-nrefine trial points. */
    std::vector<double> xbest = gads_fmincon_from(p.fn, p.x0, lbm, ubm);
    double fbest = gads_eval(f, xbest);
    for (int r = 0; r < nrefine && r < static_cast<int>(trials.size()); ++r) {
        std::vector<double> xs = gads_fmincon_from(p.fn, trials[static_cast<size_t>(r)].second, lbm, ubm);
        double fs = gads_eval(f, xs);
        if (fs < fbest) { fbest = fs; xbest = xs; }
    }
    return gads_col(xbest);
}

/* run(solver, problem [, k]) — runtime-dispatched.  The Lowering arm
 * forwards the solver object + restart count `k`; we read the solver's
 * class via the compiler-emitted class registry and branch to the
 * MultiStart / GlobalSearch loop.  Reading the class at runtime (rather
 * than from a Sema-pinned type) lets `run` work in the line-by-line REPL,
 * where cross-line class pinning is not retained.  The objective + bounds
 * still ride in the thread-local set by createOptimProblem. */
matlab_mat *matlab_gads_run(void *solver, double k_d) {
    bool is_globalsearch = false;
    if (solver && matlab_obj_is_known(solver)) {
        matlab_obj *o = reinterpret_cast<matlab_obj *>(solver);
        int32_t cid = static_cast<int32_t>(matlab_obj_class_id(o));
        int64_t len = 0;
        const char *cn = matlab_dbg_class_name(cid, &len);
        if (cn && len == 12 && std::strncmp(cn, "GlobalSearch", 12) == 0)
            is_globalsearch = true;
    }
    return is_globalsearch ? matlab_gads_globalsearch()
                           : matlab_gads_multistart(k_d);
}

/* ================================================================ */
/* matlab_gads_ga — real-coded genetic algorithm                    */
/*                                                                  */
/* ga(fun, nvars, lb, ub, hybrid):  tournament selection, blend     */
/* (BLX-α) crossover, Gaussian mutation, elitism.  Returns the best */
/* individual as an nvars×1 column.                                 */
/* ---------------------------------------------------------------- */
/* Shared GA core (Tier-1 + Tier-6).  `pop`/`gens` are resolved by the
 * caller (defaults or optimoptions overrides); `isint` marks integer
 * variables (rounded each generation — the IntCon mixed-integer path);
 * `do_hybrid` runs the fmincon polish, but only when there are no
 * integer variables (a continuous polish is meaningless for them). */
static matlab_mat *gads_ga_core(void *fn_p, int n, matlab_mat *lb, matlab_mat *ub,
                                int pop, int gens, bool do_hybrid,
                                const std::vector<char> &isint) {
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(fn_p);
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);
    bool has_int = false;
    for (int i = 0; i < n; ++i) if (i < static_cast<int>(isint.size()) && isint[static_cast<size_t>(i)]) has_int = true;

    int elite = std::max(1, pop / 20);
    const double mut_rate = 0.1, mut_scale0 = 0.5, alpha = 0.5;
    auto span = [&](int i) {
        double s = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
        return (s < 1e12 && s > 0) ? s : 10.0;
    };
    auto lobnd = [&](int i) {
        return (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)] < 1e12)
                   ? lo[static_cast<size_t>(i)] : -5.0;
    };
    /* Round integer variables to the nearest feasible integer. */
    auto round_int = [&](std::vector<double> &x) {
        if (!has_int) return;
        for (int i = 0; i < n; ++i)
            if (i < static_cast<int>(isint.size()) && isint[static_cast<size_t>(i)]) {
                double r = round(x[static_cast<size_t>(i)]);
                double ilo = ceil(lo[static_cast<size_t>(i)] - 1e-9);
                double ihi = floor(hi[static_cast<size_t>(i)] + 1e-9);
                x[static_cast<size_t>(i)] = gads_clamp(r, ilo, ihi);
            }
    };

    std::vector<std::vector<double>> P(static_cast<size_t>(pop),
                                       std::vector<double>(static_cast<size_t>(n)));
    std::vector<double> fit(static_cast<size_t>(pop));
    for (int p = 0; p < pop; ++p) {
        for (int i = 0; i < n; ++i)
            P[static_cast<size_t>(p)][static_cast<size_t>(i)] = lobnd(i) + gads_uniform() * span(i);
        round_int(P[static_cast<size_t>(p)]);
        fit[static_cast<size_t>(p)] = gads_eval(f, P[static_cast<size_t>(p)]);
    }

    std::vector<int> order(static_cast<size_t>(pop));
    for (int g = 0; g < gens; ++g) {
        for (int p = 0; p < pop; ++p) order[static_cast<size_t>(p)] = p;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return fit[static_cast<size_t>(a)] < fit[static_cast<size_t>(b)]; });
        std::vector<std::vector<double>> N(static_cast<size_t>(pop));
        for (int e = 0; e < elite; ++e)
            N[static_cast<size_t>(e)] = P[static_cast<size_t>(order[static_cast<size_t>(e)])];
        double mut_scale = mut_scale0 * (1.0 - static_cast<double>(g) / gens);
        for (int c = elite; c < pop; ++c) {
            auto pick = [&]() -> const std::vector<double>& {
                int a = static_cast<int>(gads_uniform() * pop);
                int b = static_cast<int>(gads_uniform() * pop);
                if (a >= pop) a = pop - 1; if (b >= pop) b = pop - 1;
                return (fit[static_cast<size_t>(a)] < fit[static_cast<size_t>(b)])
                           ? P[static_cast<size_t>(a)] : P[static_cast<size_t>(b)];
            };
            const std::vector<double>& pa = pick();
            const std::vector<double>& pb = pick();
            std::vector<double> child(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                double xa = pa[static_cast<size_t>(i)], xb = pb[static_cast<size_t>(i)];
                double cmin = std::min(xa, xb), cmax = std::max(xa, xb);
                double d = cmax - cmin;
                double v = (cmin - alpha * d) + gads_uniform() * (d * (1 + 2 * alpha));
                if (gads_uniform() < mut_rate) v += mut_scale * span(i) * gads_normal();
                child[static_cast<size_t>(i)] = gads_clamp(v, lo[static_cast<size_t>(i)], hi[static_cast<size_t>(i)]);
            }
            round_int(child);
            N[static_cast<size_t>(c)] = std::move(child);
        }
        P.swap(N);
        for (int p = 0; p < pop; ++p)
            fit[static_cast<size_t>(p)] = gads_eval(f, P[static_cast<size_t>(p)]);
    }
    int best = 0;
    for (int p = 1; p < pop; ++p)
        if (fit[static_cast<size_t>(p)] < fit[static_cast<size_t>(best)]) best = p;
    std::vector<double> x = P[static_cast<size_t>(best)];
    /* fmincon polish only for continuous problems (integer vars excluded). */
    if (do_hybrid && !has_int) x = gads_hybrid(fn_p, x, lb, ub, f);
    round_int(x);
    return gads_col(x);
}

/* Default GA pop / gens (MathWorks-ish). */
static void gads_ga_defaults(int n, int *pop, int *gens) {
    *pop = std::min(200, std::max(20, 10 * n));
    *gens = std::min(400, 100 * n);
}

matlab_mat *matlab_gads_ga(void *fn_p, double nvars_d,
                           matlab_mat *lb, matlab_mat *ub, double hybrid) {
    if (!fn_p) return mat_alloc(0, 0);
    int n = static_cast<int>(nvars_d);
    if (n < 1) return mat_alloc(0, 0);
    int pop, gens; gads_ga_defaults(n, &pop, &gens);
    return gads_ga_core(fn_p, n, lb, ub, pop, gens, hybrid != 0.0, std::vector<char>());
}

/* Tier-6 — ga with an optimoptions object: PopulationSize / MaxGenerations
 * / IntCon read from the carrier obj (sentinel −1 = "use default"). */
matlab_mat *matlab_gads_ga_opts(void *fn_p, double nvars_d, matlab_mat *lb,
                                matlab_mat *ub, double hybrid, void *opts_v) {
    if (!fn_p) return mat_alloc(0, 0);
    int n = static_cast<int>(nvars_d);
    if (n < 1) return mat_alloc(0, 0);
    int pop, gens; gads_ga_defaults(n, &pop, &gens);
    bool do_hybrid = (hybrid != 0.0);
    std::vector<char> isint;
    if (opts_v) {
        matlab_obj *o = reinterpret_cast<matlab_obj *>(opts_v);
        double ps = matlab_obj_get_f64(o, "PopulationSize", 14);
        if (ps >= 1) pop = static_cast<int>(ps);
        double mg = matlab_obj_get_f64(o, "MaxGenerations", 14);
        if (mg >= 1) gens = static_cast<int>(mg);
        matlab_mat *ic = matlab_obj_get_mat(o, "IntCon", 6);
        if (ic && ic->rows * ic->cols > 0) {
            isint.assign(static_cast<size_t>(n), 0);
            for (int64_t k = 0; k < ic->rows * ic->cols; ++k) {
                int idx = static_cast<int>(ic->data[k]);
                if (idx >= 1 && idx <= n) isint[static_cast<size_t>(idx - 1)] = 1;
            }
        }
    }
    return gads_ga_core(fn_p, n, lb, ub, pop, gens, do_hybrid, isint);
}

/* ================================================================ */
/* matlab_gads_particleswarm — particle swarm optimization          */
/*                                                                  */
/* Standard PSO: inertia-weight velocity update                     */
/*   v ← w·v + c₁·r₁·(pbest−x) + c₂·r₂·(gbest−x)                    */
/* with bound reflection.  Returns the global-best column.          */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_gads_particleswarm(void *fn_p, double nvars_d,
                                      matlab_mat *lb, matlab_mat *ub,
                                      double hybrid) {
    if (!fn_p) return mat_alloc(0, 0);
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(fn_p);
    int n = static_cast<int>(nvars_d);
    if (n < 1) return mat_alloc(0, 0);
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);

    int swarm = std::min(100, std::max(20, 10 * n));
    int iters = 200 * n; if (iters > 600) iters = 600;
    const double w = 0.729, c1 = 1.49, c2 = 1.49;   /* Clerc-Kennedy constriction */

    auto span = [&](int i) {
        double s = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
        return (s < 1e12 && s > 0) ? s : 10.0;
    };
    auto lobnd = [&](int i) {
        return (hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)] < 1e12)
                   ? lo[static_cast<size_t>(i)] : -5.0;
    };
    std::vector<std::vector<double>> X(static_cast<size_t>(swarm),
                                       std::vector<double>(static_cast<size_t>(n)));
    std::vector<std::vector<double>> V(static_cast<size_t>(swarm),
                                       std::vector<double>(static_cast<size_t>(n), 0.0));
    std::vector<std::vector<double>> Pb = X;
    std::vector<double> fpb(static_cast<size_t>(swarm));
    std::vector<double> gbest(static_cast<size_t>(n));
    double fg = 1e300;
    for (int s = 0; s < swarm; ++s) {
        for (int i = 0; i < n; ++i) {
            X[static_cast<size_t>(s)][static_cast<size_t>(i)] = lobnd(i) + gads_uniform() * span(i);
            V[static_cast<size_t>(s)][static_cast<size_t>(i)] = (gads_uniform() - 0.5) * span(i);
        }
        Pb[static_cast<size_t>(s)] = X[static_cast<size_t>(s)];
        fpb[static_cast<size_t>(s)] = gads_eval(f, X[static_cast<size_t>(s)]);
        if (fpb[static_cast<size_t>(s)] < fg) { fg = fpb[static_cast<size_t>(s)]; gbest = X[static_cast<size_t>(s)]; }
    }
    for (int it = 0; it < iters; ++it) {
        for (int s = 0; s < swarm; ++s) {
            for (int i = 0; i < n; ++i) {
                double r1 = gads_uniform(), r2 = gads_uniform();
                double v = w * V[static_cast<size_t>(s)][static_cast<size_t>(i)]
                         + c1 * r1 * (Pb[static_cast<size_t>(s)][static_cast<size_t>(i)] - X[static_cast<size_t>(s)][static_cast<size_t>(i)])
                         + c2 * r2 * (gbest[static_cast<size_t>(i)] - X[static_cast<size_t>(s)][static_cast<size_t>(i)]);
                double xn = X[static_cast<size_t>(s)][static_cast<size_t>(i)] + v;
                /* Bound reflection. */
                if (xn < lo[static_cast<size_t>(i)]) { xn = lo[static_cast<size_t>(i)]; v = -0.5 * v; }
                if (xn > hi[static_cast<size_t>(i)]) { xn = hi[static_cast<size_t>(i)]; v = -0.5 * v; }
                V[static_cast<size_t>(s)][static_cast<size_t>(i)] = v;
                X[static_cast<size_t>(s)][static_cast<size_t>(i)] = xn;
            }
            double fx = gads_eval(f, X[static_cast<size_t>(s)]);
            if (fx < fpb[static_cast<size_t>(s)]) {
                fpb[static_cast<size_t>(s)] = fx;
                Pb[static_cast<size_t>(s)] = X[static_cast<size_t>(s)];
                if (fx < fg) { fg = fx; gbest = X[static_cast<size_t>(s)]; }
            }
        }
    }
    std::vector<double> x = gbest;
    if (hybrid != 0.0) x = gads_hybrid(fn_p, x, lb, ub, f);
    return gads_col(x);
}

/* ================================================================ */
/* matlab_gads_simulannealbnd — bounded simulated annealing         */
/*                                                                  */
/* Boltzmann annealing from x0: adaptive Gaussian proposal scaled   */
/* by the temperature, Metropolis acceptance, exponential cooling   */
/* with periodic reannealing.  Returns the best point found.        */
/* ---------------------------------------------------------------- */
matlab_mat *matlab_gads_simulannealbnd(void *fn_p, matlab_mat *x0,
                                       matlab_mat *lb, matlab_mat *ub,
                                       double hybrid) {
    if (!fn_p || !x0) return mat_alloc(0, 0);
    gads_obj_fn f = reinterpret_cast<gads_obj_fn>(fn_p);
    int n = static_cast<int>(x0->rows * x0->cols);
    if (n < 1) return mat_alloc(0, 0);
    std::vector<double> lo, hi;
    gads_bounds(lb, ub, n, lo, hi);

    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = gads_clamp(x0->data[i], lo[static_cast<size_t>(i)],
                                               hi[static_cast<size_t>(i)]);
    double fx = gads_eval(f, x);
    std::vector<double> xbest = x;
    double fbest = fx;

    auto span = [&](int i) {
        double s = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
        return (s < 1e12 && s > 0) ? s : 10.0;
    };
    /* Slow geometric cooling (T ← 0.95·T per annealing step) with
     * periodic reannealing — explores the basin lattice of a highly
     * multi-modal objective instead of quenching into the nearest well
     * (a fast 1/k² schedule traps on Rastrigin). */
    const double T0 = 100.0;
    const int max_iter = 6000;
    const int anneal_interval = 50;   /* steps per temperature drop  */
    const int reanneal = 400;         /* steps stalled → reheat       */
    double T = T0;
    int since_improve = 0;
    for (int k = 1; k <= max_iter; ++k) {
        /* Proposal: Gaussian step scaled by the current temperature. */
        std::vector<double> xnew = x;
        double scale = sqrt(T / T0);
        for (int i = 0; i < n; ++i) {
            double v = x[static_cast<size_t>(i)] + scale * 0.5 * span(i) * gads_normal();
            xnew[static_cast<size_t>(i)] = gads_clamp(v, lo[static_cast<size_t>(i)],
                                                      hi[static_cast<size_t>(i)]);
        }
        double fnew = gads_eval(f, xnew);
        double dE = fnew - fx;
        if (dE < 0.0 || gads_uniform() < exp(-dE / std::max(T, 1e-12))) {
            x = xnew; fx = fnew;
            if (fx < fbest) { fbest = fx; xbest = x; since_improve = 0; }
            else since_improve++;
        } else since_improve++;
        if (k % anneal_interval == 0) T *= 0.95;     /* geometric cooling */
        if (T < 1e-8) T = 1e-8;
        if (since_improve > reanneal) {              /* reheat from best  */
            T = T0; x = xbest; fx = fbest; since_improve = 0;
        }
    }
    std::vector<double> xr = xbest;
    if (hybrid != 0.0) xr = gads_hybrid(fn_p, xr, lb, ub, f);
    return gads_col(xr);
}

}  /* extern "C" */
