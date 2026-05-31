// Reinforcement Learning Toolbox runtime — Tier 1 (tabular, autodiff-free).
//
// All exported symbols use C-linkage extern "C".  Wiring:
//   - lib/Sema/Resolver.cpp        : builtin registry (names + matlab_rl_* symbols)
//   - lib/MLIR/Lowering.cpp        : classdef constructor + free-fn + method intercepts
//   - tools/matlabc/main.cpp       : prelude trigger table (loads rl_classdefs.m)
//
// No external dependency (no Gym/Stable-Baselines/RLlib): the grid-world
// environment, the table Q critic, and the Q-learning + SARSA training loops
// are hand-coded over the shipped matlab_runtime kernel + its seeded PRNG.
// See docs/reinforcement_learning_toolbox_roadmap.md.
//
// Storage model (classdef carriers over packed-matrix properties):
//   rlMDPEnv          : T (S×A next-state, 1-based), R (S×A reward),
//                       NumStates, NumActions, StartState, TerminalState,
//                       GridRows, GridCols
//   rlFiniteSetSpec   : Elements (1×N), Dimension
//   rlTable           : Table (S×A)
//   rlQValueFunction  : QTable (S×A), NumStates, NumActions
//   rlQAgent/SARSA    : QTable (S×A) + scalar hyperparameters (Discount /
//                       LearnRate / Epsilon / EpsilonDecay / EpsilonMin) +
//                       IsSARSA flag.  `train` mutates QTable in place.
//
// The whole episode loop runs here over the deterministic next-state/reward
// tables — no MATLAB-side loop and no neural network for the tabular tier.

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" matlab_mat *matlab_zeros(double m, double n);
extern "C" matlab_mat *matlab_rand(double m, double n);
extern "C" matlab_mat *matlab_randn(double m, double n);

// ----- Reused Deep Learning Toolbox autodiff tape (zero duplication) -------
// The deep RL agents build their actor/critic forward passes and obtain
// parameter gradients by driving the SHIPPED dlnet reverse-mode tape.  A
// `dlarray` is just a matlab_obj carrying "Data" (matrix) + "Id" (tape-node
// index); we allocate bare shells with matlab_obj_new and call the existing
// matlab_dlnet_* C-ABI, so all forward kernels + the backward sweep are
// reused.  Only the Adam moment update (optimizer state) lives RL-side.
extern "C" matlab_obj *matlab_obj_new(int32_t class_id);
extern "C" matlab_mat *matlab_dlnet_reset0(void);
extern "C" int         matlab_dlnet_set_free_forward(int on);
extern "C" matlab_mat *matlab_dlnet_dlarray_init(void *obj, matlab_mat *X);
extern "C" matlab_mat *matlab_dlnet_extractdata(void *obj);
extern "C" matlab_mat *matlab_dlnet_mtimes(void *r, void *a, void *b);
extern "C" matlab_mat *matlab_dlnet_plus  (void *r, void *a, void *b);
extern "C" matlab_mat *matlab_dlnet_minus (void *r, void *a, void *b);
extern "C" matlab_mat *matlab_dlnet_times (void *r, void *a, void *b);
extern "C" matlab_mat *matlab_dlnet_relu  (void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_tanh  (void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_mse   (void *r, void *y, void *t);
extern "C" matlab_mat *matlab_dlnet_vertcat(void *r, void *a, void *b);
extern "C" matlab_mat *matlab_dlnet_softmax(void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_log   (void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_sum   (void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_exp   (void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_softplus(void *r, void *x);
extern "C" matlab_mat *matlab_dlnet_grad  (void *loss, void *var);

namespace rl {

// ---- thin wrappers: each dlnet op takes a fresh result shell + operands ----
inline matlab_obj *dl_leaf(matlab_mat *X) {
    matlab_obj *o = matlab_obj_new(0);
    matlab_dlnet_dlarray_init(o, X);
    return o;
}
inline matlab_obj *dl_mm(matlab_obj *a, matlab_obj *b) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_mtimes(r, a, b); return r;
}
inline matlab_obj *dl_add(matlab_obj *a, matlab_obj *b) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_plus(r, a, b); return r;
}
inline matlab_obj *dl_relu(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_relu(r, a); return r;
}
inline matlab_obj *dl_tanh(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_tanh(r, a); return r;
}
inline matlab_obj *dl_mse(matlab_obj *pred, matlab_obj *target) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_mse(r, pred, target); return r;
}
inline matlab_obj *dl_times(matlab_obj *a, matlab_obj *b) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_times(r, a, b); return r;
}
inline matlab_obj *dl_softmax(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_softmax(r, a); return r;
}
inline matlab_obj *dl_log(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_log(r, a); return r;
}
inline matlab_obj *dl_sum(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_sum(r, a); return r;
}
inline matlab_obj *dl_vcat(matlab_obj *a, matlab_obj *b) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_vertcat(r, a, b); return r;
}
inline matlab_obj *dl_sub(matlab_obj *a, matlab_obj *b) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_minus(r, a, b); return r;
}
inline matlab_obj *dl_exp(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_exp(r, a); return r;
}
inline matlab_obj *dl_softplus(matlab_obj *a) {
    matlab_obj *r = matlab_obj_new(0); matlab_dlnet_softplus(r, a); return r;
}
inline matlab_mat *dl_data(matlab_obj *o) {
    return matlab_obj_get_mat(o, "Data", 4);
}
inline matlab_mat *dl_grad(matlab_obj *loss, matlab_obj *var) {
    return matlab_dlnet_grad(loss, var);
}

inline void set_mat(void *o, const char *n, matlab_mat *m) {
    matlab_obj_set_mat(reinterpret_cast<matlab_obj *>(o), n,
                       static_cast<int64_t>(std::strlen(n)), m);
}
inline void set_f64(void *o, const char *n, double v) {
    matlab_obj_set_f64(reinterpret_cast<matlab_obj *>(o), n,
                       static_cast<int64_t>(std::strlen(n)), v);
}
inline matlab_mat *get_mat(void *o, const char *n) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(o), n,
                              static_cast<int64_t>(std::strlen(n)));
}
inline double get_f64(void *o, const char *n) {
    return matlab_obj_get_f64(reinterpret_cast<matlab_obj *>(o), n,
                              static_cast<int64_t>(std::strlen(n)));
}

inline double wrap_pi(double a) {
    a = std::fmod(a + M_PI, 2 * M_PI);
    if (a < 0) a += 2 * M_PI;
    return a - M_PI;
}

inline matlab_mat *clone_mat(matlab_mat *m) {
    if (!m) return matlab_zeros(1, 1);
    matlab_mat *c = matlab_zeros(static_cast<double>(m->rows), static_cast<double>(m->cols));
    std::memcpy(c->data, m->data, sizeof(double) * m->rows * m->cols);
    return c;
}

// A uniform draw in [0,1) honouring the global (rng-seeded) stream.  Uses the
// allocation-free scalar primitive — bit-identical to matlab_rand(1,1) but
// without leaking a 1x1 matrix on every call (deep-RL loops draw millions).
inline double urand() { return matlab_rand_scalar(); }
// A standard-normal draw off the same stream, allocation-free.
inline double urandn() { return matlab_randn_scalar(); }

// Free a plain (2-D) matrix the RL loops allocate as scratch — minibatch
// tensors, gradient clones returned by dl_grad, small constant operands.  All
// such matrices are dense 2-D (data + struct), never the magic-tagged 3-D/N-D
// descriptors, so a direct free is correct and avoids leaking across the tens
// of thousands of training steps a deep-RL run takes.
inline void free_mat(matlab_mat *m) { if (m) { free(m->data); free(m); } }

// argmax over a row of the Q table (S×A, row-major), ties → lowest index.
inline int64_t argmax_row(const double *Q, int64_t s, int64_t A) {
    const double *row = Q + (s - 1) * A;
    int64_t best = 0;
    double bv = row[0];
    for (int64_t a = 1; a < A; ++a)
        if (row[a] > bv) { bv = row[a]; best = a; }
    return best + 1;  // 1-based
}

inline double max_row(const double *Q, int64_t s, int64_t A) {
    const double *row = Q + (s - 1) * A;
    double mv = row[0];
    for (int64_t a = 1; a < A; ++a) mv = std::max(mv, row[a]);
    return mv;
}

// A state is terminal (absorbing) when every action self-loops back to it —
// the layout matlab_rl_gridworld_init / matlab_rl_mdp_init use for terminals.
// This generalises beyond a single TerminalState scalar (MDPs may have several).
inline bool is_terminal(const double *NS, int64_t s, int64_t A) {
    const double *row = NS + (s - 1) * A;
    for (int64_t a = 0; a < A; ++a)
        if (static_cast<int64_t>(row[a]) != s) return false;
    return true;
}

// ===== Deep-agent helpers (continuous-state envs) ==========================
// A 2-layer MLP (obsDim → H relu → out) is the default actor/critic network.
// Params are stored as matrices on the agent object; the gradient step reuses
// the dlnet tape (dl_* above), while inference + Adam live here.

// Plain forward (inference): single column input x[obsDim] → out[outDim].
// Matches the tape forward (W1·x+b1 → relu → W2·h+b2) so greedy/target-net
// evaluation is bit-consistent with the trained graph.
inline void mlp_forward(matlab_mat *W1, matlab_mat *b1, matlab_mat *W2,
                        matlab_mat *b2, const double *x, double *out) {
    int64_t H = W1->rows, obsDim = W1->cols, outDim = W2->rows;
    std::vector<double> h(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; ++i) {
        double s = b1->data[i];
        for (int64_t j = 0; j < obsDim; ++j) s += W1->data[i * obsDim + j] * x[j];
        h[static_cast<size_t>(i)] = (s > 0) ? s : 0.0;   // relu
    }
    for (int64_t a = 0; a < outDim; ++a) {
        double s = b2->data[a];
        for (int64_t i = 0; i < H; ++i) s += W2->data[a * H + i] * h[static_cast<size_t>(i)];
        out[a] = s;
    }
}

// Policy forward: logits → softmax → action probabilities probs[outDim].
inline void mlp_policy(matlab_mat *W1, matlab_mat *b1, matlab_mat *W2,
                       matlab_mat *b2, const double *x, double *probs) {
    int64_t A = W2->rows;
    mlp_forward(W1, b1, W2, b2, x, probs);     // logits in probs
    double mx = probs[0];
    for (int64_t a = 1; a < A; ++a) mx = std::max(mx, probs[a]);
    double s = 0;
    for (int64_t a = 0; a < A; ++a) { probs[a] = std::exp(probs[a] - mx); s += probs[a]; }
    for (int64_t a = 0; a < A; ++a) probs[a] /= (s > 0 ? s : 1.0);
}

// Adam in-place update of param P from grad G, with persistent moments M,V.
// Consumes (frees) G — every caller passes a fresh dl_grad() clone used only
// here, so freeing it reclaims the per-step gradient instead of leaking it.
inline void adam_step(matlab_mat *P, matlab_mat *G, matlab_mat *M, matlab_mat *V,
                      double lr, double t) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    if (P && G && M && V && G->rows * G->cols == P->rows * P->cols) {
        int64_t n = P->rows * P->cols;
        double bc1 = 1.0 - std::pow(b1, t), bc2 = 1.0 - std::pow(b2, t);
        for (int64_t k = 0; k < n; ++k) {
            double g = G->data[k];
            M->data[k] = b1 * M->data[k] + (1 - b1) * g;
            V->data[k] = b2 * V->data[k] + (1 - b2) * g * g;
            double mhat = M->data[k] / bc1, vhat = V->data[k] / bc2;
            P->data[k] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
    free_mat(G);
}

// Cart-pole dynamics (Barto-Sutton-Anderson).  state = [x, xdot, th, thdot];
// action index 1 → force −10, 2 → +10.  Returns reward and sets done.
inline void cartpole_step(double *st, int64_t action, double *reward, bool *done) {
    const double g = 9.8, mc = 1.0, mp = 0.1, l = 0.5, dt = 0.02;
    const double force = (action == 1) ? -10.0 : 10.0;
    double x = st[0], xd = st[1], th = st[2], thd = st[3];
    double ct = std::cos(th), sct = std::sin(th);
    double temp = (force + mp * l * thd * thd * sct) / (mc + mp);
    double thacc = (g * sct - ct * temp) / (l * (4.0 / 3.0 - mp * ct * ct / (mc + mp)));
    double xacc = temp - mp * l * thacc * ct / (mc + mp);
    st[0] = x + dt * xd;
    st[1] = xd + dt * xacc;
    st[2] = th + dt * thd;
    st[3] = thd + dt * thacc;
    bool fail = (std::fabs(st[0]) > 2.4) || (std::fabs(st[2]) > 12.0 * M_PI / 180.0);
    *done = fail;
    *reward = fail ? 0.0 : 1.0;
}

// ----- DDPG continuous-control helpers -------------------------------------
// Deterministic actor forward (inference): obs → relu(W1) → linear → tanh →
// scaled to ±actLimit.  Returns the continuous action vector.
inline void actor_forward(matlab_mat *W1, matlab_mat *b1, matlab_mat *W2,
                          matlab_mat *b2, const double *x, double actLimit,
                          double *act) {
    int64_t H = W1->rows, obsDim = W1->cols, actDim = W2->rows;
    std::vector<double> h(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; ++i) {
        double s = b1->data[i];
        for (int64_t j = 0; j < obsDim; ++j) s += W1->data[i * obsDim + j] * x[j];
        h[static_cast<size_t>(i)] = (s > 0) ? s : 0.0;
    }
    for (int64_t a = 0; a < actDim; ++a) {
        double s = b2->data[a];
        for (int64_t i = 0; i < H; ++i) s += W2->data[a * H + i] * h[static_cast<size_t>(i)];
        act[a] = std::tanh(s) * actLimit;
    }
}

// Q-critic forward (inference): input [obs; act] → relu(W1) → linear → scalar.
inline double critic_forward(matlab_mat *W1, matlab_mat *b1, matlab_mat *W2,
                             matlab_mat *b2, const double *sa) {
    int64_t H = W1->rows, inDim = W1->cols;
    std::vector<double> h(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; ++i) {
        double s = b1->data[i];
        for (int64_t j = 0; j < inDim; ++j) s += W1->data[i * inDim + j] * sa[j];
        h[static_cast<size_t>(i)] = (s > 0) ? s : 0.0;
    }
    double q = b2->data[0];
    for (int64_t i = 0; i < H; ++i) q += W2->data[i] * h[static_cast<size_t>(i)];
    return q;
}

// Soft target update: target ← tau·main + (1−tau)·target (elementwise).
inline void soft_update(matlab_mat *tgt, matlab_mat *main, double tau) {
    if (!tgt || !main) return;
    int64_t n = tgt->rows * tgt->cols;
    if (main->rows * main->cols != n) return;
    for (int64_t k = 0; k < n; ++k)
        tgt->data[k] = tau * main->data[k] + (1.0 - tau) * tgt->data[k];
}

// Pendulum swing-up dynamics (Gym-style).  Internal state st=[theta,thetadot]
// (theta=0 upright); action = torque clamped to ±actLimit.  Returns reward
// −(θ_wrapped² + 0.1·θ̇² + 0.001·u²); no early termination (fixed horizon).
inline double pendulum_step(double *st, double torque, double actLimit) {
    const double g = 10.0, m = 1.0, l = 1.0, dt = 0.05;
    double u = torque; if (u > actLimit) u = actLimit; if (u < -actLimit) u = -actLimit;
    double th = st[0], thd = st[1];
    double thn = wrap_pi(th);
    double cost = thn * thn + 0.1 * thd * thd + 0.001 * u * u;
    double thdd = (3.0 * g / (2.0 * l)) * std::sin(th) + (3.0 / (m * l * l)) * u;
    thd += thdd * dt;
    if (thd > 8.0) thd = 8.0; if (thd < -8.0) thd = -8.0;
    th += thd * dt;
    st[0] = th; st[1] = thd;
    return -cost;
}

// Observation encoding for the pendulum: [cos θ, sin θ, θ̇].
inline void pendulum_obs(const double *st, double *obs) {
    obs[0] = std::cos(st[0]); obs[1] = std::sin(st[0]); obs[2] = st[1];
}

}  // namespace rl

extern "C" {

// ===========================================================================
// Environment — rlPredefinedEnv("BasicGridWorld")
// ===========================================================================
// Canonical 5×5 grid (column-major state numbering: state = (col-1)*5 + row,
// matching MATLAB's createGridWorld).  Actions N=1,S=2,E=3,W=4 with row
// increasing downward.  Start [2,1] (state 2); terminal [5,5] (state 25,
// reward +10).  Obstacles {[3,3],[3,4],[3,5],[4,3]} block movement (the agent
// stays put on a blocked move).  Special jump: any action from [2,4] (state
// 17) teleports to [4,4] (state 19) with reward +5.  Every other step −1.
void matlab_rl_gridworld_init(matlab_obj *env) {
    const int64_t R = 5, C = 5, A = 4;
    const int64_t S = R * C;          // 25
    auto state = [&](int64_t row, int64_t col) { return (col - 1) * R + row; };

    // Obstacle set.
    bool obst[26] = {false};
    obst[state(3, 3)] = true;
    obst[state(3, 4)] = true;
    obst[state(3, 5)] = true;
    obst[state(4, 3)] = true;

    const int64_t term = state(5, 5);   // 25
    const int64_t jumpFrom = state(2, 4);   // 17
    const int64_t jumpTo   = state(4, 4);   // 19

    matlab_mat *T = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    matlab_mat *Rw = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    double *Td = T->data, *Rd = Rw->data;   // row-major S×A

    for (int64_t s = 1; s <= S; ++s) {
        // current (row,col): row = ((s-1) % R) + 1 ; col = ((s-1) / R) + 1
        int64_t row = ((s - 1) % R) + 1;
        int64_t col = ((s - 1) / R) + 1;
        for (int64_t a = 0; a < A; ++a) {
            int64_t idx = (s - 1) * A + a;
            // Terminal states self-loop with zero reward (absorbing).
            if (s == term) { Td[idx] = static_cast<double>(s); Rd[idx] = 0.0; continue; }
            // Jump cell: any action → jumpTo, reward +5.
            if (s == jumpFrom) { Td[idx] = static_cast<double>(jumpTo); Rd[idx] = 5.0; continue; }
            int64_t nr = row, nc = col;
            switch (a) {
                case 0: nr = row - 1; break;  // N
                case 1: nr = row + 1; break;  // S
                case 2: nc = col + 1; break;  // E
                case 3: nc = col - 1; break;  // W
            }
            int64_t ns = s;  // default: blocked → stay
            if (nr >= 1 && nr <= R && nc >= 1 && nc <= C) {
                int64_t cand = state(nr, nc);
                if (!obst[cand]) ns = cand;
            }
            Td[idx] = static_cast<double>(ns);
            Rd[idx] = (ns == term) ? 10.0 : -1.0;
        }
    }

    rl::set_mat(env, "T", T);
    rl::set_mat(env, "R", Rw);
    rl::set_f64(env, "NumStates",     static_cast<double>(S));
    rl::set_f64(env, "NumActions",    static_cast<double>(A));
    rl::set_f64(env, "StartState",    static_cast<double>(state(2, 1)));  // 2
    rl::set_f64(env, "TerminalState", static_cast<double>(term));
    rl::set_f64(env, "GridRows",      static_cast<double>(R));
    rl::set_f64(env, "GridCols",      static_cast<double>(C));
}

// rlMDPEnv(nextState, reward) — build a finite deterministic MDP directly
// from an S×A next-state table (1-based) and an S×A reward table.  Terminal
// states are those whose every action self-loops (encode them as NS(t,:)=t,
// reward 0).  StartState defaults to 1.  This is the adapted builder for the
// "Train RL Agent in MDP Environment" workflow (the MathWorks form chains
// createMDP → rlMDPEnv(mdp); 3-D `mdp.T(i,j,k)=v` indexed-property assignment
// is a documented frontend carve, so the deterministic table form is used).
void matlab_rl_mdp_init(matlab_obj *env, matlab_mat *NS, matlab_mat *RW) {
    int64_t S = NS ? NS->rows : 1, A = NS ? NS->cols : 1;
    matlab_mat *T  = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    matlab_mat *Rd = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    if (NS) std::memcpy(T->data,  NS->data, sizeof(double) * S * A);
    if (RW) std::memcpy(Rd->data, RW->data, sizeof(double) * S * A);
    rl::set_mat(env, "T", T);
    rl::set_mat(env, "R", Rd);
    rl::set_f64(env, "NumStates",     static_cast<double>(S));
    rl::set_f64(env, "NumActions",    static_cast<double>(A));
    rl::set_f64(env, "StartState",    1.0);
    rl::set_f64(env, "TerminalState", 0.0);   // unused; terminals auto-detected
    rl::set_f64(env, "GridRows",      0.0);
    rl::set_f64(env, "GridCols",      0.0);
}

// rlPredefinedEnv("CartPole-Discrete") — continuous-state control environment.
// Kind=2 selects the cart-pole dynamics in the deep-agent train/sim loops;
// obs is a 4-vector [x, xdot, theta, thetadot]; 2 discrete actions (±10 N).
void matlab_rl_cartpole_init(matlab_obj *env) {
    rl::set_f64(env, "Kind",       2.0);
    rl::set_f64(env, "ObsDim",     4.0);
    rl::set_f64(env, "NumStates",  4.0);   // obs dimension (continuous)
    rl::set_f64(env, "NumActions", 2.0);
    rl::set_f64(env, "MaxSteps",   500.0);
}

// rlPredefinedEnv("Pendulum-Continuous") — swing-up with a continuous torque.
// Kind=3 selects pendulum dynamics + continuous-action specs; obs = [cosθ,
// sinθ, θ̇] (dim 3); action = scalar torque in ±2.
void matlab_rl_pendulum_init(matlab_obj *env) {
    rl::set_f64(env, "Kind",        3.0);
    rl::set_f64(env, "ObsDim",      3.0);
    rl::set_f64(env, "NumStates",   3.0);
    rl::set_f64(env, "ActCont",     1.0);
    rl::set_f64(env, "ActionDim",   1.0);
    rl::set_f64(env, "ActionLimit", 2.0);
    rl::set_f64(env, "MaxSteps",    200.0);
}

// ===========================================================================
// Specs — getObservationInfo / getActionInfo
// ===========================================================================
// Fill a freshly-constructed rlFiniteSetSpec with Elements = 1:N where N is
// the environment's state count (obs) or action count (act).
static void fill_set_spec(matlab_obj *spec, int64_t n) {
    matlab_mat *el = matlab_zeros(1, static_cast<double>(n));
    for (int64_t i = 0; i < n; ++i) el->data[i] = static_cast<double>(i + 1);
    rl::set_mat(spec, "Elements", el);
    matlab_mat *dim = matlab_zeros(1, 2);
    dim->data[0] = 1; dim->data[1] = 1;
    rl::set_mat(spec, "Dimension", dim);
}

void matlab_rl_obs_info(matlab_obj *spec, matlab_obj *env) {
    // Continuous-state envs (Kind!=0) carry an observation *dimension* rather
    // than a finite element set: leave Elements empty and set Dimension=[D 1]
    // so the deep-agent ctor reads the obs width.  (Faithful MATLAB returns an
    // rlNumericSpec here; Tier-1/2 reuse the one spec class carrying both.)
    if (rl::get_f64(env, "Kind") != 0.0) {
        int64_t D = static_cast<int64_t>(rl::get_f64(env, "ObsDim"));
        rl::set_mat(spec, "Elements", matlab_zeros(1, 0));
        matlab_mat *dim = matlab_zeros(1, 2);
        dim->data[0] = static_cast<double>(D); dim->data[1] = 1;
        rl::set_mat(spec, "Dimension", dim);
        return;
    }
    fill_set_spec(spec, static_cast<int64_t>(rl::get_f64(env, "NumStates")));
}
void matlab_rl_act_info(matlab_obj *spec, matlab_obj *env) {
    // Continuous-action envs carry an action dimension + limit (rlNumericSpec
    // territory); discrete envs carry a finite element set.
    if (rl::get_f64(env, "ActCont") != 0.0) {
        int64_t D = static_cast<int64_t>(rl::get_f64(env, "ActionDim"));
        rl::set_mat(spec, "Elements", matlab_zeros(1, 0));
        matlab_mat *dim = matlab_zeros(1, 2);
        dim->data[0] = static_cast<double>(D); dim->data[1] = 1;
        rl::set_mat(spec, "Dimension", dim);
        rl::set_f64(spec, "Limit", rl::get_f64(env, "ActionLimit"));
        return;
    }
    fill_set_spec(spec, static_cast<int64_t>(rl::get_f64(env, "NumActions")));
}

// ===========================================================================
// Approximators — rlTable / rlQValueFunction
// ===========================================================================
void matlab_rl_table_init(matlab_obj *tbl, matlab_obj *obsInfo, matlab_obj *actInfo) {
    matlab_mat *oe = rl::get_mat(obsInfo, "Elements");
    matlab_mat *ae = rl::get_mat(actInfo, "Elements");
    int64_t S = oe ? (oe->rows * oe->cols) : 1;
    int64_t A = ae ? (ae->rows * ae->cols) : 1;
    rl::set_mat(tbl, "Table", matlab_zeros(static_cast<double>(S), static_cast<double>(A)));
}

void matlab_rl_qvf_init(matlab_obj *qvf, matlab_obj *tbl) {
    matlab_mat *t = rl::get_mat(tbl, "Table");
    int64_t S = t ? t->rows : 1, A = t ? t->cols : 1;
    matlab_mat *q = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    if (t) std::memcpy(q->data, t->data, sizeof(double) * S * A);
    rl::set_mat(qvf, "QTable", q);
    rl::set_f64(qvf, "NumStates",  static_cast<double>(S));
    rl::set_f64(qvf, "NumActions", static_cast<double>(A));
}

// ===========================================================================
// Agents — rlQAgent / rlSARSAAgent
// ===========================================================================
void matlab_rl_agent_init(matlab_obj *agent, matlab_obj *critic, double isSarsa) {
    matlab_mat *q = rl::get_mat(critic, "QTable");
    int64_t S = q ? q->rows : 1, A = q ? q->cols : 1;
    matlab_mat *qa = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    if (q) std::memcpy(qa->data, q->data, sizeof(double) * S * A);
    rl::set_mat(agent, "QTable", qa);
    rl::set_f64(agent, "NumStates",  static_cast<double>(S));
    rl::set_f64(agent, "NumActions", static_cast<double>(A));
    rl::set_f64(agent, "IsSARSA", isSarsa);
}

// getCritic(agent) → fresh rlQValueFunction carrying the agent's current Q.
void matlab_rl_get_critic(matlab_obj *qvf, matlab_obj *agent) {
    matlab_mat *q = rl::get_mat(agent, "QTable");
    int64_t S = q ? q->rows : 1, A = q ? q->cols : 1;
    matlab_mat *qc = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    if (q) std::memcpy(qc->data, q->data, sizeof(double) * S * A);
    rl::set_mat(qvf, "QTable", qc);
    rl::set_f64(qvf, "NumStates",  static_cast<double>(S));
    rl::set_f64(qvf, "NumActions", static_cast<double>(A));
}

// getLearnableParameters(critic) → the S×A Q matrix.
matlab_mat *matlab_rl_get_params(matlab_obj *critic) {
    matlab_mat *q = rl::get_mat(critic, "QTable");
    int64_t S = q ? q->rows : 1, A = q ? q->cols : 1;
    matlab_mat *out = matlab_zeros(static_cast<double>(S), static_cast<double>(A));
    if (q) std::memcpy(out->data, q->data, sizeof(double) * S * A);
    return out;
}

// ===========================================================================
// train(agent, env, trainOpts) — tabular Q-learning / SARSA
// ===========================================================================
// Mutates agent.QTable in place and returns an MaxEpisodes×1 column of
// per-episode cumulative rewards.
matlab_mat *matlab_rl_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    matlab_mat *Qm = rl::get_mat(agent, "QTable");
    matlab_mat *T  = rl::get_mat(env, "T");
    matlab_mat *Rw = rl::get_mat(env, "R");
    if (!Qm || !T || !Rw) return matlab_zeros(1, 1);

    int64_t A = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    int64_t start = static_cast<int64_t>(rl::get_f64(env, "StartState"));
    double  gamma = rl::get_f64(agent, "DiscountFactor");
    double  lr    = rl::get_f64(agent, "LearnRate");
    double  eps0  = rl::get_f64(agent, "Epsilon");
    double  epsD  = rl::get_f64(agent, "EpsilonDecay");
    double  epsM  = rl::get_f64(agent, "EpsilonMin");
    bool    sarsa = rl::get_f64(agent, "IsSARSA") != 0.0;

    int64_t maxEp   = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode"));
    if (maxEp < 1) maxEp = 1;
    if (maxStep < 1) maxStep = 1;

    double *Q  = Qm->data;
    double *Td = T->data;
    double *Rd = Rw->data;

    auto eps_greedy = [&](int64_t s, double eps) -> int64_t {
        if (rl::urand() < eps)
            return static_cast<int64_t>(rl::urand() * A) % A + 1;  // 1..A
        return rl::argmax_row(Q, s, A);
    };

    matlab_mat *stats = matlab_zeros(static_cast<double>(maxEp), 1);

    for (int64_t ep = 0; ep < maxEp; ++ep) {
        double eps = std::max(epsM, eps0 - epsD * static_cast<double>(ep));
        int64_t s = start;
        double epReward = 0.0;
        for (int64_t step = 0; step < maxStep; ++step) {
            if (rl::is_terminal(Td, s, A)) break;
            int64_t a = eps_greedy(s, eps);
            int64_t qi = (s - 1) * A + (a - 1);
            int64_t ns = static_cast<int64_t>(Td[qi]);
            double  r  = Rd[qi];
            epReward += r;
            double target;
            if (rl::is_terminal(Td, ns, A)) {
                target = r;
            } else if (sarsa) {
                int64_t a2 = eps_greedy(ns, eps);
                target = r + gamma * Q[(ns - 1) * A + (a2 - 1)];
            } else {
                target = r + gamma * rl::max_row(Q, ns, A);
            }
            Q[qi] += lr * (target - Q[qi]);
            s = ns;
        }
        stats->data[ep] = epReward;
    }
    return stats;
}

// ===========================================================================
// sim(agent, env) — one greedy rollout; returns cumulative reward (1×1)
// ===========================================================================
matlab_mat *matlab_rl_sim(matlab_obj *agent, matlab_obj *env) {
    matlab_mat *Qm = rl::get_mat(agent, "QTable");
    matlab_mat *T  = rl::get_mat(env, "T");
    matlab_mat *Rw = rl::get_mat(env, "R");
    matlab_mat *out = matlab_zeros(1, 1);
    if (!Qm || !T || !Rw) return out;

    int64_t A = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    int64_t start = static_cast<int64_t>(rl::get_f64(env, "StartState"));

    double *Q  = Qm->data;
    double *Td = T->data;
    double *Rd = Rw->data;

    int64_t s = start;
    double total = 0.0;
    for (int64_t step = 0; step < 1000; ++step) {
        if (rl::is_terminal(Td, s, A)) break;
        int64_t a = rl::argmax_row(Q, s, A);
        int64_t qi = (s - 1) * A + (a - 1);
        total += Rd[qi];
        s = static_cast<int64_t>(Td[qi]);
    }
    out->data[0] = total;
    return out;
}

// ===========================================================================
// Deep Q-Network (DQN) — rlDQNAgent(obsInfo, actInfo)
// ===========================================================================
// Auto-builds a 2-layer MLP critic (obsDim → H relu → numActions) with random
// init.  Forward/inference + Adam are local; the gradient step reuses the
// dlnet autodiff tape (dl_* above) — no autodiff code is duplicated here.

static const int64_t kDqnHidden = 24;

void matlab_rl_dqn_init(matlab_obj *agent, matlab_obj *obsInfo, matlab_obj *actInfo) {
    matlab_mat *dim = rl::get_mat(obsInfo, "Dimension");
    int64_t obsDim = 1;
    if (dim && dim->rows * dim->cols >= 1) {
        obsDim = 1;
        for (int64_t k = 0; k < dim->rows * dim->cols; ++k)
            obsDim *= static_cast<int64_t>(dim->data[k]);
    }
    if (obsDim < 1) obsDim = 1;
    matlab_mat *ae = rl::get_mat(actInfo, "Elements");
    int64_t A = ae ? (ae->rows * ae->cols) : 2;
    int64_t H = kDqnHidden;

    // He-scaled random init for the relu layer; small init for the output.
    matlab_mat *W1 = matlab_randn(static_cast<double>(H), static_cast<double>(obsDim));
    double s1 = std::sqrt(2.0 / static_cast<double>(obsDim));
    for (int64_t k = 0; k < H * obsDim; ++k) W1->data[k] *= s1;
    matlab_mat *W2 = matlab_randn(static_cast<double>(A), static_cast<double>(H));
    double s2 = std::sqrt(1.0 / static_cast<double>(H));
    for (int64_t k = 0; k < A * H; ++k) W2->data[k] *= s2;
    matlab_mat *b1 = matlab_zeros(static_cast<double>(H), 1);
    matlab_mat *b2 = matlab_zeros(static_cast<double>(A), 1);

    rl::set_mat(agent, "W1", W1);  rl::set_mat(agent, "b1", b1);
    rl::set_mat(agent, "W2", W2);  rl::set_mat(agent, "b2", b2);
    // target-net copies
    rl::set_mat(agent, "tW1", rl::clone_mat(W1)); rl::set_mat(agent, "tb1", rl::clone_mat(b1));
    rl::set_mat(agent, "tW2", rl::clone_mat(W2)); rl::set_mat(agent, "tb2", rl::clone_mat(b2));
    // Adam moments (zeros, same shapes)
    rl::set_mat(agent, "mW1", matlab_zeros(static_cast<double>(H), static_cast<double>(obsDim)));
    rl::set_mat(agent, "vW1", matlab_zeros(static_cast<double>(H), static_cast<double>(obsDim)));
    rl::set_mat(agent, "mb1", matlab_zeros(static_cast<double>(H), 1));
    rl::set_mat(agent, "vb1", matlab_zeros(static_cast<double>(H), 1));
    rl::set_mat(agent, "mW2", matlab_zeros(static_cast<double>(A), static_cast<double>(H)));
    rl::set_mat(agent, "vW2", matlab_zeros(static_cast<double>(A), static_cast<double>(H)));
    rl::set_mat(agent, "mb2", matlab_zeros(static_cast<double>(A), 1));
    rl::set_mat(agent, "vb2", matlab_zeros(static_cast<double>(A), 1));

    rl::set_f64(agent, "ObsDim", static_cast<double>(obsDim));
    rl::set_f64(agent, "NumActions", static_cast<double>(A));
    rl::set_f64(agent, "HiddenSize", static_cast<double>(H));
}

// One Adam gradient step on the critic from a minibatch (X obsDim×B, target
// A×B).  Builds the forward on the dlnet tape and pulls the four parameter
// gradients, then Adam-updates the stored matrices.  t = Adam step counter.
static void dqn_grad_step(matlab_obj *agent, matlab_mat *X, matlab_mat *Tg,
                          double lr, double t) {
    matlab_obj *W1L = rl::dl_leaf(rl::get_mat(agent, "W1"));
    matlab_obj *b1L = rl::dl_leaf(rl::get_mat(agent, "b1"));
    matlab_obj *W2L = rl::dl_leaf(rl::get_mat(agent, "W2"));
    matlab_obj *b2L = rl::dl_leaf(rl::get_mat(agent, "b2"));
    matlab_obj *xL  = rl::dl_leaf(X);
    matlab_obj *h   = rl::dl_relu(rl::dl_add(rl::dl_mm(W1L, xL), b1L));
    matlab_obj *pred = rl::dl_add(rl::dl_mm(W2L, h), b2L);
    matlab_obj *tL  = rl::dl_leaf(Tg);
    matlab_obj *loss = rl::dl_mse(pred, tL);

    matlab_mat *gW1 = rl::dl_grad(loss, W1L);
    matlab_mat *gb1 = rl::dl_grad(loss, b1L);
    matlab_mat *gW2 = rl::dl_grad(loss, W2L);
    matlab_mat *gb2 = rl::dl_grad(loss, b2L);

    rl::adam_step(rl::get_mat(agent, "W1"), gW1, rl::get_mat(agent, "mW1"), rl::get_mat(agent, "vW1"), lr, t);
    rl::adam_step(rl::get_mat(agent, "b1"), gb1, rl::get_mat(agent, "mb1"), rl::get_mat(agent, "vb1"), lr, t);
    rl::adam_step(rl::get_mat(agent, "W2"), gW2, rl::get_mat(agent, "mW2"), rl::get_mat(agent, "vW2"), lr, t);
    rl::adam_step(rl::get_mat(agent, "b2"), gb2, rl::get_mat(agent, "mb2"), rl::get_mat(agent, "vb2"), lr, t);

    matlab_dlnet_reset0();   // clear the tape for the next step
}

// train(dqnAgent, env, trainOpts) — DQN with experience replay + target net.
// Returns an MaxEpisodes×1 column of per-episode returns.
matlab_mat *matlab_rl_dqn_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t A      = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    double gamma   = rl::get_f64(agent, "DiscountFactor");
    double lr      = rl::get_f64(agent, "LearnRate");
    double eps     = rl::get_f64(agent, "Epsilon");
    double epsD    = rl::get_f64(agent, "EpsilonDecay");
    double epsM    = rl::get_f64(agent, "EpsilonMin");
    if (gamma <= 0) gamma = 0.99;
    if (lr <= 0) lr = 1e-3;

    int64_t maxEp   = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode"));
    if (maxEp < 1) maxEp = 1;
    if (maxStep < 1) maxStep = 500;

    const int64_t batch = 32, targetEvery = 200, minReplay = batch;
    const size_t  replayCap = 20000;

    // Replay: each row [s(obsDim), a, r, sp(obsDim), done].
    int64_t W = 2 * obsDim + 3;
    std::vector<double> replay; replay.reserve(replayCap * static_cast<size_t>(W));
    size_t rcount = 0, rhead = 0;

    matlab_mat *stats = matlab_zeros(static_cast<double>(maxEp), 1);
    double tstep = 0.0;          // Adam step counter
    int64_t gstep = 0;           // global env-step counter (target sync)

    std::vector<double> st(static_cast<size_t>(obsDim)), sp(static_cast<size_t>(obsDim));
    std::vector<double> q(static_cast<size_t>(A));

    for (int64_t ep = 0; ep < maxEp; ++ep) {
        // reset cart-pole: small uniform perturbation around upright.
        for (int64_t k = 0; k < obsDim; ++k) st[static_cast<size_t>(k)] = (rl::urand() * 2 - 1) * 0.05;
        double epReturn = 0.0;
        for (int64_t step = 0; step < maxStep; ++step) {
            // epsilon-greedy on the main net
            int64_t a;
            if (rl::urand() < eps) {
                a = static_cast<int64_t>(rl::urand() * A) % A + 1;
            } else {
                rl::mlp_forward(rl::get_mat(agent, "W1"), rl::get_mat(agent, "b1"),
                                rl::get_mat(agent, "W2"), rl::get_mat(agent, "b2"),
                                st.data(), q.data());
                a = 1; for (int64_t j = 1; j < A; ++j) if (q[static_cast<size_t>(j)] > q[static_cast<size_t>(a - 1)]) a = j + 1;
            }
            for (int64_t k = 0; k < obsDim; ++k) sp[static_cast<size_t>(k)] = st[static_cast<size_t>(k)];
            double r; bool done;
            rl::cartpole_step(sp.data(), a, &r, &done);
            epReturn += r;

            // push transition into the circular replay buffer
            std::vector<double> tr(static_cast<size_t>(W));
            for (int64_t k = 0; k < obsDim; ++k) tr[static_cast<size_t>(k)] = st[static_cast<size_t>(k)];
            tr[static_cast<size_t>(obsDim)] = static_cast<double>(a);
            tr[static_cast<size_t>(obsDim + 1)] = r;
            for (int64_t k = 0; k < obsDim; ++k) tr[static_cast<size_t>(obsDim + 2 + k)] = sp[static_cast<size_t>(k)];
            tr[static_cast<size_t>(W - 1)] = done ? 1.0 : 0.0;
            if (replay.size() < replayCap * static_cast<size_t>(W)) {
                replay.insert(replay.end(), tr.begin(), tr.end()); rcount++;
            } else {
                std::copy(tr.begin(), tr.end(), replay.begin() + static_cast<long>(rhead * static_cast<size_t>(W)));
                rhead = (rhead + 1) % replayCap;
            }

            // learn from a minibatch
            if (rcount >= static_cast<size_t>(minReplay)) {
                matlab_mat *X  = matlab_zeros(static_cast<double>(obsDim), static_cast<double>(batch));
                matlab_mat *Tg = matlab_zeros(static_cast<double>(A), static_cast<double>(batch));
                for (int64_t b = 0; b < batch; ++b) {
                    size_t idx = static_cast<size_t>(rl::urand() * static_cast<double>(rcount)) % rcount;
                    double *row = &replay[idx * static_cast<size_t>(W)];
                    double *sj = row, aj = row[obsDim], rj = row[obsDim + 1];
                    double *spj = row + obsDim + 2, dj = row[W - 1];
                    // current Q(s) (column b of X + base target)
                    std::vector<double> qc(static_cast<size_t>(A));
                    rl::mlp_forward(rl::get_mat(agent, "W1"), rl::get_mat(agent, "b1"),
                                    rl::get_mat(agent, "W2"), rl::get_mat(agent, "b2"), sj, qc.data());
                    for (int64_t k = 0; k < obsDim; ++k) X->data[k * batch + b] = sj[k];
                    for (int64_t k = 0; k < A; ++k) Tg->data[k * batch + b] = qc[static_cast<size_t>(k)];
                    // TD target for the taken action via the target net
                    double tgt = rj;
                    if (dj == 0.0) {
                        std::vector<double> qn(static_cast<size_t>(A));
                        rl::mlp_forward(rl::get_mat(agent, "tW1"), rl::get_mat(agent, "tb1"),
                                        rl::get_mat(agent, "tW2"), rl::get_mat(agent, "tb2"), spj, qn.data());
                        double mx = qn[0]; for (int64_t k = 1; k < A; ++k) mx = std::max(mx, qn[static_cast<size_t>(k)]);
                        tgt = rj + gamma * mx;
                    }
                    Tg->data[(static_cast<int64_t>(aj) - 1) * batch + b] = tgt;
                }
                tstep += 1.0;
                dqn_grad_step(agent, X, Tg, lr, tstep);
            }

            // target-net sync
            if (++gstep % targetEvery == 0) {
                rl::set_mat(agent, "tW1", rl::clone_mat(rl::get_mat(agent, "W1")));
                rl::set_mat(agent, "tb1", rl::clone_mat(rl::get_mat(agent, "b1")));
                rl::set_mat(agent, "tW2", rl::clone_mat(rl::get_mat(agent, "W2")));
                rl::set_mat(agent, "tb2", rl::clone_mat(rl::get_mat(agent, "b2")));
            }

            for (int64_t k = 0; k < obsDim; ++k) st[static_cast<size_t>(k)] = sp[static_cast<size_t>(k)];
            if (done) break;
        }
        eps = std::max(epsM, eps - epsD);
        stats->data[ep] = epReturn;
    }
    return stats;
}

// sim(dqnAgent, env) — one greedy episode; returns total reward (steps balanced).
matlab_mat *matlab_rl_dqn_sim(matlab_obj *agent, matlab_obj *env) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t A = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(env, "MaxSteps"));
    if (maxStep < 1) maxStep = 500;
    matlab_mat *out = matlab_zeros(1, 1);
    std::vector<double> st(static_cast<size_t>(obsDim), 0.0), q(static_cast<size_t>(A));
    double total = 0.0;
    for (int64_t step = 0; step < maxStep; ++step) {
        rl::mlp_forward(rl::get_mat(agent, "W1"), rl::get_mat(agent, "b1"),
                        rl::get_mat(agent, "W2"), rl::get_mat(agent, "b2"), st.data(), q.data());
        int64_t a = 1; for (int64_t j = 1; j < A; ++j) if (q[static_cast<size_t>(j)] > q[static_cast<size_t>(a - 1)]) a = j + 1;
        double r; bool done;
        rl::cartpole_step(st.data(), a, &r, &done);
        total += r;
        if (done) break;
    }
    out->data[0] = total;
    return out;
}

// ===========================================================================
// REINFORCE policy gradient (PG) — rlPGAgent(obsInfo, actInfo)
// ===========================================================================
// Auto-builds a stochastic-policy actor MLP (obsDim → H relu → numActions
// logits → softmax).  Per episode: roll out by sampling actions, compute
// discounted normalized returns, and take ONE policy-gradient step.  The
// −Σ logπ(aₜ|sₜ)·Ĝₜ loss is assembled on the reused dlnet tape (softmax/log/
// times/sum) and differentiated by dlgradient.

void matlab_rl_pg_init(matlab_obj *agent, matlab_obj *obsInfo, matlab_obj *actInfo) {
    // Same MLP layout + Adam moments as DQN (no target net for REINFORCE).
    matlab_rl_dqn_init(agent, obsInfo, actInfo);
}

matlab_mat *matlab_rl_pg_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t A      = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    double gamma   = rl::get_f64(agent, "DiscountFactor");
    double lr      = rl::get_f64(agent, "LearnRate");
    if (gamma <= 0) gamma = 0.99;
    if (lr <= 0) lr = 1e-2;

    int64_t maxEp   = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode"));
    if (maxEp < 1) maxEp = 1;
    if (maxStep < 1) maxStep = 500;

    matlab_mat *stats = matlab_zeros(static_cast<double>(maxEp), 1);
    double tstep = 0.0;
    std::vector<double> st(static_cast<size_t>(obsDim)), probs(static_cast<size_t>(A));

    for (int64_t ep = 0; ep < maxEp; ++ep) {
        // ---- roll out one episode, sampling actions from the policy ----
        std::vector<double> trajS;          // obsDim per step (row-major over steps)
        std::vector<int64_t> trajA;
        std::vector<double> trajR;
        for (int64_t k = 0; k < obsDim; ++k) st[static_cast<size_t>(k)] = (rl::urand() * 2 - 1) * 0.05;
        double epReturn = 0.0;
        for (int64_t step = 0; step < maxStep; ++step) {
            rl::mlp_policy(rl::get_mat(agent, "W1"), rl::get_mat(agent, "b1"),
                           rl::get_mat(agent, "W2"), rl::get_mat(agent, "b2"),
                           st.data(), probs.data());
            double u = rl::urand(), acc = 0; int64_t a = A;
            for (int64_t j = 0; j < A; ++j) { acc += probs[static_cast<size_t>(j)]; if (u <= acc) { a = j + 1; break; } }
            for (int64_t k = 0; k < obsDim; ++k) trajS.push_back(st[static_cast<size_t>(k)]);
            trajA.push_back(a);
            double r; bool done;
            rl::cartpole_step(st.data(), a, &r, &done);
            trajR.push_back(r);
            epReturn += r;
            if (done) break;
        }
        int64_t T = static_cast<int64_t>(trajA.size());
        if (T < 1) { stats->data[ep] = epReturn; continue; }

        // ---- discounted reward-to-go, normalized for variance reduction ----
        std::vector<double> G(static_cast<size_t>(T));
        double running = 0;
        for (int64_t t = T - 1; t >= 0; --t) { running = trajR[static_cast<size_t>(t)] + gamma * running; G[static_cast<size_t>(t)] = running; }
        double mean = 0; for (double g : G) mean += g; mean /= static_cast<double>(T);
        double var = 0; for (double g : G) var += (g - mean) * (g - mean); var /= static_cast<double>(T);
        double sd = std::sqrt(var) + 1e-8;
        for (int64_t t = 0; t < T; ++t) G[static_cast<size_t>(t)] = (G[static_cast<size_t>(t)] - mean) / sd;

        // ---- build X (obsDim×T) and the negative-advantage weight mask ----
        matlab_mat *X = matlab_zeros(static_cast<double>(obsDim), static_cast<double>(T));
        matlab_mat *Wt = matlab_zeros(static_cast<double>(A), static_cast<double>(T));
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t k = 0; k < obsDim; ++k) X->data[k * T + t] = trajS[static_cast<size_t>(t * obsDim + k)];
            // loss = Σ logπ · Wt ; minimizing with Wt = −Ĝ maximizes Σ logπ·Ĝ.
            Wt->data[(trajA[static_cast<size_t>(t)] - 1) * T + t] = -G[static_cast<size_t>(t)];
        }

        // ---- policy-gradient step on the reused tape ----
        matlab_obj *W1L = rl::dl_leaf(rl::get_mat(agent, "W1"));
        matlab_obj *b1L = rl::dl_leaf(rl::get_mat(agent, "b1"));
        matlab_obj *W2L = rl::dl_leaf(rl::get_mat(agent, "W2"));
        matlab_obj *b2L = rl::dl_leaf(rl::get_mat(agent, "b2"));
        matlab_obj *xL  = rl::dl_leaf(X);
        matlab_obj *logits = rl::dl_add(rl::dl_mm(W2L, rl::dl_relu(rl::dl_add(rl::dl_mm(W1L, xL), b1L))), b2L);
        matlab_obj *logp = rl::dl_log(rl::dl_softmax(logits));
        matlab_obj *loss = rl::dl_sum(rl::dl_times(logp, rl::dl_leaf(Wt)));
        tstep += 1.0;
        rl::adam_step(rl::get_mat(agent, "W1"), rl::dl_grad(loss, W1L), rl::get_mat(agent, "mW1"), rl::get_mat(agent, "vW1"), lr, tstep);
        rl::adam_step(rl::get_mat(agent, "b1"), rl::dl_grad(loss, b1L), rl::get_mat(agent, "mb1"), rl::get_mat(agent, "vb1"), lr, tstep);
        rl::adam_step(rl::get_mat(agent, "W2"), rl::dl_grad(loss, W2L), rl::get_mat(agent, "mW2"), rl::get_mat(agent, "vW2"), lr, tstep);
        rl::adam_step(rl::get_mat(agent, "b2"), rl::dl_grad(loss, b2L), rl::get_mat(agent, "mb2"), rl::get_mat(agent, "vb2"), lr, tstep);
        matlab_dlnet_reset0();

        stats->data[ep] = epReturn;
    }
    return stats;
}

// ===========================================================================
// PPO — Proximal Policy Optimization.  rlPPOAgent(obsInfo, actInfo)
// ===========================================================================
// On-policy actor-critic for the discrete cart-pole.  Each iteration collects
// a fresh rollout batch (no replay), estimates advantages with GAE(λ) off a
// learned value baseline, then runs several epochs of a CLIPPED surrogate
// update on the same batch.  Reuses the REINFORCE softmax-policy actor + its
// log π tape construction and the cart-pole env; adds a value network (MSE to
// the GAE returns) and the clip.
//
// The clipped objective L = E[min(rₜ·Âₜ, clip(rₜ,1±ε)·Âₜ)] is realised on the
// tape as a reweighted −Σ coefₜ·log π(aₜ|sₜ): each epoch a host forward gives
// the current ratio rₜ = exp(logπ_new − logπ_old); the per-sample coefficient
// is rₜ·Âₜ when the unclipped term is the minimum (so its gradient is live)
// and 0 when the clipped term wins (flat → zero gradient) — exactly the PPO
// gradient at the current parameters.

void matlab_rl_ppo_init(matlab_obj *agent, matlab_obj *obsInfo, matlab_obj *actInfo) {
    matlab_rl_dqn_init(agent, obsInfo, actInfo);   // actor W1/b1/W2/b2 (obs->H->A) + Adam m*/v*
    // Value baseline net, capital-V names so they don't collide with the
    // actor's Adam second-moment buffers (vW1…) that dqn_init created.
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t H = kDqnHidden;
    matlab_mat *VW1 = matlab_randn(static_cast<double>(H), static_cast<double>(obsDim));
    double s1 = std::sqrt(2.0 / static_cast<double>(obsDim));
    for (int64_t k = 0; k < H * obsDim; ++k) VW1->data[k] *= s1;
    matlab_mat *VW2 = matlab_randn(1, static_cast<double>(H));
    for (int64_t k = 0; k < H; ++k) VW2->data[k] *= 1e-3;
    rl::set_mat(agent, "VW1", VW1); rl::set_mat(agent, "Vb1", matlab_zeros(static_cast<double>(H),1));
    rl::set_mat(agent, "VW2", VW2); rl::set_mat(agent, "Vb2", matlab_zeros(1,1));
    const char *VP[4] = {"VW1","Vb1","VW2","Vb2"};
    for (const char *p : VP) { matlab_mat *pm = rl::get_mat(agent, p);
        rl::set_mat(agent, (std::string("m")+p).c_str(), matlab_zeros(pm->rows, pm->cols));
        rl::set_mat(agent, (std::string("n")+p).c_str(), matlab_zeros(pm->rows, pm->cols)); }
}

matlab_mat *matlab_rl_ppo_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t A      = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    double gamma = rl::get_f64(agent, "DiscountFactor"); if (gamma <= 0) gamma = 0.99;
    double lr    = rl::get_f64(agent, "LearnRate");      if (lr <= 0) lr = 1e-3;
    double lambda = rl::get_f64(agent, "GAELambda");  if (lambda <= 0) lambda = 0.95;
    double clip   = rl::get_f64(agent, "ClipRatio");  if (clip   <= 0) clip   = 0.2;
    int64_t epochs   = static_cast<int64_t>(rl::get_f64(agent, "NumEpoch"));   if (epochs   < 1) epochs   = 10;
    // Larger rollout batches markedly stabilise the on-policy update on
    // cart-pole (short batches plateau); 2048 steps is the sweet spot.
    int64_t rollout  = static_cast<int64_t>(rl::get_f64(agent, "RolloutLen")); if (rollout  < 1) rollout  = 2048;

    int64_t maxIter = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));   if (maxIter < 1) maxIter = 1;
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode")); if (maxStep < 1) maxStep = 500;

    int prevFreeFwd = matlab_dlnet_set_free_forward(1);
    matlab_mat *stats = matlab_zeros(static_cast<double>(maxIter), 1);
    double atstep = 0.0, vtstep = 0.0;
    std::vector<double> st(static_cast<size_t>(obsDim)), probs(static_cast<size_t>(A));

    for (int64_t it = 0; it < maxIter; ++it) {
        // ---- collect a rollout batch (>= rollout steps, whole episodes) ----
        std::vector<double> bS;            // obsDim per step
        std::vector<int64_t> bA;
        std::vector<double> bOldlp, bR, bV, bNextV; std::vector<char> bLast;
        double sumRet = 0.0; int64_t nEp = 0;
        while (static_cast<int64_t>(bA.size()) < rollout) {
            for (int64_t k = 0; k < obsDim; ++k) st[static_cast<size_t>(k)] = (rl::urand() * 2 - 1) * 0.05;
            double epRet = 0.0;
            for (int64_t step = 0; step < maxStep; ++step) {
                rl::mlp_policy(rl::get_mat(agent,"W1"), rl::get_mat(agent,"b1"),
                               rl::get_mat(agent,"W2"), rl::get_mat(agent,"b2"), st.data(), probs.data());
                double u = rl::urand(), acc = 0; int64_t a = A;
                for (int64_t j = 0; j < A; ++j) { acc += probs[static_cast<size_t>(j)]; if (u <= acc) { a = j + 1; break; } }
                double v; rl::mlp_forward(rl::get_mat(agent,"VW1"), rl::get_mat(agent,"Vb1"),
                                          rl::get_mat(agent,"VW2"), rl::get_mat(agent,"Vb2"), st.data(), &v);
                for (int64_t k = 0; k < obsDim; ++k) bS.push_back(st[static_cast<size_t>(k)]);
                bA.push_back(a);
                bOldlp.push_back(std::log(probs[static_cast<size_t>(a-1)] + 1e-12));
                bV.push_back(v);
                double r; bool done;
                rl::cartpole_step(st.data(), a, &r, &done);   // advances st to next state
                bR.push_back(r); epRet += r;
                bool last = done || (step == maxStep - 1);
                bLast.push_back(last ? 1 : 0);
                double nv = 0.0;
                if (!done) rl::mlp_forward(rl::get_mat(agent,"VW1"), rl::get_mat(agent,"Vb1"),
                                           rl::get_mat(agent,"VW2"), rl::get_mat(agent,"Vb2"), st.data(), &nv);
                bNextV.push_back(done ? 0.0 : nv);
                if (last) break;
            }
            sumRet += epRet; nEp++;
        }
        int64_t N = static_cast<int64_t>(bA.size());

        // ---- GAE(λ) advantages + returns, episode-aware ----
        std::vector<double> adv(static_cast<size_t>(N)), ret(static_cast<size_t>(N));
        double aNext = 0.0;
        for (int64_t t = N - 1; t >= 0; --t) {
            if (bLast[static_cast<size_t>(t)]) aNext = 0.0;
            double delta = bR[static_cast<size_t>(t)] + gamma * bNextV[static_cast<size_t>(t)] - bV[static_cast<size_t>(t)];
            double a_t = delta + gamma * lambda * aNext;
            adv[static_cast<size_t>(t)] = a_t; aNext = a_t;
            ret[static_cast<size_t>(t)] = a_t + bV[static_cast<size_t>(t)];
        }
        double mean = 0; for (double a : adv) mean += a; mean /= static_cast<double>(N);
        double var = 0; for (double a : adv) var += (a-mean)*(a-mean); var /= static_cast<double>(N);
        double sd = std::sqrt(var) + 1e-8;
        for (int64_t t = 0; t < N; ++t) adv[static_cast<size_t>(t)] = (adv[static_cast<size_t>(t)] - mean) / sd;

        // X (obsDim×N) is shared across epochs; R row (1×N) is the value target.
        matlab_mat *X = matlab_zeros(static_cast<double>(obsDim), static_cast<double>(N));
        matlab_mat *Rm = matlab_zeros(1, static_cast<double>(N));
        for (int64_t t = 0; t < N; ++t) {
            for (int64_t k = 0; k < obsDim; ++k) X->data[k*N+t] = bS[static_cast<size_t>(t*obsDim+k)];
            Rm->data[t] = ret[static_cast<size_t>(t)];
        }

        for (int64_t ep = 0; ep < epochs; ++ep) {
            // ---- clipped-surrogate actor step ----
            matlab_mat *Wt = matlab_zeros(static_cast<double>(A), static_cast<double>(N));
            for (int64_t t = 0; t < N; ++t) {
                rl::mlp_policy(rl::get_mat(agent,"W1"), rl::get_mat(agent,"b1"),
                               rl::get_mat(agent,"W2"), rl::get_mat(agent,"b2"),
                               &bS[static_cast<size_t>(t*obsDim)], probs.data());
                double newlp = std::log(probs[static_cast<size_t>(bA[static_cast<size_t>(t)]-1)] + 1e-12);
                double ratio = std::exp(newlp - bOldlp[static_cast<size_t>(t)]);
                double Aadv = adv[static_cast<size_t>(t)];
                double unclipped = ratio * Aadv;
                double rc = ratio < 1.0 - clip ? 1.0 - clip : (ratio > 1.0 + clip ? 1.0 + clip : ratio);
                double clipped = rc * Aadv;
                // gradient lives only when the unclipped term is the minimum.
                double coef = (unclipped <= clipped) ? ratio * Aadv : 0.0;
                Wt->data[(bA[static_cast<size_t>(t)]-1)*N + t] = -coef;   // minimise -Σ coef·logπ
            }
            matlab_obj *W1L=rl::dl_leaf(rl::get_mat(agent,"W1")), *b1L=rl::dl_leaf(rl::get_mat(agent,"b1"));
            matlab_obj *W2L=rl::dl_leaf(rl::get_mat(agent,"W2")), *b2L=rl::dl_leaf(rl::get_mat(agent,"b2"));
            matlab_obj *xL=rl::dl_leaf(X);
            matlab_obj *logits=rl::dl_add(rl::dl_mm(W2L, rl::dl_relu(rl::dl_add(rl::dl_mm(W1L,xL),b1L))), b2L);
            matlab_obj *logp=rl::dl_log(rl::dl_softmax(logits));
            matlab_obj *loss=rl::dl_sum(rl::dl_times(logp, rl::dl_leaf(Wt)));
            atstep += 1.0;
            rl::adam_step(rl::get_mat(agent,"W1"), rl::dl_grad(loss,W1L), rl::get_mat(agent,"mW1"), rl::get_mat(agent,"vW1"), lr, atstep);
            rl::adam_step(rl::get_mat(agent,"b1"), rl::dl_grad(loss,b1L), rl::get_mat(agent,"mb1"), rl::get_mat(agent,"vb1"), lr, atstep);
            rl::adam_step(rl::get_mat(agent,"W2"), rl::dl_grad(loss,W2L), rl::get_mat(agent,"mW2"), rl::get_mat(agent,"vW2"), lr, atstep);
            rl::adam_step(rl::get_mat(agent,"b2"), rl::dl_grad(loss,b2L), rl::get_mat(agent,"mb2"), rl::get_mat(agent,"vb2"), lr, atstep);
            matlab_dlnet_reset0();
            rl::free_mat(Wt);

            // ---- value MSE step ----
            matlab_obj *vW1L=rl::dl_leaf(rl::get_mat(agent,"VW1")), *vb1L=rl::dl_leaf(rl::get_mat(agent,"Vb1"));
            matlab_obj *vW2L=rl::dl_leaf(rl::get_mat(agent,"VW2")), *vb2L=rl::dl_leaf(rl::get_mat(agent,"Vb2"));
            matlab_obj *vh=rl::dl_relu(rl::dl_add(rl::dl_mm(vW1L, rl::dl_leaf(X)), vb1L));
            matlab_obj *vpred=rl::dl_add(rl::dl_mm(vW2L, vh), vb2L);
            matlab_obj *vloss=rl::dl_mse(vpred, rl::dl_leaf(Rm));
            vtstep += 1.0;
            rl::adam_step(rl::get_mat(agent,"VW1"), rl::dl_grad(vloss,vW1L), rl::get_mat(agent,"mVW1"), rl::get_mat(agent,"nVW1"), lr, vtstep);
            rl::adam_step(rl::get_mat(agent,"Vb1"), rl::dl_grad(vloss,vb1L), rl::get_mat(agent,"mVb1"), rl::get_mat(agent,"nVb1"), lr, vtstep);
            rl::adam_step(rl::get_mat(agent,"VW2"), rl::dl_grad(vloss,vW2L), rl::get_mat(agent,"mVW2"), rl::get_mat(agent,"nVW2"), lr, vtstep);
            rl::adam_step(rl::get_mat(agent,"Vb2"), rl::dl_grad(vloss,vb2L), rl::get_mat(agent,"mVb2"), rl::get_mat(agent,"nVb2"), lr, vtstep);
            matlab_dlnet_reset0();
        }
        rl::free_mat(X); rl::free_mat(Rm);
        stats->data[it] = sumRet / static_cast<double>(nEp > 0 ? nEp : 1);
    }
    matlab_dlnet_set_free_forward(prevFreeFwd);
    return stats;
}

// sim(ppoAgent, env) — greedy (argmax) cart-pole rollout; same actor layout as
// the REINFORCE agent, so the PG greedy sim applies verbatim.
matlab_mat *matlab_rl_pg_sim(matlab_obj *agent, matlab_obj *env);   // defined below
matlab_mat *matlab_rl_ppo_sim(matlab_obj *agent, matlab_obj *env) {
    return matlab_rl_pg_sim(agent, env);
}

// ===========================================================================
// Deep Deterministic Policy Gradient (DDPG) — rlDDPGAgent(obsInfo, actInfo)
// ===========================================================================
// Continuous control.  Auto-builds a deterministic actor (obsDim → H relu → 1
// tanh·limit) and a Q(s,a) critic ([obsDim+actDim] → H relu → 1), each with a
// soft-updated target copy.  Critic learns a TD target (MSE); the actor
// follows the deterministic policy gradient — its loss −Σ Q(s,actor(s)) is
// built on the reused tape by vertcat-ing the state with the actor output and
// forwarding the critic, taking gradients w.r.t. actor params only.

static const int64_t kDdpgHidden = 32;

void matlab_rl_ddpg_init(matlab_obj *agent, matlab_obj *obsInfo, matlab_obj *actInfo) {
    matlab_mat *od = rl::get_mat(obsInfo, "Dimension");
    int64_t obsDim = (od && od->rows * od->cols) ? static_cast<int64_t>(od->data[0]) : 1;
    matlab_mat *ad = rl::get_mat(actInfo, "Dimension");
    int64_t actDim = (ad && ad->rows * ad->cols) ? static_cast<int64_t>(ad->data[0]) : 1;
    double actLimit = rl::get_f64(actInfo, "Limit");
    if (actLimit <= 0) actLimit = 1.0;
    int64_t H = kDdpgHidden, cin = obsDim + actDim;

    auto he = [&](int64_t r, int64_t c) {
        matlab_mat *m = matlab_randn(static_cast<double>(r), static_cast<double>(c));
        double s = std::sqrt(2.0 / static_cast<double>(c));
        for (int64_t k = 0; k < r * c; ++k) m->data[k] *= s;
        return m;
    };
    auto small = [&](int64_t r, int64_t c) {
        matlab_mat *m = matlab_randn(static_cast<double>(r), static_cast<double>(c));
        for (int64_t k = 0; k < r * c; ++k) m->data[k] *= 1e-3;
        return m;
    };
    auto z = [&](int64_t r, int64_t c) { return matlab_zeros(static_cast<double>(r), static_cast<double>(c)); };

    // actor
    rl::set_mat(agent, "aW1", he(H, obsDim));   rl::set_mat(agent, "ab1", z(H, 1));
    rl::set_mat(agent, "aW2", small(actDim, H)); rl::set_mat(agent, "ab2", z(actDim, 1));
    // critic
    rl::set_mat(agent, "cW1", he(H, cin));      rl::set_mat(agent, "cb1", z(H, 1));
    rl::set_mat(agent, "cW2", small(1, H));      rl::set_mat(agent, "cb2", z(1, 1));
    // targets
    const char *P[8] = {"aW1","ab1","aW2","ab2","cW1","cb1","cW2","cb2"};
    for (const char *p : P) {
        std::string tn = std::string("t") + p;
        rl::set_mat(agent, tn.c_str(), rl::clone_mat(rl::get_mat(agent, p)));
        // Adam moments
        matlab_mat *pm = rl::get_mat(agent, p);
        rl::set_mat(agent, (std::string("m") + p).c_str(), z(pm->rows, pm->cols));
        rl::set_mat(agent, (std::string("v") + p).c_str(), z(pm->rows, pm->cols));
    }
    rl::set_f64(agent, "ObsDim", static_cast<double>(obsDim));
    rl::set_f64(agent, "ActDim", static_cast<double>(actDim));
    rl::set_f64(agent, "ActLimit", actLimit);
    rl::set_f64(agent, "HiddenSize", static_cast<double>(H));
}

// One critic Adam step (TD MSE) on the reused tape.
static void ddpg_critic_step(matlab_obj *ag, matlab_mat *Xsa, matlab_mat *Y, double lr, double t) {
    matlab_obj *W1=rl::dl_leaf(rl::get_mat(ag,"cW1")), *b1=rl::dl_leaf(rl::get_mat(ag,"cb1"));
    matlab_obj *W2=rl::dl_leaf(rl::get_mat(ag,"cW2")), *b2=rl::dl_leaf(rl::get_mat(ag,"cb2"));
    matlab_obj *x=rl::dl_leaf(Xsa);
    matlab_obj *q=rl::dl_add(rl::dl_mm(W2, rl::dl_relu(rl::dl_add(rl::dl_mm(W1,x),b1))), b2);
    matlab_obj *loss=rl::dl_mse(q, rl::dl_leaf(Y));
    rl::adam_step(rl::get_mat(ag,"cW1"), rl::dl_grad(loss,W1), rl::get_mat(ag,"mcW1"), rl::get_mat(ag,"vcW1"), lr, t);
    rl::adam_step(rl::get_mat(ag,"cb1"), rl::dl_grad(loss,b1), rl::get_mat(ag,"mcb1"), rl::get_mat(ag,"vcb1"), lr, t);
    rl::adam_step(rl::get_mat(ag,"cW2"), rl::dl_grad(loss,W2), rl::get_mat(ag,"mcW2"), rl::get_mat(ag,"vcW2"), lr, t);
    rl::adam_step(rl::get_mat(ag,"cb2"), rl::dl_grad(loss,b2), rl::get_mat(ag,"mcb2"), rl::get_mat(ag,"vcb2"), lr, t);
    matlab_dlnet_reset0();   // frees non-leaf forward values + adjoints + temps
}

// One actor Adam step (deterministic policy gradient) on the reused tape:
// loss = −Σ Q(s, actor(s)); the actor output is tanh·limit, vertcat'd with the
// state and fed through the critic.  Gradients taken w.r.t. actor params only.
static void ddpg_actor_step(matlab_obj *ag, matlab_mat *S, double actLimit, double lr, double t) {
    matlab_obj *aW1=rl::dl_leaf(rl::get_mat(ag,"aW1")), *ab1=rl::dl_leaf(rl::get_mat(ag,"ab1"));
    matlab_obj *aW2=rl::dl_leaf(rl::get_mat(ag,"aW2")), *ab2=rl::dl_leaf(rl::get_mat(ag,"ab2"));
    matlab_obj *cW1=rl::dl_leaf(rl::get_mat(ag,"cW1")), *cb1=rl::dl_leaf(rl::get_mat(ag,"cb1"));
    matlab_obj *cW2=rl::dl_leaf(rl::get_mat(ag,"cW2")), *cb2=rl::dl_leaf(rl::get_mat(ag,"cb2"));
    matlab_obj *sL=rl::dl_leaf(S);
    matlab_mat *lim=matlab_zeros(1,1); lim->data[0]=actLimit;
    matlab_obj *aPred=rl::dl_times(rl::dl_tanh(rl::dl_add(rl::dl_mm(aW2, rl::dl_relu(rl::dl_add(rl::dl_mm(aW1,sL),ab1))), ab2)), rl::dl_leaf(lim));
    matlab_obj *sa=rl::dl_vcat(sL, aPred);
    matlab_obj *q=rl::dl_add(rl::dl_mm(cW2, rl::dl_relu(rl::dl_add(rl::dl_mm(cW1,sa),cb1))), cb2);
    matlab_mat *neg=matlab_zeros(1,1); neg->data[0]=-1.0;
    matlab_obj *loss=rl::dl_times(rl::dl_sum(q), rl::dl_leaf(neg));
    rl::adam_step(rl::get_mat(ag,"aW1"), rl::dl_grad(loss,aW1), rl::get_mat(ag,"maW1"), rl::get_mat(ag,"vaW1"), lr, t);
    rl::adam_step(rl::get_mat(ag,"ab1"), rl::dl_grad(loss,ab1), rl::get_mat(ag,"mab1"), rl::get_mat(ag,"vab1"), lr, t);
    rl::adam_step(rl::get_mat(ag,"aW2"), rl::dl_grad(loss,aW2), rl::get_mat(ag,"maW2"), rl::get_mat(ag,"vaW2"), lr, t);
    rl::adam_step(rl::get_mat(ag,"ab2"), rl::dl_grad(loss,ab2), rl::get_mat(ag,"mab2"), rl::get_mat(ag,"vab2"), lr, t);
    matlab_dlnet_reset0();   // frees non-leaf forward values + adjoints + temps
    rl::free_mat(lim); rl::free_mat(neg);   // 1x1 leaf constants (not tape-freed)
}

matlab_mat *matlab_rl_ddpg_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(agent, "ActDim"));
    double actLimit = rl::get_f64(agent, "ActLimit");
    double gamma = rl::get_f64(agent, "DiscountFactor"); if (gamma <= 0) gamma = 0.99;
    double lr    = rl::get_f64(agent, "LearnRate");      if (lr <= 0) lr = 1e-3;
    double tau   = rl::get_f64(agent, "Tau");            if (tau <= 0) tau = 5e-3;
    // Tunables (default to a sane DDPG recipe when the field is unset/<=0).
    double actorLr = rl::get_f64(agent, "ActorLR");    if (actorLr <= 0) actorLr = lr;
    // Reward scaling for the critic target.  With gamma=0.99 and per-step
    // pendulum rewards up to ~-16, undiscounted returns reach ~-1600, whose
    // raw TD targets swamp the small critic; scaling by 0.1 keeps Q at O(100)
    // so the swing-up actually learns (untrained ≈ -1680 → trained ≈ -380).
    double rscale  = rl::get_f64(agent, "RewardScale"); if (rscale <= 0) rscale = 0.1;
    double ouSigma = rl::get_f64(agent, "OUSigma");     if (ouSigma <= 0) ouSigma = 0.2;
    int64_t gradSteps = static_cast<int64_t>(rl::get_f64(agent, "GradSteps")); if (gradSteps < 1) gradSteps = 1;

    int64_t maxEp   = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));   if (maxEp < 1) maxEp = 1;
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode")); if (maxStep < 1) maxStep = 200;

    // DDPG runs ~maxEp·maxStep·gradSteps grad calls on the reused tape; reclaim
    // non-leaf forward values on each reset so the run stays bounded (see
    // matlab_dlnet_set_free_forward).  This loop never holds a non-leaf dlarray
    // across a reset, so it is safe.  Restored on exit.
    int prevFreeFwd = matlab_dlnet_set_free_forward(1);

    const int64_t batch = 64, minReplay = batch;
    const size_t  replayCap = 50000;
    int64_t Wd = 2 * obsDim + actDim + 2;   // [obs, act, r, nextObs, done]
    std::vector<double> replay; size_t rcount = 0, rhead = 0;

    matlab_mat *stats = matlab_zeros(static_cast<double>(maxEp), 1);
    double tstep = 0.0;
    std::vector<double> sint(2), obs(static_cast<size_t>(obsDim)), nobs(static_cast<size_t>(obsDim)), act(static_cast<size_t>(actDim));
    const double ouTheta = 0.15;

    for (int64_t ep = 0; ep < maxEp; ++ep) {
        // reset: pendulum hanging down (theta=pi) + small noise
        sint[0] = M_PI + (rl::urand() * 2 - 1) * 0.1; sint[1] = (rl::urand() * 2 - 1) * 0.1;
        std::vector<double> ou(static_cast<size_t>(actDim), 0.0);
        rl::pendulum_obs(sint.data(), obs.data());
        double epReturn = 0.0;
        for (int64_t step = 0; step < maxStep; ++step) {
            rl::actor_forward(rl::get_mat(agent,"aW1"), rl::get_mat(agent,"ab1"),
                              rl::get_mat(agent,"aW2"), rl::get_mat(agent,"ab2"),
                              obs.data(), actLimit, act.data());
            // OU exploration noise
            for (int64_t k = 0; k < actDim; ++k) {
                ou[static_cast<size_t>(k)] += -ouTheta * ou[static_cast<size_t>(k)] + ouSigma * rl::urandn();
                act[static_cast<size_t>(k)] += ou[static_cast<size_t>(k)];
                if (act[static_cast<size_t>(k)] > actLimit) act[static_cast<size_t>(k)] = actLimit;
                if (act[static_cast<size_t>(k)] < -actLimit) act[static_cast<size_t>(k)] = -actLimit;
            }
            double r = rl::pendulum_step(sint.data(), act[0], actLimit);
            rl::pendulum_obs(sint.data(), nobs.data());
            epReturn += r;

            std::vector<double> tr(static_cast<size_t>(Wd));
            for (int64_t k = 0; k < obsDim; ++k) tr[static_cast<size_t>(k)] = obs[static_cast<size_t>(k)];
            for (int64_t k = 0; k < actDim; ++k) tr[static_cast<size_t>(obsDim + k)] = act[static_cast<size_t>(k)];
            tr[static_cast<size_t>(obsDim + actDim)] = r;
            for (int64_t k = 0; k < obsDim; ++k) tr[static_cast<size_t>(obsDim + actDim + 1 + k)] = nobs[static_cast<size_t>(k)];
            tr[static_cast<size_t>(Wd - 1)] = 0.0;
            if (replay.size() < replayCap * static_cast<size_t>(Wd)) { replay.insert(replay.end(), tr.begin(), tr.end()); rcount++; }
            else { std::copy(tr.begin(), tr.end(), replay.begin() + static_cast<long>(rhead * static_cast<size_t>(Wd))); rhead = (rhead + 1) % replayCap; }

            for (int64_t gs = 0; gs < gradSteps && rcount >= static_cast<size_t>(minReplay); ++gs) {
                matlab_mat *Xsa = matlab_zeros(static_cast<double>(obsDim + actDim), static_cast<double>(batch));
                matlab_mat *Y   = matlab_zeros(1, static_cast<double>(batch));
                matlab_mat *S   = matlab_zeros(static_cast<double>(obsDim), static_cast<double>(batch));
                for (int64_t b = 0; b < batch; ++b) {
                    size_t idx = static_cast<size_t>(rl::urand() * static_cast<double>(rcount)) % rcount;
                    double *row = &replay[idx * static_cast<size_t>(Wd)];
                    double *oj = row, *aj = row + obsDim, rj = row[obsDim + actDim];
                    double *npj = row + obsDim + actDim + 1;
                    for (int64_t k = 0; k < obsDim; ++k) { Xsa->data[k*batch+b]=oj[k]; S->data[k*batch+b]=oj[k]; }
                    for (int64_t k = 0; k < actDim; ++k) Xsa->data[(obsDim+k)*batch+b]=aj[k];
                    // TD target via target nets
                    std::vector<double> na(static_cast<size_t>(actDim)), sap(static_cast<size_t>(obsDim+actDim));
                    rl::actor_forward(rl::get_mat(agent,"taW1"), rl::get_mat(agent,"tab1"),
                                      rl::get_mat(agent,"taW2"), rl::get_mat(agent,"tab2"), npj, actLimit, na.data());
                    for (int64_t k = 0; k < obsDim; ++k) sap[static_cast<size_t>(k)] = npj[k];
                    for (int64_t k = 0; k < actDim; ++k) sap[static_cast<size_t>(obsDim+k)] = na[static_cast<size_t>(k)];
                    double qn = rl::critic_forward(rl::get_mat(agent,"tcW1"), rl::get_mat(agent,"tcb1"),
                                                   rl::get_mat(agent,"tcW2"), rl::get_mat(agent,"tcb2"), sap.data());
                    Y->data[b] = rscale * rj + gamma * qn;
                }
                tstep += 1.0;
                ddpg_critic_step(agent, Xsa, Y, lr, tstep);
                ddpg_actor_step(agent, S, actLimit, actorLr, tstep);
                const char *P[8] = {"aW1","ab1","aW2","ab2","cW1","cb1","cW2","cb2"};
                for (const char *p : P)
                    rl::soft_update(rl::get_mat(agent, (std::string("t")+p).c_str()), rl::get_mat(agent, p), tau);
                rl::free_mat(Xsa); rl::free_mat(Y); rl::free_mat(S);
            }
            for (int64_t k = 0; k < obsDim; ++k) obs[static_cast<size_t>(k)] = nobs[static_cast<size_t>(k)];
        }
        stats->data[ep] = epReturn;
    }
    matlab_dlnet_set_free_forward(prevFreeFwd);
    return stats;
}

// sim(ddpgAgent, env) — one greedy (noise-free) episode; returns total reward.
matlab_mat *matlab_rl_ddpg_sim(matlab_obj *agent, matlab_obj *env) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(agent, "ActDim"));
    double actLimit = rl::get_f64(agent, "ActLimit");
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(env, "MaxSteps")); if (maxStep < 1) maxStep = 200;
    matlab_mat *out = matlab_zeros(1, 1);
    std::vector<double> sint(2), obs(static_cast<size_t>(obsDim)), act(static_cast<size_t>(actDim));
    sint[0] = M_PI; sint[1] = 0.0;          // deterministic start: hanging down
    rl::pendulum_obs(sint.data(), obs.data());
    double total = 0.0;
    for (int64_t step = 0; step < maxStep; ++step) {
        rl::actor_forward(rl::get_mat(agent,"aW1"), rl::get_mat(agent,"ab1"),
                          rl::get_mat(agent,"aW2"), rl::get_mat(agent,"ab2"), obs.data(), actLimit, act.data());
        total += rl::pendulum_step(sint.data(), act[0], actLimit);
        rl::pendulum_obs(sint.data(), obs.data());
    }
    out->data[0] = total;
    return out;
}

// ===========================================================================
// TD3 — Twin Delayed DDPG.  rlTD3Agent(obsInfo, actInfo)
// ===========================================================================
// Three fixes on DDPG that tame the critic's Q-value overestimation:
//   (1) TWIN critics — the TD target takes the MINIMUM of two target critics,
//       so an over-optimistic critic can't drive the policy;
//   (2) TARGET-POLICY SMOOTHING — clipped Gaussian noise is added to the
//       target action, so the critic can't exploit a sharp Q peak;
//   (3) DELAYED updates — the actor and all target networks update once every
//       d critic steps (d=2), letting the critics settle first.
// Reuses the DDPG actor/critic networks, replay, pendulum env, reward scaling
// and the shared autodiff tape; only the second critic + the TD-target
// computation + the update schedule differ.  The actor step (DPG through
// critic 1) and the greedy sim are the DDPG ones verbatim.

void matlab_rl_td3_init(matlab_obj *agent, matlab_obj *obsInfo, matlab_obj *actInfo) {
    matlab_rl_ddpg_init(agent, obsInfo, actInfo);   // actor + critic "c" + targets + moments
    // Add the second critic "c2" + its target copy + Adam moments.
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(agent, "ActDim"));
    int64_t H = static_cast<int64_t>(rl::get_f64(agent, "HiddenSize"));
    int64_t cin = obsDim + actDim;
    auto he = [&](int64_t r, int64_t c) {
        matlab_mat *m = matlab_randn(static_cast<double>(r), static_cast<double>(c));
        double s = std::sqrt(2.0 / static_cast<double>(c));
        for (int64_t k = 0; k < r * c; ++k) m->data[k] *= s; return m; };
    auto small = [&](int64_t r, int64_t c) {
        matlab_mat *m = matlab_randn(static_cast<double>(r), static_cast<double>(c));
        for (int64_t k = 0; k < r * c; ++k) m->data[k] *= 1e-3; return m; };
    auto z = [&](int64_t r, int64_t c) { return matlab_zeros(static_cast<double>(r), static_cast<double>(c)); };
    rl::set_mat(agent, "c2W1", he(H, cin));  rl::set_mat(agent, "c2b1", z(H, 1));
    rl::set_mat(agent, "c2W2", small(1, H)); rl::set_mat(agent, "c2b2", z(1, 1));
    const char *P[4] = {"c2W1","c2b1","c2W2","c2b2"};
    for (const char *p : P) {
        rl::set_mat(agent, (std::string("t")+p).c_str(), rl::clone_mat(rl::get_mat(agent, p)));
        matlab_mat *pm = rl::get_mat(agent, p);
        rl::set_mat(agent, (std::string("m")+p).c_str(), z(pm->rows, pm->cols));
        rl::set_mat(agent, (std::string("v")+p).c_str(), z(pm->rows, pm->cols));
    }
}

// One Adam step for the critic with field prefix "c" or "c2", to the shared
// TD target Y (both twins regress the same min-target).
static void td3_critic_step(matlab_obj *ag, const char *pfx, matlab_mat *Xsa, matlab_mat *Y, double lr, double t) {
    auto F = [&](const char *s) { return std::string(pfx) + s; };
    matlab_obj *W1=rl::dl_leaf(rl::get_mat(ag,F("W1").c_str())), *b1=rl::dl_leaf(rl::get_mat(ag,F("b1").c_str()));
    matlab_obj *W2=rl::dl_leaf(rl::get_mat(ag,F("W2").c_str())), *b2=rl::dl_leaf(rl::get_mat(ag,F("b2").c_str()));
    matlab_obj *x=rl::dl_leaf(Xsa);
    matlab_obj *q=rl::dl_add(rl::dl_mm(W2, rl::dl_relu(rl::dl_add(rl::dl_mm(W1,x),b1))), b2);
    matlab_obj *loss=rl::dl_mse(q, rl::dl_leaf(Y));
    rl::adam_step(rl::get_mat(ag,F("W1").c_str()), rl::dl_grad(loss,W1), rl::get_mat(ag,("m"+F("W1")).c_str()), rl::get_mat(ag,("v"+F("W1")).c_str()), lr, t);
    rl::adam_step(rl::get_mat(ag,F("b1").c_str()), rl::dl_grad(loss,b1), rl::get_mat(ag,("m"+F("b1")).c_str()), rl::get_mat(ag,("v"+F("b1")).c_str()), lr, t);
    rl::adam_step(rl::get_mat(ag,F("W2").c_str()), rl::dl_grad(loss,W2), rl::get_mat(ag,("m"+F("W2")).c_str()), rl::get_mat(ag,("v"+F("W2")).c_str()), lr, t);
    rl::adam_step(rl::get_mat(ag,F("b2").c_str()), rl::dl_grad(loss,b2), rl::get_mat(ag,("m"+F("b2")).c_str()), rl::get_mat(ag,("v"+F("b2")).c_str()), lr, t);
    matlab_dlnet_reset0();
}

matlab_mat *matlab_rl_td3_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(agent, "ActDim"));
    double actLimit = rl::get_f64(agent, "ActLimit");
    double gamma = rl::get_f64(agent, "DiscountFactor"); if (gamma <= 0) gamma = 0.99;
    double lr    = rl::get_f64(agent, "LearnRate");      if (lr <= 0) lr = 1e-3;
    double tau   = rl::get_f64(agent, "Tau");            if (tau <= 0) tau = 5e-3;
    double actorLr = rl::get_f64(agent, "ActorLR");    if (actorLr <= 0) actorLr = lr;
    double rscale  = rl::get_f64(agent, "RewardScale"); if (rscale <= 0) rscale = 0.1;
    // TD3-specific knobs (action units): exploration noise, target-smoothing
    // noise + its clip, and the actor/target update period.
    double explSigma  = rl::get_f64(agent, "ExplNoise");   if (explSigma  <= 0) explSigma  = 0.2;
    double polNoise   = rl::get_f64(agent, "PolicyNoise"); if (polNoise   <= 0) polNoise   = 0.2;
    double noiseClip  = rl::get_f64(agent, "NoiseClip");   if (noiseClip  <= 0) noiseClip  = 0.5;
    int64_t polDelay  = static_cast<int64_t>(rl::get_f64(agent, "PolicyDelay")); if (polDelay < 1) polDelay = 2;

    int64_t maxEp   = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));   if (maxEp < 1) maxEp = 1;
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode")); if (maxStep < 1) maxStep = 200;

    int prevFreeFwd = matlab_dlnet_set_free_forward(1);   // bounded memory across grad calls

    const int64_t batch = 64, minReplay = batch;
    const size_t  replayCap = 50000;
    int64_t Wd = 2 * obsDim + actDim + 2;
    std::vector<double> replay; size_t rcount = 0, rhead = 0;

    matlab_mat *stats = matlab_zeros(static_cast<double>(maxEp), 1);
    double tstep = 0.0, atstep = 0.0;
    auto clip = [](double v, double lo, double hi){ return v < lo ? lo : (v > hi ? hi : v); };
    std::vector<double> sint(2), obs(static_cast<size_t>(obsDim)), nobs(static_cast<size_t>(obsDim)), act(static_cast<size_t>(actDim));

    const double ouTheta = 0.15;
    for (int64_t ep = 0; ep < maxEp; ++ep) {
        sint[0] = M_PI + (rl::urand() * 2 - 1) * 0.1; sint[1] = (rl::urand() * 2 - 1) * 0.1;
        rl::pendulum_obs(sint.data(), obs.data());
        std::vector<double> ou(static_cast<size_t>(actDim), 0.0);
        double epReturn = 0.0;
        for (int64_t step = 0; step < maxStep; ++step) {
            rl::actor_forward(rl::get_mat(agent,"aW1"), rl::get_mat(agent,"ab1"),
                              rl::get_mat(agent,"aW2"), rl::get_mat(agent,"ab2"), obs.data(), actLimit, act.data());
            // Temporally-correlated (Ornstein-Uhlenbeck) exploration noise —
            // pumps energy more effectively than i.i.d. Gaussian on the
            // swing-up, which markedly stabilises convergence across seeds.
            for (int64_t k = 0; k < actDim; ++k) {
                ou[static_cast<size_t>(k)] += -ouTheta * ou[static_cast<size_t>(k)] + explSigma * rl::urandn();
                act[static_cast<size_t>(k)] = clip(act[static_cast<size_t>(k)] + ou[static_cast<size_t>(k)], -actLimit, actLimit);
            }
            double r = rl::pendulum_step(sint.data(), act[0], actLimit);
            rl::pendulum_obs(sint.data(), nobs.data());
            epReturn += r;

            std::vector<double> tr(static_cast<size_t>(Wd));
            for (int64_t k = 0; k < obsDim; ++k) tr[static_cast<size_t>(k)] = obs[static_cast<size_t>(k)];
            for (int64_t k = 0; k < actDim; ++k) tr[static_cast<size_t>(obsDim + k)] = act[static_cast<size_t>(k)];
            tr[static_cast<size_t>(obsDim + actDim)] = r;
            for (int64_t k = 0; k < obsDim; ++k) tr[static_cast<size_t>(obsDim + actDim + 1 + k)] = nobs[static_cast<size_t>(k)];
            tr[static_cast<size_t>(Wd - 1)] = 0.0;
            if (replay.size() < replayCap * static_cast<size_t>(Wd)) { replay.insert(replay.end(), tr.begin(), tr.end()); rcount++; }
            else { std::copy(tr.begin(), tr.end(), replay.begin() + static_cast<long>(rhead * static_cast<size_t>(Wd))); rhead = (rhead + 1) % replayCap; }

            if (rcount >= static_cast<size_t>(minReplay)) {
                matlab_mat *Xsa = matlab_zeros(static_cast<double>(obsDim + actDim), static_cast<double>(batch));
                matlab_mat *Y   = matlab_zeros(1, static_cast<double>(batch));
                matlab_mat *S   = matlab_zeros(static_cast<double>(obsDim), static_cast<double>(batch));
                for (int64_t b = 0; b < batch; ++b) {
                    size_t idx = static_cast<size_t>(rl::urand() * static_cast<double>(rcount)) % rcount;
                    double *row = &replay[idx * static_cast<size_t>(Wd)];
                    double *oj = row, *aj = row + obsDim, rj = row[obsDim + actDim];
                    double *npj = row + obsDim + actDim + 1;
                    for (int64_t k = 0; k < obsDim; ++k) { Xsa->data[k*batch+b]=oj[k]; S->data[k*batch+b]=oj[k]; }
                    for (int64_t k = 0; k < actDim; ++k) Xsa->data[(obsDim+k)*batch+b]=aj[k];
                    // target action with policy smoothing
                    std::vector<double> na(static_cast<size_t>(actDim)), sap(static_cast<size_t>(obsDim+actDim));
                    rl::actor_forward(rl::get_mat(agent,"taW1"), rl::get_mat(agent,"tab1"),
                                      rl::get_mat(agent,"taW2"), rl::get_mat(agent,"tab2"), npj, actLimit, na.data());
                    for (int64_t k = 0; k < actDim; ++k) {
                        double nz = clip(polNoise * rl::urandn(), -noiseClip, noiseClip);
                        na[static_cast<size_t>(k)] = clip(na[static_cast<size_t>(k)] + nz, -actLimit, actLimit);
                    }
                    for (int64_t k = 0; k < obsDim; ++k) sap[static_cast<size_t>(k)] = npj[k];
                    for (int64_t k = 0; k < actDim; ++k) sap[static_cast<size_t>(obsDim+k)] = na[static_cast<size_t>(k)];
                    // twin target critics -> min
                    double q1 = rl::critic_forward(rl::get_mat(agent,"tcW1"), rl::get_mat(agent,"tcb1"),
                                                   rl::get_mat(agent,"tcW2"), rl::get_mat(agent,"tcb2"), sap.data());
                    double q2 = rl::critic_forward(rl::get_mat(agent,"tc2W1"), rl::get_mat(agent,"tc2b1"),
                                                   rl::get_mat(agent,"tc2W2"), rl::get_mat(agent,"tc2b2"), sap.data());
                    Y->data[b] = rscale * rj + gamma * std::min(q1, q2);
                }
                tstep += 1.0;
                td3_critic_step(agent, "c",  Xsa, Y, lr, tstep);
                td3_critic_step(agent, "c2", Xsa, Y, lr, tstep);
                // delayed actor + target update
                if (static_cast<int64_t>(tstep) % polDelay == 0) {
                    atstep += 1.0;
                    ddpg_actor_step(agent, S, actLimit, actorLr, atstep);
                    const char *P[12] = {"aW1","ab1","aW2","ab2","cW1","cb1","cW2","cb2","c2W1","c2b1","c2W2","c2b2"};
                    for (const char *p : P)
                        rl::soft_update(rl::get_mat(agent, (std::string("t")+p).c_str()), rl::get_mat(agent, p), tau);
                }
                rl::free_mat(Xsa); rl::free_mat(Y); rl::free_mat(S);
            }
            for (int64_t k = 0; k < obsDim; ++k) obs[static_cast<size_t>(k)] = nobs[static_cast<size_t>(k)];
        }
        stats->data[ep] = epReturn;
    }
    matlab_dlnet_set_free_forward(prevFreeFwd);
    return stats;
}

// sim(td3Agent, env) — greedy rollout; identical to the DDPG greedy sim
// (same deterministic actor field layout).
matlab_mat *matlab_rl_td3_sim(matlab_obj *agent, matlab_obj *env) {
    return matlab_rl_ddpg_sim(agent, env);
}

// ===========================================================================
// SAC — Soft Actor-Critic (fixed entropy coefficient).  rlSACAgent(obs,act)
// ===========================================================================
// Off-policy max-entropy continuous control.  A stochastic SQUASHED-GAUSSIAN
// actor (shared trunk → mean + log-std heads; a = tanh(μ + σ·ε)·limit via the
// reparameterisation trick) is trained to maximise Q1(s,a) − α·log π(a|s);
// twin critics regress the soft TD target
//   y = r + γ·( min(Qt1,Qt2) − α·log π(a'|s') ),   a' ~ π(·|s').
// The actor's log-prob (with the tanh-squash change-of-variables correction
// log(1−tanh²u) = 2(log2 − u − softplus(−2u))) is differentiated through the
// reused autodiff tape.  The entropy temperature α is fixed — the canonical
// fixed-coefficient SAC variant (automatic-temperature tuning is carved out).

static const double kSacLogStdMin = -5.0, kSacLogStdMax = 2.0;
static inline double sac_softplus(double z) {
    return z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z));
}
// log-std bounded smoothly to [min,max] via tanh (matches the tape path).
static inline double sac_ls(double raw) {
    double mid = 0.5 * (kSacLogStdMin + kSacLogStdMax);
    double half = 0.5 * (kSacLogStdMax - kSacLogStdMin);
    return mid + half * std::tanh(raw);
}

// Host squashed-Gaussian forward.  Fills act[]; returns log π(act|s).  When
// deterministic, uses the mean (greedy) and the returned log-prob is unused.
static double sac_sample(matlab_obj *ag, const double *x, double actLimit, bool deterministic, double *act) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(ag, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(ag, "ActDim"));
    matlab_mat *W1=rl::get_mat(ag,"aW1"), *b1=rl::get_mat(ag,"ab1");
    matlab_mat *Wmu=rl::get_mat(ag,"aWmu"), *bmu=rl::get_mat(ag,"abmu");
    matlab_mat *Wls=rl::get_mat(ag,"aWls"), *bls=rl::get_mat(ag,"abls");
    int64_t H = W1->rows;
    std::vector<double> h(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; ++i) { double s = b1->data[i];
        for (int64_t j = 0; j < obsDim; ++j) s += W1->data[i*obsDim+j]*x[j];
        h[static_cast<size_t>(i)] = s > 0 ? s : 0.0; }
    const double LOG2 = std::log(2.0), HALF_LOG2PI = 0.5*std::log(2.0*M_PI);
    double logp = 0.0;
    for (int64_t k = 0; k < actDim; ++k) {
        double mu = bmu->data[k], lr = bls->data[k];
        for (int64_t i = 0; i < H; ++i) { mu += Wmu->data[k*H+i]*h[static_cast<size_t>(i)]; lr += Wls->data[k*H+i]*h[static_cast<size_t>(i)]; }
        double ls = sac_ls(lr), sd = std::exp(ls);
        double eps = deterministic ? 0.0 : rl::urandn();
        double u = mu + sd * eps;
        act[k] = std::tanh(u) * actLimit;
        if (!deterministic)
            logp += -0.5*eps*eps - ls - HALF_LOG2PI - std::log(actLimit) - 2.0*(LOG2 - u - sac_softplus(-2.0*u));
    }
    return logp;
}

void matlab_rl_sac_init(matlab_obj *agent, matlab_obj *obsInfo, matlab_obj *actInfo) {
    matlab_mat *od = rl::get_mat(obsInfo, "Dimension");
    int64_t obsDim = (od && od->rows*od->cols) ? static_cast<int64_t>(od->data[0]) : 1;
    matlab_mat *ad = rl::get_mat(actInfo, "Dimension");
    int64_t actDim = (ad && ad->rows*ad->cols) ? static_cast<int64_t>(ad->data[0]) : 1;
    double actLimit = rl::get_f64(actInfo, "Limit"); if (actLimit <= 0) actLimit = 1.0;
    int64_t H = kDdpgHidden, cin = obsDim + actDim;
    auto he = [&](int64_t r, int64_t c){ matlab_mat *m=matlab_randn(static_cast<double>(r),static_cast<double>(c)); double s=std::sqrt(2.0/static_cast<double>(c)); for(int64_t k=0;k<r*c;++k) m->data[k]*=s; return m; };
    auto small=[&](int64_t r,int64_t c){ matlab_mat *m=matlab_randn(static_cast<double>(r),static_cast<double>(c)); for(int64_t k=0;k<r*c;++k) m->data[k]*=1e-3; return m; };
    auto z=[&](int64_t r,int64_t c){ return matlab_zeros(static_cast<double>(r),static_cast<double>(c)); };
    // actor: shared trunk + mean head + log-std head
    rl::set_mat(agent,"aW1",he(H,obsDim));   rl::set_mat(agent,"ab1",z(H,1));
    rl::set_mat(agent,"aWmu",small(actDim,H)); rl::set_mat(agent,"abmu",z(actDim,1));
    rl::set_mat(agent,"aWls",small(actDim,H)); rl::set_mat(agent,"abls",z(actDim,1));
    // twin critics "c","c2" (+ targets + moments) — same layout as TD3.
    rl::set_mat(agent,"cW1",he(H,cin));  rl::set_mat(agent,"cb1",z(H,1));  rl::set_mat(agent,"cW2",small(1,H)); rl::set_mat(agent,"cb2",z(1,1));
    rl::set_mat(agent,"c2W1",he(H,cin)); rl::set_mat(agent,"c2b1",z(H,1)); rl::set_mat(agent,"c2W2",small(1,H)); rl::set_mat(agent,"c2b2",z(1,1));
    const char *AP[6] = {"aW1","ab1","aWmu","abmu","aWls","abls"};
    for (const char *p : AP) { matlab_mat *pm=rl::get_mat(agent,p);
        rl::set_mat(agent,(std::string("m")+p).c_str(), z(pm->rows,pm->cols));
        rl::set_mat(agent,(std::string("v")+p).c_str(), z(pm->rows,pm->cols)); }
    const char *CP[8] = {"cW1","cb1","cW2","cb2","c2W1","c2b1","c2W2","c2b2"};
    for (const char *p : CP) { rl::set_mat(agent,(std::string("t")+p).c_str(), rl::clone_mat(rl::get_mat(agent,p)));
        matlab_mat *pm=rl::get_mat(agent,p);
        rl::set_mat(agent,(std::string("m")+p).c_str(), z(pm->rows,pm->cols));
        rl::set_mat(agent,(std::string("v")+p).c_str(), z(pm->rows,pm->cols)); }
    rl::set_f64(agent,"ObsDim",static_cast<double>(obsDim)); rl::set_f64(agent,"ActDim",static_cast<double>(actDim));
    rl::set_f64(agent,"ActLimit",actLimit);     rl::set_f64(agent,"HiddenSize",static_cast<double>(H));
}

// SAC actor step: reparameterised squashed-Gaussian, loss = α·Σlogπ − ΣQ1.
static void sac_actor_step(matlab_obj *ag, matlab_mat *S, matlab_mat *Eps, double actLimit, double alpha, double lr, double t) {
    matlab_obj *aW1=rl::dl_leaf(rl::get_mat(ag,"aW1")), *ab1=rl::dl_leaf(rl::get_mat(ag,"ab1"));
    matlab_obj *aWmu=rl::dl_leaf(rl::get_mat(ag,"aWmu")), *abmu=rl::dl_leaf(rl::get_mat(ag,"abmu"));
    matlab_obj *aWls=rl::dl_leaf(rl::get_mat(ag,"aWls")), *abls=rl::dl_leaf(rl::get_mat(ag,"abls"));
    matlab_obj *cW1=rl::dl_leaf(rl::get_mat(ag,"cW1")), *cb1=rl::dl_leaf(rl::get_mat(ag,"cb1"));
    matlab_obj *cW2=rl::dl_leaf(rl::get_mat(ag,"cW2")), *cb2=rl::dl_leaf(rl::get_mat(ag,"cb2"));
    matlab_obj *sL=rl::dl_leaf(S), *epsL=rl::dl_leaf(Eps);
    auto scalar=[&](double v){ matlab_mat *m=matlab_zeros(1,1); m->data[0]=v; return rl::dl_leaf(m); };
    matlab_obj *h=rl::dl_relu(rl::dl_add(rl::dl_mm(aW1,sL),ab1));
    matlab_obj *mu=rl::dl_add(rl::dl_mm(aWmu,h),abmu);
    matlab_obj *lsRaw=rl::dl_add(rl::dl_mm(aWls,h),abls);
    double mid=0.5*(kSacLogStdMin+kSacLogStdMax), half=0.5*(kSacLogStdMax-kSacLogStdMin);
    matlab_obj *ls=rl::dl_add(rl::dl_times(rl::dl_tanh(lsRaw), scalar(half)), scalar(mid));
    matlab_obj *sd=rl::dl_exp(ls);
    matlab_obj *u=rl::dl_add(mu, rl::dl_times(sd, epsL));
    matlab_obj *a=rl::dl_times(rl::dl_tanh(u), scalar(actLimit));
    // logπ variable part (constants dropped): 2u − ls + 2·softplus(−2u)
    matlab_obj *sp=rl::dl_softplus(rl::dl_times(u, scalar(-2.0)));
    matlab_obj *logpVar=rl::dl_add(rl::dl_sub(rl::dl_times(u, scalar(2.0)), ls), rl::dl_times(sp, scalar(2.0)));
    matlab_obj *sa=rl::dl_vcat(sL, a);
    matlab_obj *q=rl::dl_add(rl::dl_mm(cW2, rl::dl_relu(rl::dl_add(rl::dl_mm(cW1,sa),cb1))), cb2);
    matlab_obj *loss=rl::dl_sub(rl::dl_times(rl::dl_sum(logpVar), scalar(alpha)), rl::dl_sum(q));
    const char *AP[6] = {"aW1","ab1","aWmu","abmu","aWls","abls"};
    matlab_obj *leaves[6] = {aW1,ab1,aWmu,abmu,aWls,abls};
    for (int i = 0; i < 6; ++i)
        rl::adam_step(rl::get_mat(ag,AP[i]), rl::dl_grad(loss,leaves[i]), rl::get_mat(ag,(std::string("m")+AP[i]).c_str()), rl::get_mat(ag,(std::string("v")+AP[i]).c_str()), lr, t);
    matlab_dlnet_reset0();
}

matlab_mat *matlab_rl_sac_train(matlab_obj *agent, matlab_obj *env, matlab_obj *opts) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(agent, "ActDim"));
    double actLimit = rl::get_f64(agent, "ActLimit");
    double gamma = rl::get_f64(agent, "DiscountFactor"); if (gamma <= 0) gamma = 0.99;
    double lr    = rl::get_f64(agent, "LearnRate");      if (lr <= 0) lr = 1e-3;
    double tau   = rl::get_f64(agent, "Tau");            if (tau <= 0) tau = 5e-3;
    double rscale = rl::get_f64(agent, "RewardScale"); if (rscale <= 0) rscale = 0.1;
    double alpha  = rl::get_f64(agent, "EntropyWeight"); if (alpha <= 0) alpha = 0.2;

    int64_t maxEp   = static_cast<int64_t>(rl::get_f64(opts, "MaxEpisodes"));   if (maxEp < 1) maxEp = 1;
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(opts, "MaxStepsPerEpisode")); if (maxStep < 1) maxStep = 200;

    int prevFreeFwd = matlab_dlnet_set_free_forward(1);
    const int64_t batch = 64, minReplay = batch;
    const size_t replayCap = 50000;
    int64_t Wd = 2*obsDim + actDim + 2;
    std::vector<double> replay; size_t rcount = 0, rhead = 0;
    matlab_mat *stats = matlab_zeros(static_cast<double>(maxEp), 1);
    double tstep = 0.0;
    std::vector<double> sint(2), obs(static_cast<size_t>(obsDim)), nobs(static_cast<size_t>(obsDim)), act(static_cast<size_t>(actDim));

    for (int64_t ep = 0; ep < maxEp; ++ep) {
        sint[0] = M_PI + (rl::urand()*2-1)*0.1; sint[1] = (rl::urand()*2-1)*0.1;
        rl::pendulum_obs(sint.data(), obs.data());
        double epReturn = 0.0;
        for (int64_t step = 0; step < maxStep; ++step) {
            sac_sample(agent, obs.data(), actLimit, false, act.data());   // stochastic policy = exploration
            double r = rl::pendulum_step(sint.data(), act[0], actLimit);
            rl::pendulum_obs(sint.data(), nobs.data());
            epReturn += r;
            std::vector<double> tr(static_cast<size_t>(Wd));
            for (int64_t k=0;k<obsDim;++k) tr[static_cast<size_t>(k)]=obs[static_cast<size_t>(k)];
            for (int64_t k=0;k<actDim;++k) tr[static_cast<size_t>(obsDim+k)]=act[static_cast<size_t>(k)];
            tr[static_cast<size_t>(obsDim+actDim)]=r;
            for (int64_t k=0;k<obsDim;++k) tr[static_cast<size_t>(obsDim+actDim+1+k)]=nobs[static_cast<size_t>(k)];
            tr[static_cast<size_t>(Wd-1)]=0.0;
            if (replay.size() < replayCap*static_cast<size_t>(Wd)) { replay.insert(replay.end(), tr.begin(), tr.end()); rcount++; }
            else { std::copy(tr.begin(), tr.end(), replay.begin()+static_cast<long>(rhead*static_cast<size_t>(Wd))); rhead=(rhead+1)%replayCap; }

            if (rcount >= static_cast<size_t>(minReplay)) {
                matlab_mat *Xsa=matlab_zeros(static_cast<double>(obsDim+actDim),static_cast<double>(batch));
                matlab_mat *Y=matlab_zeros(1,static_cast<double>(batch));
                matlab_mat *S=matlab_zeros(static_cast<double>(obsDim),static_cast<double>(batch));
                matlab_mat *Eps=matlab_randn(static_cast<double>(actDim),static_cast<double>(batch));   // fresh reparam noise for the actor
                for (int64_t b=0;b<batch;++b) {
                    size_t idx=static_cast<size_t>(rl::urand()*static_cast<double>(rcount))%rcount;
                    double *row=&replay[idx*static_cast<size_t>(Wd)];
                    double *oj=row,*aj=row+obsDim,rj=row[obsDim+actDim];
                    double *npj=row+obsDim+actDim+1;
                    for (int64_t k=0;k<obsDim;++k){ Xsa->data[k*batch+b]=oj[k]; S->data[k*batch+b]=oj[k]; }
                    for (int64_t k=0;k<actDim;++k) Xsa->data[(obsDim+k)*batch+b]=aj[k];
                    // a' ~ current policy at s'; soft target uses min twin critics − α·logπ'
                    std::vector<double> na(static_cast<size_t>(actDim)), sap(static_cast<size_t>(obsDim+actDim));
                    double logpn = sac_sample(agent, npj, actLimit, false, na.data());
                    for (int64_t k=0;k<obsDim;++k) sap[static_cast<size_t>(k)]=npj[k];
                    for (int64_t k=0;k<actDim;++k) sap[static_cast<size_t>(obsDim+k)]=na[static_cast<size_t>(k)];
                    double q1=rl::critic_forward(rl::get_mat(agent,"tcW1"),rl::get_mat(agent,"tcb1"),rl::get_mat(agent,"tcW2"),rl::get_mat(agent,"tcb2"),sap.data());
                    double q2=rl::critic_forward(rl::get_mat(agent,"tc2W1"),rl::get_mat(agent,"tc2b1"),rl::get_mat(agent,"tc2W2"),rl::get_mat(agent,"tc2b2"),sap.data());
                    Y->data[b]=rscale*rj + gamma*(std::min(q1,q2) - alpha*logpn);
                }
                tstep += 1.0;
                td3_critic_step(agent,"c", Xsa,Y,lr,tstep);
                td3_critic_step(agent,"c2",Xsa,Y,lr,tstep);
                sac_actor_step(agent,S,Eps,actLimit,alpha,lr,tstep);
                const char *CP[8]={"cW1","cb1","cW2","cb2","c2W1","c2b1","c2W2","c2b2"};
                for (const char *p:CP) rl::soft_update(rl::get_mat(agent,(std::string("t")+p).c_str()), rl::get_mat(agent,p), tau);
                rl::free_mat(Xsa); rl::free_mat(Y); rl::free_mat(S); rl::free_mat(Eps);
            }
            for (int64_t k=0;k<obsDim;++k) obs[static_cast<size_t>(k)]=nobs[static_cast<size_t>(k)];
        }
        stats->data[ep]=epReturn;
    }
    matlab_dlnet_set_free_forward(prevFreeFwd);
    return stats;
}

// sim(sacAgent, env) — greedy (mean-action, noise-free) pendulum rollout.
matlab_mat *matlab_rl_sac_sim(matlab_obj *agent, matlab_obj *env) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t actDim = static_cast<int64_t>(rl::get_f64(agent, "ActDim"));
    double actLimit = rl::get_f64(agent, "ActLimit");
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(env, "MaxSteps")); if (maxStep < 1) maxStep = 200;
    matlab_mat *out = matlab_zeros(1,1);
    std::vector<double> sint(2), obs(static_cast<size_t>(obsDim)), act(static_cast<size_t>(actDim));
    sint[0] = M_PI; sint[1] = 0.0;
    rl::pendulum_obs(sint.data(), obs.data());
    double total = 0.0;
    for (int64_t step = 0; step < maxStep; ++step) {
        sac_sample(agent, obs.data(), actLimit, true, act.data());   // deterministic (mean)
        total += rl::pendulum_step(sint.data(), act[0], actLimit);
        rl::pendulum_obs(sint.data(), obs.data());
    }
    out->data[0] = total;
    return out;
}

// ===========================================================================
// Policy-use accessors — getAction / getMaxQValue / getGreedyPolicy
// ===========================================================================
// These query a trained agent (or a greedy policy extracted from it).  They
// dispatch on the stored representation: a network agent (DQN/PG, has "W1")
// runs the MLP forward; a tabular agent (Q/SARSA, has "QTable") indexes the
// table by the observation's state number.

// getAction(agent, obs) → greedy action.  Network agents return the 1-based
// action index argmaxₐ Q(obs,a) (DQN) / argmaxₐ π(a|obs) (PG); tabular agents
// take obs as the scalar state number and return argmax over the Q row.
matlab_mat *matlab_rl_get_action(matlab_obj *agent, matlab_mat *obs) {
    matlab_mat *out = matlab_zeros(1, 1);
    // NB: matlab_obj_get_mat returns a non-null EMPTY (0×0) matrix for an unset
    // field, so "is this network present?" must test rows>0, not the pointer.
    // DDPG (continuous actor): return the deterministic continuous action.
    matlab_mat *aW1 = rl::get_mat(agent, "aW1");
    if (aW1 && aW1->rows > 0) {
        matlab_mat *aW2 = rl::get_mat(agent, "aW2");
        int64_t actDim = (aW2 && aW2->rows > 0) ? aW2->rows : 1;
        matlab_mat *a = matlab_zeros(static_cast<double>(actDim), 1);
        rl::actor_forward(aW1, rl::get_mat(agent, "ab1"), aW2,
                          rl::get_mat(agent, "ab2"), obs ? obs->data : nullptr,
                          rl::get_f64(agent, "ActLimit"), a->data);
        return a;
    }
    matlab_mat *W1 = rl::get_mat(agent, "W1");
    if (W1 && W1->rows > 0) {
        matlab_mat *W2 = rl::get_mat(agent, "W2");
        int64_t A = W2 ? W2->rows : 1;
        std::vector<double> q(static_cast<size_t>(A));
        rl::mlp_forward(W1, rl::get_mat(agent, "b1"), W2, rl::get_mat(agent, "b2"),
                        obs ? obs->data : nullptr, q.data());
        int64_t a = 1; for (int64_t j = 1; j < A; ++j) if (q[static_cast<size_t>(j)] > q[static_cast<size_t>(a - 1)]) a = j + 1;
        out->data[0] = static_cast<double>(a);
        return out;
    }
    matlab_mat *Q = rl::get_mat(agent, "QTable");
    if (Q && Q->rows > 0 && obs) {
        int64_t A = Q->cols, s = static_cast<int64_t>(obs->data[0]);
        if (s >= 1 && s <= Q->rows) out->data[0] = static_cast<double>(rl::argmax_row(Q->data, s, A));
    }
    return out;
}

// getMaxQValue(agent, obs) → maxₐ Q(obs,a).
matlab_mat *matlab_rl_get_maxq(matlab_obj *agent, matlab_mat *obs) {
    matlab_mat *out = matlab_zeros(1, 1);
    matlab_mat *W1 = rl::get_mat(agent, "W1");
    if (W1 && W1->rows > 0) {
        matlab_mat *W2 = rl::get_mat(agent, "W2");
        int64_t A = W2 ? W2->rows : 1;
        std::vector<double> q(static_cast<size_t>(A));
        rl::mlp_forward(W1, rl::get_mat(agent, "b1"), W2, rl::get_mat(agent, "b2"),
                        obs ? obs->data : nullptr, q.data());
        double mx = q[0]; for (int64_t j = 1; j < A; ++j) mx = std::max(mx, q[static_cast<size_t>(j)]);
        out->data[0] = mx;
        return out;
    }
    matlab_mat *Q = rl::get_mat(agent, "QTable");
    if (Q && Q->rows > 0 && obs) {
        int64_t A = Q->cols, s = static_cast<int64_t>(obs->data[0]);
        if (s >= 1 && s <= Q->rows) out->data[0] = rl::max_row(Q->data, s, A);
    }
    return out;
}

// getGreedyPolicy(agent) → rlMaxQPolicy carrying a copy of the agent's network
// (or Q table).  getAction on the policy then dispatches identically.
void matlab_rl_greedy_policy(matlab_obj *policy, matlab_obj *agent) {
    matlab_mat *W1 = rl::get_mat(agent, "W1");
    if (W1 && W1->rows > 0) {
        rl::set_mat(policy, "W1", rl::clone_mat(W1));
        rl::set_mat(policy, "b1", rl::clone_mat(rl::get_mat(agent, "b1")));
        rl::set_mat(policy, "W2", rl::clone_mat(rl::get_mat(agent, "W2")));
        rl::set_mat(policy, "b2", rl::clone_mat(rl::get_mat(agent, "b2")));
    } else {
        rl::set_mat(policy, "QTable", rl::clone_mat(rl::get_mat(agent, "QTable")));
    }
}

// sim(pgAgent, env) — greedy (argmax-probability) rollout; returns steps held.
matlab_mat *matlab_rl_pg_sim(matlab_obj *agent, matlab_obj *env) {
    int64_t obsDim = static_cast<int64_t>(rl::get_f64(agent, "ObsDim"));
    int64_t A = static_cast<int64_t>(rl::get_f64(agent, "NumActions"));
    int64_t maxStep = static_cast<int64_t>(rl::get_f64(env, "MaxSteps"));
    if (maxStep < 1) maxStep = 500;
    matlab_mat *out = matlab_zeros(1, 1);
    std::vector<double> st(static_cast<size_t>(obsDim), 0.0), probs(static_cast<size_t>(A));
    double total = 0.0;
    for (int64_t step = 0; step < maxStep; ++step) {
        rl::mlp_policy(rl::get_mat(agent, "W1"), rl::get_mat(agent, "b1"),
                       rl::get_mat(agent, "W2"), rl::get_mat(agent, "b2"), st.data(), probs.data());
        int64_t a = 1; for (int64_t j = 1; j < A; ++j) if (probs[static_cast<size_t>(j)] > probs[static_cast<size_t>(a - 1)]) a = j + 1;
        double r; bool done;
        rl::cartpole_step(st.data(), a, &r, &done);
        total += r;
        if (done) break;
    }
    out->data[0] = total;
    return out;
}

}  // extern "C"
