// Deep Learning Toolbox runtime — Tiers 1–2 (inference + autodiff).
//
// Scope: the 2-D dense surface — `dlarray` over a reverse-mode automatic-
// differentiation tape.  An MLP forward pass written with natural operators
// (`W*X + b`, `relu(...)`, `softmax(...)`) both *evaluates* (T1 inference) and
// *records* onto a thread-local tape so `dlgradient(loss, var)` can sweep it
// (T2).  Convolution / 4-D `SSCB` tensors are carved (the runtime has no rank-N
// type yet); the object-array `dlnetwork`/layer-object container is carved (no
// classdef array literals) — both documented in docs/deep_learning_toolbox_roadmap.md.
//
// Storage: `dlarray` is a classdef carrier with two properties —
//   Data : the value matrix (matlab_mat)
//   Id   : the tape-node index (double; -1 = untracked constant)
// Every op is a real `dlarray` method (so operator overloading dispatches to
// it); the method allocates a fresh `dlarray` shell then calls the matching
// `matlab_dlnet_*` runtime entry, which computes the forward value, records a
// tape node, and populates the shell.  See the thread-local-tape pattern in
// runtime/toolbox/ident/runtime_ident.cpp (the lsqnonlin residual context).

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);

namespace dlnet {

inline matlab_mat *get_data(void *o) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(o), "Data", 4);
}
inline int get_id(void *o) {
    return static_cast<int>(matlab_obj_get_f64(reinterpret_cast<matlab_obj *>(o), "Id", 2));
}
inline void set_data(void *o, matlab_mat *m) {
    matlab_obj_set_mat(reinterpret_cast<matlab_obj *>(o), "Data", 4, m);
}
inline void set_id(void *o, int id) {
    matlab_obj_set_f64(reinterpret_cast<matlab_obj *>(o), "Id", 2, static_cast<double>(id));
}

inline matlab_mat *clone(const matlab_mat *m) {
    if (!m) return mat_alloc(0, 0);
    matlab_mat *o = mat_alloc(m->rows, m->cols);
    for (int64_t i = 0; i < m->rows * m->cols; ++i) o->data[i] = m->data[i];
    return o;
}

// ---- reverse-mode tape ----------------------------------------------------
// Opcodes.
enum { OP_LEAF, OP_ADD, OP_SUB, OP_MTIMES, OP_TIMES, OP_RELU, OP_SIGMOID,
       OP_TANH, OP_SOFTMAX, OP_SUM, OP_MEAN, OP_LOG, OP_EXP, OP_CE, OP_MSE,
       OP_LSTM, OP_TRANSPOSE, OP_EMBED, OP_GRU, OP_BILSTM, OP_LSTMP };

struct Node {
    int op;
    int p0, p1;            // parent node ids (-1 = none / not differentiable)
    matlab_mat *val;       // forward value (owned by the tape)
    matlab_mat *adj;       // adjoint accumulator (lazily allocated; nullptr = 0)
    // Multi-parent / multi-state ops (e.g. OP_LSTM).  Empty for everything else.
    std::vector<int>          auxParents;  // extra parent node ids beyond p0/p1
    std::vector<matlab_mat *> auxData;     // saved per-timestep tensors for BPTT
};

static thread_local std::vector<Node> g_tape;

inline int record(int op, int p0, int p1, matlab_mat *val) {
    Node n; n.op = op; n.p0 = p0; n.p1 = p1; n.val = val; n.adj = nullptr;
    g_tape.push_back(n);
    return static_cast<int>(g_tape.size()) - 1;
}

// adj += contribution (allocating the accumulator on first touch).
inline void accum(int id, const matlab_mat *contrib) {
    if (id < 0 || !contrib) return;
    Node &n = g_tape[id];
    if (!n.adj) { n.adj = mat_alloc(contrib->rows, contrib->cols);
                  for (int64_t i = 0; i < contrib->rows*contrib->cols; ++i) n.adj->data[i] = 0; }
    int64_t na = n.adj->rows * n.adj->cols, nc = contrib->rows * contrib->cols;
    if (na == nc) { for (int64_t i = 0; i < na; ++i) n.adj->data[i] += contrib->data[i]; }
    else if (n.adj->cols == 1 && contrib->rows == n.adj->rows) {
        // bias broadcast: sum the contribution across columns into the column adj.
        for (int64_t r = 0; r < contrib->rows; ++r) {
            double s = 0; for (int64_t c = 0; c < contrib->cols; ++c) s += contrib->data[r*contrib->cols+c];
            n.adj->data[r] += s;
        }
    } else if (na == 1) { double s = 0; for (int64_t i = 0; i < nc; ++i) s += contrib->data[i]; n.adj->data[0] += s; }
}

}  // namespace dlnet

extern "C" {

// dlarray(X): wrap a numeric matrix as a tracked leaf.
matlab_mat *matlab_dlnet_dlarray_init(void *obj_v, matlab_mat *X) {
    matlab_mat *v = dlnet::clone(X);
    dlnet::set_data(obj_v, v);
    dlnet::set_id(obj_v, dlnet::record(dlnet::OP_LEAF, -1, -1, v));
    return mat_alloc(0, 0);
}

// extractdata(dl) -> the underlying matrix.
matlab_mat *matlab_dlnet_extractdata(void *obj_v) {
    matlab_mat *d = dlnet::get_data(obj_v);
    return dlnet::clone(d);
}

// ---- binary ops: result obj r populated from operands a, b ----------------
static matlab_mat *bin_forward(int op, matlab_mat *A, matlab_mat *B) {
    int64_t m = A->rows, n = A->cols;
    if (op == dlnet::OP_MTIMES) {
        matlab_mat *C = mat_alloc(A->rows, B->cols);
        for (int64_t i = 0; i < A->rows; ++i)
            for (int64_t j = 0; j < B->cols; ++j) {
                double s = 0; for (int64_t k = 0; k < A->cols; ++k) s += A->data[i*A->cols+k]*B->data[k*B->cols+j];
                C->data[i*C->cols+j] = s;
            }
        return C;
    }
    matlab_mat *C = mat_alloc(m, n);
    bool bcol = (B->cols == 1 && B->rows == m && n > 1);   // bias-style broadcast
    for (int64_t r = 0; r < m; ++r)
        for (int64_t c = 0; c < n; ++c) {
            double a = A->data[r*n+c];
            double b = bcol ? B->data[r] : B->data[(B->rows==m&&B->cols==n)? r*n+c : (r* (B->cols) + (c % B->cols))];
            double v = 0;
            if (op == dlnet::OP_ADD)   v = a + b;
            if (op == dlnet::OP_SUB)   v = a - b;
            if (op == dlnet::OP_TIMES) v = a * b;
            C->data[r*n+c] = v;
        }
    return C;
}

static matlab_mat *bin_op(void *r, void *a, void *b, int op) {
    matlab_mat *A = dlnet::get_data(a), *B = dlnet::get_data(b);
    matlab_mat *C = bin_forward(op, A, B);
    dlnet::set_data(r, C);
    dlnet::set_id(r, dlnet::record(op, dlnet::get_id(a), dlnet::get_id(b), C));
    return mat_alloc(0, 0);
}
matlab_mat *matlab_dlnet_plus  (void *r, void *a, void *b) { return bin_op(r, a, b, dlnet::OP_ADD); }
matlab_mat *matlab_dlnet_minus (void *r, void *a, void *b) { return bin_op(r, a, b, dlnet::OP_SUB); }
matlab_mat *matlab_dlnet_mtimes(void *r, void *a, void *b) { return bin_op(r, a, b, dlnet::OP_MTIMES); }
matlab_mat *matlab_dlnet_times (void *r, void *a, void *b) { return bin_op(r, a, b, dlnet::OP_TIMES); }

// ---- unary / activation ops -----------------------------------------------
static matlab_mat *un_forward(int op, matlab_mat *X) {
    int64_t m = X->rows, n = X->cols;
    if (op == dlnet::OP_SUM)  { matlab_mat *o = mat_alloc(1,1); double s=0; for (int64_t i=0;i<m*n;++i) s+=X->data[i]; o->data[0]=s; return o; }
    if (op == dlnet::OP_MEAN) { matlab_mat *o = mat_alloc(1,1); double s=0; for (int64_t i=0;i<m*n;++i) s+=X->data[i]; o->data[0]= (m*n? s/(m*n):0); return o; }
    matlab_mat *Y = mat_alloc(m, n);
    if (op == dlnet::OP_SOFTMAX) {
        for (int64_t c = 0; c < n; ++c) {
            double mx = -1e300; for (int64_t r=0;r<m;++r) mx = std::max(mx, X->data[r*n+c]);
            double sm = 0; for (int64_t r=0;r<m;++r){ double e=std::exp(X->data[r*n+c]-mx); Y->data[r*n+c]=e; sm+=e; }
            for (int64_t r=0;r<m;++r) Y->data[r*n+c] /= (sm>0?sm:1);
        }
        return Y;
    }
    for (int64_t i = 0; i < m*n; ++i) {
        double x = X->data[i], y = x;
        if (op == dlnet::OP_RELU)    y = x > 0 ? x : 0;
        if (op == dlnet::OP_SIGMOID) y = 1.0/(1.0+std::exp(-x));
        if (op == dlnet::OP_TANH)    y = std::tanh(x);
        if (op == dlnet::OP_LOG)     y = std::log(x);
        if (op == dlnet::OP_EXP)     y = std::exp(x);
        Y->data[i] = y;
    }
    return Y;
}
static matlab_mat *un_op(void *r, void *x, int op) {
    matlab_mat *X = dlnet::get_data(x);
    matlab_mat *Y = un_forward(op, X);
    dlnet::set_data(r, Y);
    dlnet::set_id(r, dlnet::record(op, dlnet::get_id(x), -1, Y));
    return mat_alloc(0, 0);
}
// embed(E, idx) -- wordEmbeddingLayer's lookup.
//   E    D × V       embedding matrix (D-dim per token, V vocab)
//   idx  1 × N       integer token ids (MATLAB 1-based; expected to be a
//                    plain numeric matrix, NOT a dlarray)
//   Y    D × N       Y(:,n) = E(:, idx(n))
// Forward = gather columns; backward = scatter-add into dE(:, idx(n)).
matlab_mat *matlab_dlnet_embed(void *robj, void *Ev, matlab_mat *idx) {
    using namespace dlnet;
    matlab_mat *E = get_data(Ev);
    int D = static_cast<int>(E->rows);
    int N = static_cast<int>(idx->rows * idx->cols);
    matlab_mat *Y = mat_alloc(D, N);
    for (int n = 0; n < N; ++n) {
        int j = static_cast<int>(idx->data[n]) - 1;  // 1-based -> 0-based
        if (j < 0 || j >= E->cols) j = 0;
        for (int d = 0; d < D; ++d) Y->data[d*N + n] = E->data[d*E->cols + j];
    }
    Node n;
    n.op = OP_EMBED;
    n.p0 = get_id(Ev);
    n.p1 = -1;
    n.val = Y;
    n.adj = nullptr;
    // Save the index vector so the pullback can scatter-add into dE.
    matlab_mat *idxSaved = mat_alloc(idx->rows, idx->cols);
    for (int64_t i = 0; i < idx->rows * idx->cols; ++i) idxSaved->data[i] = idx->data[i];
    n.auxData = { idxSaved };
    g_tape.push_back(n);
    int nid = static_cast<int>(g_tape.size()) - 1;
    set_data(robj, Y);
    set_id(robj, nid);
    return mat_alloc(0, 0);
}

// transpose(X) -- needed for attention (Q*K').  Forward = transpose of X.
matlab_mat *matlab_dlnet_transpose(void *r, void *xv) {
    using namespace dlnet;
    matlab_mat *X = get_data(xv);
    matlab_mat *Y = mat_alloc(X->cols, X->rows);
    for (int64_t i = 0; i < X->rows; ++i)
        for (int64_t j = 0; j < X->cols; ++j)
            Y->data[j*Y->cols + i] = X->data[i*X->cols + j];
    set_data(r, Y);
    set_id(r, record(OP_TRANSPOSE, get_id(xv), -1, Y));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_dlnet_relu   (void *r, void *x) { return un_op(r, x, dlnet::OP_RELU); }
matlab_mat *matlab_dlnet_sigmoid(void *r, void *x) { return un_op(r, x, dlnet::OP_SIGMOID); }
matlab_mat *matlab_dlnet_tanh   (void *r, void *x) { return un_op(r, x, dlnet::OP_TANH); }
matlab_mat *matlab_dlnet_softmax(void *r, void *x) { return un_op(r, x, dlnet::OP_SOFTMAX); }
matlab_mat *matlab_dlnet_sum    (void *r, void *x) { return un_op(r, x, dlnet::OP_SUM); }
matlab_mat *matlab_dlnet_mean   (void *r, void *x) { return un_op(r, x, dlnet::OP_MEAN); }
matlab_mat *matlab_dlnet_log    (void *r, void *x) { return un_op(r, x, dlnet::OP_LOG); }
matlab_mat *matlab_dlnet_exp    (void *r, void *x) { return un_op(r, x, dlnet::OP_EXP); }

// ---- losses (scalar) ------------------------------------------------------
// crossentropy(Y, T): Y already softmax probs (M classes × N batch);
// scalar = -sum(T .* log(Y)) / N.
matlab_mat *matlab_dlnet_crossentropy(void *r, void *yv, void *tv) {
    matlab_mat *Y = dlnet::get_data(yv), *T = dlnet::get_data(tv);
    int64_t N = Y->cols > 0 ? Y->cols : 1;
    double s = 0;
    for (int64_t i = 0; i < Y->rows*Y->cols; ++i) { double y = Y->data[i]; s += T->data[i]*std::log(y>1e-12?y:1e-12); }
    matlab_mat *L = mat_alloc(1,1); L->data[0] = -s/N;
    dlnet::set_data(r, L);
    dlnet::set_id(r, dlnet::record(dlnet::OP_CE, dlnet::get_id(yv), dlnet::get_id(tv), L));
    return mat_alloc(0, 0);
}
// mse(Y, T) = sum((Y-T).^2) / numel.
matlab_mat *matlab_dlnet_mse(void *r, void *yv, void *tv) {
    matlab_mat *Y = dlnet::get_data(yv), *T = dlnet::get_data(tv);
    int64_t nel = Y->rows*Y->cols; double s = 0;
    for (int64_t i = 0; i < nel; ++i) { double d = Y->data[i]-T->data[i]; s += d*d; }
    matlab_mat *L = mat_alloc(1,1); L->data[0] = nel? s/nel : 0;
    dlnet::set_data(r, L);
    dlnet::set_id(r, dlnet::record(dlnet::OP_MSE, dlnet::get_id(yv), dlnet::get_id(tv), L));
    return mat_alloc(0, 0);
}

// ---- recurrent: functional LSTM (T4) --------------------------------------
// MATLAB:  Y = lstm(X, H0, C0, W, R, b)
//   X  D×T      input sequence (D-dim features, T timesteps)
//   H0 H×1      initial hidden state
//   C0 H×1      initial cell state
//   W  4H×D     input weights, [i;f;g;o] stacked
//   R  4H×H     recurrent weights, [i;f;g;o] stacked
//   b  4H×1     biases, [i;f;g;o] stacked
//   Y  H×T      hidden state at every timestep (final-only is carved)
// One LSTM call is one tape node carrying every per-timestep gate + state;
// the BPTT pullback (in dlgradient below) walks them backward in time.
matlab_mat *matlab_dlnet_lstm(void *robj, void *xv, void *h0v, void *c0v,
                              void *Wv, void *Rv, void *bv) {
    using namespace dlnet;
    matlab_mat *X  = get_data(xv);
    matlab_mat *H0 = get_data(h0v);
    matlab_mat *C0 = get_data(c0v);
    matlab_mat *W  = get_data(Wv);
    matlab_mat *R  = get_data(Rv);
    matlab_mat *bm = get_data(bv);

    int D = static_cast<int>(X->rows);
    int T = static_cast<int>(X->cols);
    int H = static_cast<int>(H0->rows);

    matlab_mat *Y     = mat_alloc(H, T);
    matlab_mat *Hfull = mat_alloc(H, T + 1);
    matlab_mat *Cfull = mat_alloc(H, T + 1);
    matlab_mat *Imat  = mat_alloc(H, T);
    matlab_mat *Fmat  = mat_alloc(H, T);
    matlab_mat *Gmat  = mat_alloc(H, T);
    matlab_mat *Omat  = mat_alloc(H, T);

    for (int k = 0; k < H; ++k) {
        Hfull->data[k*(T+1) + 0] = H0->data[k];
        Cfull->data[k*(T+1) + 0] = C0->data[k];
    }

    std::vector<double> z(4*H);
    for (int t = 0; t < T; ++t) {
        // pre-activations z = W*x_t + R*h_prev + b   (4H × 1)
        for (int r = 0; r < 4*H; ++r) {
            double s = bm->data[r];
            for (int d = 0; d < D; ++d) s += W->data[r*D + d] * X->data[d*T + t];
            for (int h = 0; h < H; ++h) s += R->data[r*H + h] * Hfull->data[h*(T+1) + t];
            z[r] = s;
        }
        for (int k = 0; k < H; ++k) {
            double ig = 1.0 / (1.0 + std::exp(-z[0*H + k]));
            double fg = 1.0 / (1.0 + std::exp(-z[1*H + k]));
            double gg = std::tanh(z[2*H + k]);
            double og = 1.0 / (1.0 + std::exp(-z[3*H + k]));
            double c_prev = Cfull->data[k*(T+1) + t];
            double c_new  = fg * c_prev + ig * gg;
            double h_new  = og * std::tanh(c_new);
            Imat->data[k*T + t] = ig;
            Fmat->data[k*T + t] = fg;
            Gmat->data[k*T + t] = gg;
            Omat->data[k*T + t] = og;
            Cfull->data[k*(T+1) + t+1] = c_new;
            Hfull->data[k*(T+1) + t+1] = h_new;
            Y->data[k*T + t] = h_new;
        }
    }

    Node n;
    n.op = OP_LSTM;
    n.p0 = get_id(xv);
    n.p1 = get_id(h0v);
    n.val = Y;
    n.adj = nullptr;
    n.auxParents = { get_id(c0v), get_id(Wv), get_id(Rv), get_id(bv) };
    n.auxData    = { Hfull, Cfull, Imat, Fmat, Gmat, Omat };
    g_tape.push_back(n);
    int nid = static_cast<int>(g_tape.size()) - 1;

    set_data(robj, Y);
    set_id(robj, nid);
    return mat_alloc(0, 0);
}

// ---- recurrent: functional GRU --------------------------------------------
// MATLAB:  Y = gru(X, H0, W, R, b)
//   X  D×T   input sequence
//   H0 H×1   initial hidden state
//   W  3H×D  input weights, [r; z; h] stacked (reset / update / candidate)
//   R  3H×H  recurrent weights, [r; z; h] stacked
//   b  3H×1  biases, [r; z; h] stacked
//   Y  H×T   hidden state at every timestep
// One custom OP_GRU tape node; per-timestep r/z gates + candidate are saved
// for the BPTT pullback below.
matlab_mat *matlab_dlnet_gru(void *robj, void *xv, void *h0v,
                             void *Wv, void *Rv, void *bv) {
    using namespace dlnet;
    matlab_mat *X  = get_data(xv);
    matlab_mat *H0 = get_data(h0v);
    matlab_mat *W  = get_data(Wv);
    matlab_mat *R  = get_data(Rv);
    matlab_mat *bm = get_data(bv);

    int D = static_cast<int>(X->rows);
    int T = static_cast<int>(X->cols);
    int H = static_cast<int>(H0->rows);

    matlab_mat *Y      = mat_alloc(H, T);
    matlab_mat *Hfull  = mat_alloc(H, T + 1);
    matlab_mat *Rgate  = mat_alloc(H, T);
    matlab_mat *Zgate  = mat_alloc(H, T);
    matlab_mat *Htilde = mat_alloc(H, T);

    for (int k = 0; k < H; ++k) Hfull->data[k*(T+1) + 0] = H0->data[k];

    std::vector<double> hprev(H), rh(H);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < H; ++k) hprev[k] = Hfull->data[k*(T+1) + t];

        // r and z gates (sigmoid of W*x + R*h_prev + b)
        for (int k = 0; k < H; ++k) {
            double sr = bm->data[k];
            double sz = bm->data[H + k];
            for (int d = 0; d < D; ++d) {
                sr += W->data[k*D + d]       * X->data[d*T + t];
                sz += W->data[(H + k)*D + d] * X->data[d*T + t];
            }
            for (int h = 0; h < H; ++h) {
                sr += R->data[k*H + h]       * hprev[h];
                sz += R->data[(H + k)*H + h] * hprev[h];
            }
            Rgate->data[k*T + t] = 1.0 / (1.0 + std::exp(-sr));
            Zgate->data[k*T + t] = 1.0 / (1.0 + std::exp(-sz));
        }
        // r .* h_prev for the candidate's recurrent contribution
        for (int k = 0; k < H; ++k) rh[k] = Rgate->data[k*T + t] * hprev[k];
        // candidate h_tilde = tanh(W_h*x + R_h*(r.*h_prev) + b_h)
        for (int k = 0; k < H; ++k) {
            double s = bm->data[2*H + k];
            for (int d = 0; d < D; ++d) s += W->data[(2*H + k)*D + d] * X->data[d*T + t];
            for (int h = 0; h < H; ++h) s += R->data[(2*H + k)*H + h] * rh[h];
            Htilde->data[k*T + t] = std::tanh(s);
        }
        // h_new = (1 - z) .* h_prev + z .* h_tilde
        for (int k = 0; k < H; ++k) {
            double z_k = Zgate->data[k*T + t];
            double h_new = (1.0 - z_k) * hprev[k] + z_k * Htilde->data[k*T + t];
            Hfull->data[k*(T+1) + t+1] = h_new;
            Y->data[k*T + t] = h_new;
        }
    }

    Node n;
    n.op = OP_GRU;
    n.p0 = get_id(xv);
    n.p1 = get_id(h0v);
    n.val = Y;
    n.adj = nullptr;
    n.auxParents = { get_id(Wv), get_id(Rv), get_id(bv) };
    n.auxData    = { Hfull, Rgate, Zgate, Htilde };
    g_tape.push_back(n);
    int nid = static_cast<int>(g_tape.size()) - 1;
    set_data(robj, Y);
    set_id(robj, nid);
    return mat_alloc(0, 0);
}

// ---- recurrent: bidirectional LSTM ---------------------------------------
// MATLAB:  Y = bilstm(X, H0f, C0f, H0b, C0b, W, R, b)
//   X    D×T              input sequence
//   H0f  H×1              forward initial hidden
//   C0f  H×1              forward initial cell
//   H0b  H×1              backward initial hidden
//   C0b  H×1              backward initial cell
//   W    8H×D             [Wf; Wb] (each Wf/Wb is 4H×D, i/f/g/o stacked)
//   R    8H×H             [Rf; Rb]
//   b    8H×1             [bf; bb]
//   Y    2H×T             [Yf; Yb_aligned] (backward output is re-aligned
//                         to original time order)
// One custom OP_BILSTM tape node carrying both directions' per-timestep state.
matlab_mat *matlab_dlnet_bilstm(void *robj, void *xv,
                                void *h0fv, void *c0fv, void *h0bv, void *c0bv,
                                void *Wv, void *Rv, void *bv) {
    using namespace dlnet;
    matlab_mat *X   = get_data(xv);
    matlab_mat *H0f = get_data(h0fv);
    matlab_mat *C0f = get_data(c0fv);
    matlab_mat *H0b = get_data(h0bv);
    matlab_mat *C0b = get_data(c0bv);
    matlab_mat *W   = get_data(Wv);
    matlab_mat *R   = get_data(Rv);
    matlab_mat *bm  = get_data(bv);

    int D = static_cast<int>(X->rows);
    int T = static_cast<int>(X->cols);
    int H = static_cast<int>(H0f->rows);

    matlab_mat *Y   = mat_alloc(2*H, T);
    matlab_mat *Hf  = mat_alloc(H, T + 1);
    matlab_mat *Cf  = mat_alloc(H, T + 1);
    matlab_mat *Hb  = mat_alloc(H, T + 1);
    matlab_mat *Cb  = mat_alloc(H, T + 1);
    matlab_mat *If_ = mat_alloc(H, T), *Ff_ = mat_alloc(H, T), *Gf_ = mat_alloc(H, T), *Of_ = mat_alloc(H, T);
    matlab_mat *Ib_ = mat_alloc(H, T), *Fb_ = mat_alloc(H, T), *Gb_ = mat_alloc(H, T), *Ob_ = mat_alloc(H, T);

    for (int k = 0; k < H; ++k) {
        Hf->data[k*(T+1) + 0] = H0f->data[k];
        Cf->data[k*(T+1) + 0] = C0f->data[k];
        Hb->data[k*(T+1) + 0] = H0b->data[k];
        Cb->data[k*(T+1) + 0] = C0b->data[k];
    }

    auto run_lstm_step = [&](int t, int t_x, bool forward,
                             matlab_mat *Hs, matlab_mat *Cs,
                             matlab_mat *I, matlab_mat *F, matlab_mat *G, matlab_mat *O) {
        // Weights for this direction sit at row offset 0 (forward) or 4H (backward).
        int wofs = forward ? 0 : 4*H;
        std::vector<double> z(4*H);
        for (int r = 0; r < 4*H; ++r) {
            double s = bm->data[wofs + r];
            for (int d = 0; d < D; ++d) s += W->data[(wofs + r)*D + d] * X->data[d*T + t_x];
            for (int h = 0; h < H; ++h) s += R->data[(wofs + r)*H + h] * Hs->data[h*(T+1) + t];
            z[r] = s;
        }
        for (int k = 0; k < H; ++k) {
            double ig = 1.0 / (1.0 + std::exp(-z[0*H + k]));
            double fg = 1.0 / (1.0 + std::exp(-z[1*H + k]));
            double gg = std::tanh(z[2*H + k]);
            double og = 1.0 / (1.0 + std::exp(-z[3*H + k]));
            double c_prev = Cs->data[k*(T+1) + t];
            double c_new = fg * c_prev + ig * gg;
            double h_new = og * std::tanh(c_new);
            I->data[k*T + t] = ig; F->data[k*T + t] = fg; G->data[k*T + t] = gg; O->data[k*T + t] = og;
            Cs->data[k*(T+1) + t+1] = c_new;
            Hs->data[k*(T+1) + t+1] = h_new;
        }
    };

    // Forward direction reads X left-to-right; backward reads it right-to-left.
    for (int t = 0; t < T; ++t) run_lstm_step(t, t,         true,  Hf, Cf, If_, Ff_, Gf_, Of_);
    for (int t = 0; t < T; ++t) run_lstm_step(t, T - 1 - t, false, Hb, Cb, Ib_, Fb_, Gb_, Ob_);

    // Stack outputs:  Y[0..H-1,  t] = Hf[:, t+1]
    //                 Y[H..2H-1, t] = Hb[:, (T - 1 - t) + 1]  (re-align to original time)
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < H; ++k) {
            Y->data[k*T + t]       = Hf->data[k*(T+1) + t+1];
            Y->data[(H + k)*T + t] = Hb->data[k*(T+1) + (T - t)];
        }
    }

    Node n;
    n.op = OP_BILSTM;
    n.p0 = get_id(xv);
    n.p1 = get_id(h0fv);
    n.val = Y;
    n.adj = nullptr;
    n.auxParents = { get_id(c0fv), get_id(h0bv), get_id(c0bv),
                     get_id(Wv),   get_id(Rv),   get_id(bv) };
    n.auxData    = { Hf, Cf, Hb, Cb, If_, Ff_, Gf_, Of_, Ib_, Fb_, Gb_, Ob_ };
    g_tape.push_back(n);
    int nid = static_cast<int>(g_tape.size()) - 1;
    set_data(robj, Y);
    set_id(robj, nid);
    return mat_alloc(0, 0);
}

// ---- recurrent: LSTM with projected hidden state -------------------------
// MATLAB:  Y = lstmp(X, H0, C0, W, R, P, b)
//   X  D×T               input sequence
//   H0 Hp×1              initial *projected* hidden
//   C0 H×1               initial cell
//   W  4H×D              input weights, [i;f;g;o] stacked
//   R  4H×Hp             recurrent weights operating on the projected state
//   P  Hp×H              projection matrix (full hidden -> projected)
//   b  4H×1              biases
//   Y  Hp×T              projected hidden at every timestep
// This is `lstmProjectedLayer` collapsed to its mathematical core: the LSTM
// recurrence runs over the projected state for storage / recurrence, but the
// raw H-dim hidden is preserved internally because `P` participates in the
// gradient.
matlab_mat *matlab_dlnet_lstmp(void *robj, void *xv, void *h0v, void *c0v,
                               void *Wv, void *Rv, void *Pv, void *bv) {
    using namespace dlnet;
    matlab_mat *X  = get_data(xv);
    matlab_mat *H0 = get_data(h0v);
    matlab_mat *C0 = get_data(c0v);
    matlab_mat *W  = get_data(Wv);
    matlab_mat *R  = get_data(Rv);
    matlab_mat *P  = get_data(Pv);
    matlab_mat *bm = get_data(bv);

    int D  = static_cast<int>(X->rows);
    int T  = static_cast<int>(X->cols);
    int Hp = static_cast<int>(H0->rows);
    int H  = static_cast<int>(C0->rows);

    matlab_mat *Y      = mat_alloc(Hp, T);
    matlab_mat *Hproj  = mat_alloc(Hp, T + 1);   // projected hidden over time
    matlab_mat *Hpre   = mat_alloc(H,  T);       // raw H-dim hidden o*tanh(c) per t
    matlab_mat *Cfull  = mat_alloc(H,  T + 1);
    matlab_mat *Imat   = mat_alloc(H, T);
    matlab_mat *Fmat   = mat_alloc(H, T);
    matlab_mat *Gmat   = mat_alloc(H, T);
    matlab_mat *Omat   = mat_alloc(H, T);

    for (int k = 0; k < Hp; ++k) Hproj->data[k*(T+1) + 0] = H0->data[k];
    for (int k = 0; k < H;  ++k) Cfull->data[k*(T+1) + 0] = C0->data[k];

    std::vector<double> z(4*H);
    for (int t = 0; t < T; ++t) {
        for (int r = 0; r < 4*H; ++r) {
            double s = bm->data[r];
            for (int d = 0; d < D; ++d) s += W->data[r*D + d]  * X->data[d*T + t];
            for (int h = 0; h < Hp; ++h) s += R->data[r*Hp + h] * Hproj->data[h*(T+1) + t];
            z[r] = s;
        }
        for (int k = 0; k < H; ++k) {
            double ig = 1.0 / (1.0 + std::exp(-z[0*H + k]));
            double fg = 1.0 / (1.0 + std::exp(-z[1*H + k]));
            double gg = std::tanh(z[2*H + k]);
            double og = 1.0 / (1.0 + std::exp(-z[3*H + k]));
            double c_prev = Cfull->data[k*(T+1) + t];
            double c_new  = fg * c_prev + ig * gg;
            double h_pre  = og * std::tanh(c_new);
            Imat->data[k*T + t] = ig; Fmat->data[k*T + t] = fg;
            Gmat->data[k*T + t] = gg; Omat->data[k*T + t] = og;
            Cfull->data[k*(T+1) + t+1] = c_new;
            Hpre->data[k*T + t] = h_pre;
        }
        // h_proj = P * h_pre  (Hp × 1)
        for (int p = 0; p < Hp; ++p) {
            double s = 0;
            for (int h = 0; h < H; ++h) s += P->data[p*H + h] * Hpre->data[h*T + t];
            Hproj->data[p*(T+1) + t+1] = s;
            Y->data[p*T + t] = s;
        }
    }

    Node n;
    n.op = OP_LSTMP;
    n.p0 = get_id(xv);
    n.p1 = get_id(h0v);
    n.val = Y;
    n.adj = nullptr;
    n.auxParents = { get_id(c0v), get_id(Wv), get_id(Rv), get_id(Pv), get_id(bv) };
    n.auxData    = { Hproj, Hpre, Cfull, Imat, Fmat, Gmat, Omat };
    g_tape.push_back(n);
    int nid = static_cast<int>(g_tape.size()) - 1;
    set_data(robj, Y);
    set_id(robj, nid);
    return mat_alloc(0, 0);
}

// ---- dlgradient(loss, var): reverse sweep -> returns the var gradient -----
// Returns the gradient as a plain matrix (MATLAB returns a dlarray; we return
// the extracted value — a documented deviation that keeps the runtime from
// having to allocate a classdef instance).
matlab_mat *matlab_dlnet_grad(void *lossv, void *varv) {
    using namespace dlnet;
    int lossId = get_id(lossv), varId = get_id(varv);
    int N = static_cast<int>(g_tape.size());
    for (int i = 0; i < N; ++i) { g_tape[i].adj = nullptr; }
    if (lossId < 0 || lossId >= N) return mat_alloc(0, 0);
    // seed
    { matlab_mat *seed = clone(g_tape[lossId].val); for (int64_t i=0;i<seed->rows*seed->cols;++i) seed->data[i]=1.0; accum(lossId, seed); }
    for (int i = lossId; i >= 0; --i) {
        Node &n = g_tape[i];
        if (!n.adj) continue;
        matlab_mat *A = n.adj;
        matlab_mat *V = n.val;
        matlab_mat *P0 = (n.p0>=0)? g_tape[n.p0].val : nullptr;
        matlab_mat *P1 = (n.p1>=0)? g_tape[n.p1].val : nullptr;
        switch (n.op) {
        case OP_ADD:  accum(n.p0, A); accum(n.p1, A); break;
        case OP_SUB:  { accum(n.p0, A); matlab_mat *neg=clone(A); for(int64_t k=0;k<neg->rows*neg->cols;++k) neg->data[k]=-neg->data[k]; accum(n.p1, neg); } break;
        case OP_TIMES: {
            if (P1) { matlab_mat *g=mat_alloc(A->rows,A->cols); for(int64_t k=0;k<A->rows*A->cols;++k) g->data[k]=A->data[k]*P1->data[k]; accum(n.p0,g);}
            if (P0) { matlab_mat *g=mat_alloc(A->rows,A->cols); for(int64_t k=0;k<A->rows*A->cols;++k) g->data[k]=A->data[k]*P0->data[k]; accum(n.p1,g);}
        } break;
        case OP_MTIMES: {
            // C = P0 * P1 ; gP0 = A * P1' ; gP1 = P0' * A
            if (P0 && P1) {
                matlab_mat *gA = mat_alloc(P0->rows, P0->cols);
                for (int64_t r=0;r<P0->rows;++r) for(int64_t c=0;c<P0->cols;++c){ double s=0; for(int64_t k=0;k<P1->cols;++k) s+=A->data[r*A->cols+k]*P1->data[c*P1->cols+k]; gA->data[r*gA->cols+c]=s; }
                accum(n.p0, gA);
                matlab_mat *gB = mat_alloc(P1->rows, P1->cols);
                for (int64_t r=0;r<P1->rows;++r) for(int64_t c=0;c<P1->cols;++c){ double s=0; for(int64_t k=0;k<P0->rows;++k) s+=P0->data[k*P0->cols+r]*A->data[k*A->cols+c]; gB->data[r*gB->cols+c]=s; }
                accum(n.p1, gB);
            }
        } break;
        case OP_RELU:    { matlab_mat *g=mat_alloc(V->rows,V->cols); for(int64_t k=0;k<V->rows*V->cols;++k) g->data[k]= (V->data[k]>0)?A->data[k]:0; accum(n.p0,g);} break;
        case OP_SIGMOID: { matlab_mat *g=mat_alloc(V->rows,V->cols); for(int64_t k=0;k<V->rows*V->cols;++k){double y=V->data[k]; g->data[k]=A->data[k]*y*(1-y);} accum(n.p0,g);} break;
        case OP_TANH:    { matlab_mat *g=mat_alloc(V->rows,V->cols); for(int64_t k=0;k<V->rows*V->cols;++k){double y=V->data[k]; g->data[k]=A->data[k]*(1-y*y);} accum(n.p0,g);} break;
        case OP_EXP:     { matlab_mat *g=mat_alloc(V->rows,V->cols); for(int64_t k=0;k<V->rows*V->cols;++k) g->data[k]=A->data[k]*V->data[k]; accum(n.p0,g);} break;
        case OP_LOG:     { matlab_mat *g=mat_alloc(V->rows,V->cols); for(int64_t k=0;k<V->rows*V->cols;++k) g->data[k]=A->data[k]/P0->data[k]; accum(n.p0,g);} break;
        case OP_SOFTMAX: {
            // gx = y .* (adj - sum(adj.*y, over rows)) per column
            matlab_mat *g=mat_alloc(V->rows,V->cols);
            for (int64_t c=0;c<V->cols;++c){ double dot=0; for(int64_t r=0;r<V->rows;++r) dot+=A->data[r*A->cols+c]*V->data[r*V->cols+c];
                for(int64_t r=0;r<V->rows;++r){ double y=V->data[r*V->cols+c]; g->data[r*g->cols+c]=y*(A->data[r*A->cols+c]-dot);} }
            accum(n.p0,g);
        } break;
        case OP_SUM:  { matlab_mat *g=mat_alloc(P0->rows,P0->cols); for(int64_t k=0;k<P0->rows*P0->cols;++k) g->data[k]=A->data[0]; accum(n.p0,g);} break;
        case OP_MEAN: { int64_t nel=P0->rows*P0->cols; matlab_mat *g=mat_alloc(P0->rows,P0->cols); for(int64_t k=0;k<nel;++k) g->data[k]=A->data[0]/(nel?nel:1); accum(n.p0,g);} break;
        case OP_CE: {
            // L = -sum(T.*log(Y))/N ; gY = -(T./Y)/N * adj
            if (P0 && P1) { int64_t Nb = P0->cols>0?P0->cols:1; matlab_mat *g=mat_alloc(P0->rows,P0->cols);
                for(int64_t k=0;k<P0->rows*P0->cols;++k){ double y=P0->data[k]; g->data[k]= -A->data[0]*(P1->data[k]/(y>1e-12?y:1e-12))/Nb; } accum(n.p0,g); }
        } break;
        case OP_MSE: {
            if (P0 && P1) { int64_t nel=P0->rows*P0->cols; matlab_mat *g=mat_alloc(P0->rows,P0->cols);
                for(int64_t k=0;k<nel;++k) g->data[k]= A->data[0]*2.0*(P0->data[k]-P1->data[k])/(nel?nel:1); accum(n.p0,g); }
        } break;
        case OP_EMBED: {
            // dE(:, idx(n)) += dY(:, n) — scatter-add of A into the embedding rows.
            if (P0 && n.auxData.size() >= 1) {
                matlab_mat *idx = n.auxData[0];
                matlab_mat *g = mat_alloc(P0->rows, P0->cols);
                int64_t Nidx = idx->rows * idx->cols;
                int D = static_cast<int>(P0->rows);
                int Vv = static_cast<int>(P0->cols);
                for (int64_t nn = 0; nn < Nidx; ++nn) {
                    int j = static_cast<int>(idx->data[nn]) - 1;
                    if (j < 0 || j >= Vv) continue;
                    for (int d = 0; d < D; ++d) g->data[d*Vv + j] += A->data[d*A->cols + nn];
                }
                accum(n.p0, g);
            }
        } break;
        case OP_TRANSPOSE: {
            // Y = X' ; dY contribution to dX is dY' (transpose of the adjoint).
            if (P0) {
                matlab_mat *g = mat_alloc(P0->rows, P0->cols);
                for (int64_t i = 0; i < A->rows; ++i)
                    for (int64_t j = 0; j < A->cols; ++j)
                        g->data[j*g->cols + i] = A->data[i*A->cols + j];
                accum(n.p0, g);
            }
        } break;
        case OP_LSTM: {
            // BPTT for Y = lstm(X, H0, C0, W, R, b).
            //   p0 = X(D×T), p1 = H0(H×1)
            //   auxParents = [c0id, Wid, Rid, bid]
            //   auxData    = [Hfull(H×T+1), Cfull(H×T+1), I(H×T), F(H×T), G(H×T), O(H×T)]
            //   V = Y (H×T), A = dY (H×T) accumulated from upstream.
            if (n.auxParents.size() < 4 || n.auxData.size() < 6 || !P0) break;
            matlab_mat *Xmat = P0;
            matlab_mat *Wmat = g_tape[n.auxParents[1]].val;
            matlab_mat *Rmat = g_tape[n.auxParents[2]].val;
            matlab_mat *Hfull = n.auxData[0];
            matlab_mat *Cfull = n.auxData[1];
            matlab_mat *Imat = n.auxData[2];
            matlab_mat *Fmat = n.auxData[3];
            matlab_mat *Gmat = n.auxData[4];
            matlab_mat *Omat = n.auxData[5];
            int T = static_cast<int>(V->cols);
            int H = static_cast<int>(V->rows);
            int D = static_cast<int>(Xmat->rows);

            matlab_mat *dX = mat_alloc(D, T);
            matlab_mat *dW = mat_alloc(4*H, D);
            matlab_mat *dR = mat_alloc(4*H, H);
            matlab_mat *db = mat_alloc(4*H, 1);
            matlab_mat *dH0 = mat_alloc(H, 1);
            matlab_mat *dC0 = mat_alloc(H, 1);

            std::vector<double> dh_next(H, 0.0), dc_next(H, 0.0);
            std::vector<double> dpre(4*H, 0.0);

            for (int t = T - 1; t >= 0; --t) {
                std::vector<double> dh(H), dc(H);
                for (int k = 0; k < H; ++k) dh[k] = A->data[k*T + t] + dh_next[k];
                for (int k = 0; k < H; ++k) {
                    double ig = Imat->data[k*T + t];
                    double fg = Fmat->data[k*T + t];
                    double gg = Gmat->data[k*T + t];
                    double og = Omat->data[k*T + t];
                    double c_new  = Cfull->data[k*(T+1) + t+1];
                    double c_prev = Cfull->data[k*(T+1) + t];
                    double tc = std::tanh(c_new);

                    double do_k    = dh[k] * tc;
                    double dtanh_c = dh[k] * og;
                    dc[k] = dtanh_c * (1.0 - tc*tc) + dc_next[k];

                    double df = dc[k] * c_prev;
                    double dc_prev_k = dc[k] * fg;
                    double di = dc[k] * gg;
                    double dg = dc[k] * ig;

                    dpre[0*H + k] = di    * ig * (1.0 - ig);   // d/d pre_i
                    dpre[1*H + k] = df    * fg * (1.0 - fg);   // d/d pre_f
                    dpre[2*H + k] = dg    *      (1.0 - gg*gg);// d/d pre_g
                    dpre[3*H + k] = do_k  * og * (1.0 - og);   // d/d pre_o

                    dc_next[k] = dc_prev_k;
                }
                // dW += dpre * x_t' ; dR += dpre * h_prev'
                for (int r = 0; r < 4*H; ++r) {
                    double dp = dpre[r];
                    for (int d = 0; d < D; ++d) dW->data[r*D + d] += dp * Xmat->data[d*T + t];
                    for (int h = 0; h < H; ++h) dR->data[r*H + h] += dp * Hfull->data[h*(T+1) + t];
                    db->data[r] += dp;
                }
                // dX[:,t] = W' * dpre ; dh_next = R' * dpre
                for (int d = 0; d < D; ++d) {
                    double s = 0; for (int r = 0; r < 4*H; ++r) s += Wmat->data[r*D + d] * dpre[r];
                    dX->data[d*T + t] = s;
                }
                for (int h = 0; h < H; ++h) {
                    double s = 0; for (int r = 0; r < 4*H; ++r) s += Rmat->data[r*H + h] * dpre[r];
                    dh_next[h] = s;
                }
            }
            // after the t=0 step, dh_next / dc_next are the gradients w.r.t. H0 / C0
            for (int k = 0; k < H; ++k) { dH0->data[k] = dh_next[k]; dC0->data[k] = dc_next[k]; }

            accum(n.p0, dX);
            accum(n.p1, dH0);
            accum(n.auxParents[0], dC0);
            accum(n.auxParents[1], dW);
            accum(n.auxParents[2], dR);
            accum(n.auxParents[3], db);
        } break;
        case OP_GRU: {
            // p0=X, p1=H0, auxParents=[W, R, b]
            // auxData = [Hfull(H×T+1), Rgate(H×T), Zgate(H×T), Htilde(H×T)]
            if (n.auxParents.size() < 3 || n.auxData.size() < 4 || !P0) break;
            matlab_mat *Xmat   = P0;
            matlab_mat *Wmat   = g_tape[n.auxParents[0]].val;
            matlab_mat *Rmat   = g_tape[n.auxParents[1]].val;
            matlab_mat *Hfull  = n.auxData[0];
            matlab_mat *Rgate  = n.auxData[1];
            matlab_mat *Zgate  = n.auxData[2];
            matlab_mat *Htilde = n.auxData[3];
            int T = static_cast<int>(V->cols);
            int H = static_cast<int>(V->rows);
            int D = static_cast<int>(Xmat->rows);

            matlab_mat *dX  = mat_alloc(D, T);
            matlab_mat *dW  = mat_alloc(3*H, D);
            matlab_mat *dR  = mat_alloc(3*H, H);
            matlab_mat *db  = mat_alloc(3*H, 1);
            matlab_mat *dH0 = mat_alloc(H, 1);

            std::vector<double> dh_next(H, 0.0);
            std::vector<double> dz_pre(H), dr_pre(H), dh_pre(H);

            for (int t = T - 1; t >= 0; --t) {
                std::vector<double> dh(H), dh_prev(H);
                for (int k = 0; k < H; ++k) dh[k] = A->data[k*T + t] + dh_next[k];

                for (int k = 0; k < H; ++k) {
                    double z_k     = Zgate->data[k*T + t];
                    double ht_k    = Htilde->data[k*T + t];
                    double hprev_k = Hfull->data[k*(T+1) + t];
                    double dz_k = dh[k] * (ht_k - hprev_k);
                    double dh_tilde_k = dh[k] * z_k;
                    dh_prev[k] = dh[k] * (1.0 - z_k);
                    dh_pre[k] = dh_tilde_k * (1.0 - ht_k*ht_k);   // through tanh
                    dz_pre[k] = dz_k * z_k * (1.0 - z_k);          // through sigmoid
                }
                // d(r .* h_prev) = R_h^T * dh_pre
                std::vector<double> drhp(H, 0.0);
                for (int h = 0; h < H; ++h) {
                    double s = 0;
                    for (int k = 0; k < H; ++k) s += Rmat->data[(2*H + k)*H + h] * dh_pre[k];
                    drhp[h] = s;
                }
                std::vector<double> dr(H);
                for (int k = 0; k < H; ++k) {
                    double hprev_k = Hfull->data[k*(T+1) + t];
                    dr[k] = drhp[k] * hprev_k;
                    dh_prev[k] += drhp[k] * Rgate->data[k*T + t];
                    double r_k = Rgate->data[k*T + t];
                    dr_pre[k] = dr[k] * r_k * (1.0 - r_k);
                }

                // r .* h_prev for use in dR row 2*H..3H-1
                std::vector<double> rh(H);
                for (int k = 0; k < H; ++k) rh[k] = Rgate->data[k*T + t] * Hfull->data[k*(T+1) + t];

                // Accumulate dW/dR/db across the three stacked gate groups
                for (int k = 0; k < H; ++k) {
                    // r gate (row k)
                    double dpr = dr_pre[k];
                    for (int d = 0; d < D; ++d) dW->data[k*D + d] += dpr * Xmat->data[d*T + t];
                    for (int h = 0; h < H; ++h) dR->data[k*H + h] += dpr * Hfull->data[h*(T+1) + t];
                    db->data[k] += dpr;
                    // z gate (row H + k)
                    double dpz = dz_pre[k];
                    for (int d = 0; d < D; ++d) dW->data[(H + k)*D + d] += dpz * Xmat->data[d*T + t];
                    for (int h = 0; h < H; ++h) dR->data[(H + k)*H + h] += dpz * Hfull->data[h*(T+1) + t];
                    db->data[H + k] += dpz;
                    // h gate (row 2H + k) — R uses r.*h_prev
                    double dph = dh_pre[k];
                    for (int d = 0; d < D; ++d) dW->data[(2*H + k)*D + d] += dph * Xmat->data[d*T + t];
                    for (int h = 0; h < H; ++h) dR->data[(2*H + k)*H + h] += dph * rh[h];
                    db->data[2*H + k] += dph;
                }
                // dx_t = sum over the three gates of W_g^T * d_pre_g
                for (int d = 0; d < D; ++d) {
                    double s = 0;
                    for (int k = 0; k < H; ++k) {
                        s += Wmat->data[k*D + d]         * dr_pre[k];
                        s += Wmat->data[(H + k)*D + d]   * dz_pre[k];
                        s += Wmat->data[(2*H + k)*D + d] * dh_pre[k];
                    }
                    dX->data[d*T + t] = s;
                }
                // dh_prev contributions through R_r and R_z (h gate path is already in drhp / dh_prev)
                for (int h = 0; h < H; ++h) {
                    double sr = 0, sz = 0;
                    for (int k = 0; k < H; ++k) {
                        sr += Rmat->data[k*H + h]       * dr_pre[k];
                        sz += Rmat->data[(H + k)*H + h] * dz_pre[k];
                    }
                    dh_next[h] = dh_prev[h] + sr + sz;
                }
            }
            for (int k = 0; k < H; ++k) dH0->data[k] = dh_next[k];

            accum(n.p0, dX);
            accum(n.p1, dH0);
            accum(n.auxParents[0], dW);
            accum(n.auxParents[1], dR);
            accum(n.auxParents[2], db);
        } break;
        case OP_BILSTM: {
            // p0=X, p1=H0f, auxParents=[C0f, H0b, C0b, W, R, b]
            // auxData = [Hf, Cf, Hb, Cb, If, Ff, Gf, Of, Ib, Fb, Gb, Ob]
            if (n.auxParents.size() < 6 || n.auxData.size() < 12 || !P0) break;
            matlab_mat *Xmat = P0;
            matlab_mat *Wmat = g_tape[n.auxParents[3]].val;
            matlab_mat *Rmat = g_tape[n.auxParents[4]].val;
            matlab_mat *Hf = n.auxData[0], *Cf = n.auxData[1], *Hb = n.auxData[2], *Cb = n.auxData[3];
            matlab_mat *If_ = n.auxData[4], *Ff_ = n.auxData[5], *Gf_ = n.auxData[6], *Of_ = n.auxData[7];
            matlab_mat *Ib_ = n.auxData[8], *Fb_ = n.auxData[9], *Gb_ = n.auxData[10], *Ob_ = n.auxData[11];
            int T = static_cast<int>(V->cols);
            int twoH = static_cast<int>(V->rows);
            int H = twoH / 2;
            int D = static_cast<int>(Xmat->rows);

            matlab_mat *dX  = mat_alloc(D, T);
            matlab_mat *dW  = mat_alloc(8*H, D);
            matlab_mat *dR  = mat_alloc(8*H, H);
            matlab_mat *db  = mat_alloc(8*H, 1);
            matlab_mat *dH0f = mat_alloc(H, 1), *dC0f = mat_alloc(H, 1);
            matlab_mat *dH0b = mat_alloc(H, 1), *dC0b = mat_alloc(H, 1);

            // BPTT for one direction.  wofs = 0 (forward) or 4*H (backward).
            // adj_src(k, t) reads from Y at row offset 0 or H, at time t (forward)
            //               or T-1-t (backward, after re-alignment).
            auto bptt_dir = [&](bool forward, int wofs, int yrow_offset,
                                matlab_mat *Hs, matlab_mat *Cs,
                                matlab_mat *I, matlab_mat *F, matlab_mat *G, matlab_mat *O,
                                matlab_mat *dH0_out, matlab_mat *dC0_out) {
                std::vector<double> dh_next(H, 0.0), dc_next(H, 0.0);
                std::vector<double> dpre(4*H, 0.0);
                for (int step = T - 1; step >= 0; --step) {
                    int t_state = step;                          // index into Hs/Cs/I/F/G/O
                    int t_x     = forward ? step : (T - 1 - step); // index into X
                    int t_y     = t_x;                            // Y is in original time order
                    std::vector<double> dh(H), dc(H);
                    for (int k = 0; k < H; ++k)
                        dh[k] = A->data[(yrow_offset + k)*T + t_y] + dh_next[k];
                    for (int k = 0; k < H; ++k) {
                        double ig = I->data[k*T + t_state];
                        double fg = F->data[k*T + t_state];
                        double gg = G->data[k*T + t_state];
                        double og = O->data[k*T + t_state];
                        double c_new  = Cs->data[k*(T+1) + t_state+1];
                        double c_prev = Cs->data[k*(T+1) + t_state];
                        double tc = std::tanh(c_new);
                        double do_k    = dh[k] * tc;
                        double dtanh_c = dh[k] * og;
                        dc[k] = dtanh_c * (1.0 - tc*tc) + dc_next[k];
                        double df = dc[k] * c_prev;
                        double dc_prev_k = dc[k] * fg;
                        double di = dc[k] * gg;
                        double dg = dc[k] * ig;
                        dpre[0*H + k] = di   * ig * (1.0 - ig);
                        dpre[1*H + k] = df   * fg * (1.0 - fg);
                        dpre[2*H + k] = dg   *      (1.0 - gg*gg);
                        dpre[3*H + k] = do_k * og * (1.0 - og);
                        dc_next[k] = dc_prev_k;
                    }
                    for (int r = 0; r < 4*H; ++r) {
                        double dp = dpre[r];
                        for (int d = 0; d < D; ++d) dW->data[(wofs + r)*D + d] += dp * Xmat->data[d*T + t_x];
                        for (int h = 0; h < H; ++h) dR->data[(wofs + r)*H + h] += dp * Hs->data[h*(T+1) + t_state];
                        db->data[wofs + r] += dp;
                    }
                    for (int d = 0; d < D; ++d) {
                        double s = 0;
                        for (int r = 0; r < 4*H; ++r) s += Wmat->data[(wofs + r)*D + d] * dpre[r];
                        dX->data[d*T + t_x] += s;
                    }
                    for (int h = 0; h < H; ++h) {
                        double s = 0;
                        for (int r = 0; r < 4*H; ++r) s += Rmat->data[(wofs + r)*H + h] * dpre[r];
                        dh_next[h] = s;
                    }
                }
                for (int k = 0; k < H; ++k) { dH0_out->data[k] = dh_next[k]; dC0_out->data[k] = dc_next[k]; }
            };

            bptt_dir(true,  0,    0, Hf, Cf, If_, Ff_, Gf_, Of_, dH0f, dC0f);
            bptt_dir(false, 4*H,  H, Hb, Cb, Ib_, Fb_, Gb_, Ob_, dH0b, dC0b);

            accum(n.p0, dX);
            accum(n.p1, dH0f);
            accum(n.auxParents[0], dC0f);
            accum(n.auxParents[1], dH0b);
            accum(n.auxParents[2], dC0b);
            accum(n.auxParents[3], dW);
            accum(n.auxParents[4], dR);
            accum(n.auxParents[5], db);
        } break;
        case OP_LSTMP: {
            // p0=X, p1=H0(proj), auxParents=[C0, W, R, P, b]
            // auxData=[Hproj(Hp×T+1), Hpre(H×T), Cfull(H×T+1), Imat, Fmat, Gmat, Omat]
            if (n.auxParents.size() < 5 || n.auxData.size() < 7 || !P0) break;
            matlab_mat *Xmat  = P0;
            matlab_mat *Wmat  = g_tape[n.auxParents[1]].val;
            matlab_mat *Rmat  = g_tape[n.auxParents[2]].val;
            matlab_mat *Pmat  = g_tape[n.auxParents[3]].val;
            matlab_mat *Hproj = n.auxData[0];
            matlab_mat *Hpre  = n.auxData[1];
            matlab_mat *Cfull = n.auxData[2];
            matlab_mat *Imat  = n.auxData[3];
            matlab_mat *Fmat  = n.auxData[4];
            matlab_mat *Gmat  = n.auxData[5];
            matlab_mat *Omat  = n.auxData[6];
            int T  = static_cast<int>(V->cols);
            int Hp = static_cast<int>(V->rows);
            int H  = static_cast<int>(Pmat->cols);
            int D  = static_cast<int>(Xmat->rows);

            matlab_mat *dX  = mat_alloc(D, T);
            matlab_mat *dW  = mat_alloc(4*H, D);
            matlab_mat *dR  = mat_alloc(4*H, Hp);
            matlab_mat *dP  = mat_alloc(Hp, H);
            matlab_mat *db  = mat_alloc(4*H, 1);
            matlab_mat *dH0 = mat_alloc(Hp, 1);
            matlab_mat *dC0 = mat_alloc(H,  1);

            std::vector<double> dh_proj_next(Hp, 0.0), dc_next(H, 0.0);
            std::vector<double> dpre(4*H, 0.0);
            std::vector<double> dh_pre(H, 0.0);

            for (int t = T - 1; t >= 0; --t) {
                // Adjoint coming into projected hidden at this step.
                std::vector<double> dhp(Hp);
                for (int p = 0; p < Hp; ++p) dhp[p] = A->data[p*T + t] + dh_proj_next[p];
                // dP += dhp * Hpre[:,t]^T ; dh_pre = P^T * dhp
                for (int p = 0; p < Hp; ++p)
                    for (int h = 0; h < H; ++h) dP->data[p*H + h] += dhp[p] * Hpre->data[h*T + t];
                for (int h = 0; h < H; ++h) {
                    double s = 0;
                    for (int p = 0; p < Hp; ++p) s += Pmat->data[p*H + h] * dhp[p];
                    dh_pre[h] = s;
                }
                // Standard LSTM gate backward using dh_pre as the upstream dh.
                std::vector<double> dc(H);
                for (int k = 0; k < H; ++k) {
                    double ig = Imat->data[k*T + t];
                    double fg = Fmat->data[k*T + t];
                    double gg = Gmat->data[k*T + t];
                    double og = Omat->data[k*T + t];
                    double c_new  = Cfull->data[k*(T+1) + t+1];
                    double c_prev = Cfull->data[k*(T+1) + t];
                    double tc = std::tanh(c_new);
                    double do_k    = dh_pre[k] * tc;
                    double dtanh_c = dh_pre[k] * og;
                    dc[k] = dtanh_c * (1.0 - tc*tc) + dc_next[k];
                    double df = dc[k] * c_prev;
                    double dc_prev_k = dc[k] * fg;
                    double di = dc[k] * gg;
                    double dg = dc[k] * ig;
                    dpre[0*H + k] = di   * ig * (1.0 - ig);
                    dpre[1*H + k] = df   * fg * (1.0 - fg);
                    dpre[2*H + k] = dg   *      (1.0 - gg*gg);
                    dpre[3*H + k] = do_k * og * (1.0 - og);
                    dc_next[k] = dc_prev_k;
                }
                for (int r = 0; r < 4*H; ++r) {
                    double dp = dpre[r];
                    for (int d = 0; d < D; ++d) dW->data[r*D + d] += dp * Xmat->data[d*T + t];
                    for (int h = 0; h < Hp; ++h) dR->data[r*Hp + h] += dp * Hproj->data[h*(T+1) + t];
                    db->data[r] += dp;
                }
                for (int d = 0; d < D; ++d) {
                    double s = 0;
                    for (int r = 0; r < 4*H; ++r) s += Wmat->data[r*D + d] * dpre[r];
                    dX->data[d*T + t] = s;
                }
                // dh_proj_next from R: R^T * dpre  (Hp × 1)
                for (int h = 0; h < Hp; ++h) {
                    double s = 0;
                    for (int r = 0; r < 4*H; ++r) s += Rmat->data[r*Hp + h] * dpre[r];
                    dh_proj_next[h] = s;
                }
            }
            for (int p = 0; p < Hp; ++p) dH0->data[p] = dh_proj_next[p];
            for (int k = 0; k < H;  ++k) dC0->data[k] = dc_next[k];

            accum(n.p0, dX);
            accum(n.p1, dH0);
            accum(n.auxParents[0], dC0);
            accum(n.auxParents[1], dW);
            accum(n.auxParents[2], dR);
            accum(n.auxParents[3], dP);
            accum(n.auxParents[4], db);
        } break;
        default: break;
        }
    }
    matlab_mat *res = (varId>=0 && g_tape[varId].adj) ? clone(g_tape[varId].adj)
                    : (varId>=0 ? mat_alloc(g_tape[varId].val->rows, g_tape[varId].val->cols) : mat_alloc(0,0));
    return res;
}

// Reset the tape (dlfeval entry / start of a fresh forward pass).
matlab_mat *matlab_dlnet_reset(double dummy) { (void)dummy; dlnet::g_tape.clear(); return mat_alloc(0,0); }

// ---- HDL Tier H1 — symmetric INT8 quantization ----------------------------
// dlquantize(W)   -> the dequantized weight matrix (Q * scale), bit-accurate
//                    to the int8 storage that would land on the device.
// dlqscale(W)     -> the scalar scale factor for W (the LSB of the int8 grid).
// Symmetric per-tensor quantization:
//   scale = max(abs(W)) / 127
//   Q     = round(W / scale)              (clipped to [-127, 127])
//   W'    = Q * scale
// Outputs are plain numeric matrices — quantization is a post-training step
// outside the autodiff (no tape node is recorded).
matlab_mat *matlab_dlnet_quantize(matlab_mat *W) {
    if (!W || W->rows*W->cols == 0) return mat_alloc(0, 0);
    double m = 0;
    int64_t nel = W->rows * W->cols;
    for (int64_t i = 0; i < nel; ++i) {
        double a = std::fabs(W->data[i]);
        if (a > m) m = a;
    }
    double scale = (m > 0) ? (m / 127.0) : 1.0;
    matlab_mat *Wq = mat_alloc(W->rows, W->cols);
    for (int64_t i = 0; i < nel; ++i) {
        double q = std::round(W->data[i] / scale);
        if (q >  127.0) q =  127.0;
        if (q < -127.0) q = -127.0;
        Wq->data[i] = q * scale;
    }
    return Wq;
}
matlab_mat *matlab_dlnet_qscale(matlab_mat *W) {
    if (!W || W->rows*W->cols == 0) { matlab_mat *o = mat_alloc(1,1); o->data[0] = 1.0; return o; }
    double m = 0;
    int64_t nel = W->rows * W->cols;
    for (int64_t i = 0; i < nel; ++i) { double a = std::fabs(W->data[i]); if (a > m) m = a; }
    matlab_mat *o = mat_alloc(1, 1);
    o->data[0] = (m > 0) ? (m / 127.0) : 1.0;
    return o;
}

}  // extern "C"
