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
       OP_TANH, OP_SOFTMAX, OP_SUM, OP_MEAN, OP_LOG, OP_EXP, OP_CE, OP_MSE };

struct Node {
    int op;
    int p0, p1;            // parent node ids (-1 = none / not differentiable)
    matlab_mat *val;       // forward value (owned by the tape)
    matlab_mat *adj;       // adjoint accumulator (lazily allocated; nullptr = 0)
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
        default: break;
        }
    }
    matlab_mat *res = (varId>=0 && g_tape[varId].adj) ? clone(g_tape[varId].adj)
                    : (varId>=0 ? mat_alloc(g_tape[varId].val->rows, g_tape[varId].val->cols) : mat_alloc(0,0));
    return res;
}

// Reset the tape (dlfeval entry / start of a fresh forward pass).
matlab_mat *matlab_dlnet_reset(double dummy) { (void)dummy; dlnet::g_tape.clear(); return mat_alloc(0,0); }

}  // extern "C"
