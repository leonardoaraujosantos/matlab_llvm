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
#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <string>

extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
/* Tier-C: dlnet's conv2d_batch forward + backward delegate to the core
 * runtime's GEMM-based implementation + helpers. */
extern "C" void       *matlab_conv2d_batch(void *X, void *W);
extern "C" matlab_mat *matlab_im2col_2d(void *X, double kH, double kW);
extern "C" matlab_mat *matlab_im2col_2d_pad(void *X, double kH, double kW,
                                            double pad_h, double pad_w,
                                            double stride_h, double stride_w);
extern "C" matlab_mat *matlab_matmul_mm(matlab_mat *A, matlab_mat *B);
extern "C" matlab_mat *matlab_gpu_gemm  (matlab_mat *A, matlab_mat *B);

/* GPU-training dispatch toggle.  When set, dlnet's MTIMES forward +
 * backward routes through matlab_gpu_gemm — which Metal-accelerates
 * above the (M, N, K all ≥ 128) threshold and falls back to BLAS /
 * the naive triple-loop on small matrices.  Set via matlab_dlnet_gpu_set
 * from MATLAB (`dlnetGpu(1)` / `dlnetGpu(0)`).  Default off — keeps the
 * existing CPU lane bit-for-bit identical for the catalog of small
 * gating tests. */
static int g_dlnet_gpu = 0;
static matlab_mat *dlnet_gemm(matlab_mat *A, matlab_mat *B) {
    if (g_dlnet_gpu) return matlab_gpu_gemm(A, B);
    return matlab_matmul_mm(A, B);
}
/* Y = A^T * B (rows of A reduced).  We transpose A then call gemm — the
 * transpose is O(M·K) vs the gemm's O(M·N·K), so the asymptotic cost
 * stays gemm-dominated.  Kept inline so the GPU lane sees a single
 * gemm call per backward leg (matches MathWorks' cuBLAS strided dgemm
 * with op=T at the cuBLAS call site). */
static matlab_mat *dlnet_gemm_AtB(matlab_mat *A, matlab_mat *B) {
    matlab_mat *AT = mat_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j)
            AT->data[j * A->rows + i] = A->data[i * A->cols + j];
    return dlnet_gemm(AT, B);
}
static matlab_mat *dlnet_gemm_ABt(matlab_mat *A, matlab_mat *B) {
    matlab_mat *BT = mat_alloc(B->cols, B->rows);
    for (int64_t i = 0; i < B->rows; ++i)
        for (int64_t j = 0; j < B->cols; ++j)
            BT->data[j * B->rows + i] = B->data[i * B->cols + j];
    return dlnet_gemm(A, BT);
}

extern "C" matlab_mat *matlab_dlnet_gpu_set(double flag) {
    g_dlnet_gpu = (flag != 0.0) ? 1 : 0;
    return mat_alloc(0, 0);
}
extern "C" double matlab_dlnet_gpu_get(double dummy) {
    (void)dummy;
    return static_cast<double>(g_dlnet_gpu);
}

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

/* Shape-preserving clone: keeps the input descriptor's rank (mat / mat3
 * / matN) so the tape's adj alloc + zero-fill is correctly sized.  Without
 * this, a matN-valued forward (e.g. conv2d_batch output) would back-prop
 * through a 2-D adj which collapses the per-cell gradient to garbage.
 * Defensive: returns a 0x0 mat for unknown / NULL inputs. */
inline matlab_mat *clone(const matlab_mat *m) {
    if (!m) return mat_alloc(0, 0);
    if (mat_is_nd(m)) {
        const matlab_matN *Mn = reinterpret_cast<const matlab_matN *>(m);
        int64_t dims[16]; uint32_t lim = Mn->ndims < 16 ? Mn->ndims : 16;
        for (uint32_t k = 0; k < lim; ++k) dims[k] = Mn->dims[k];
        void *R = matN_alloc(static_cast<int>(lim), dims);
        if (!R) return mat_alloc(0, 0);
        double *dst = mat_is_nd(R) ? reinterpret_cast<matlab_matN *>(R)->data
                    : mat_is_3d(R) ? reinterpret_cast<matlab_mat3 *>(R)->data
                                   : reinterpret_cast<matlab_mat *>(R)->data;
        int64_t t = 1;
        for (uint32_t k = 0; k < Mn->ndims; ++k) t *= Mn->dims[k];
        if (dst && t > 0) memcpy(dst, Mn->data, static_cast<size_t>(t) * sizeof(double));
        return reinterpret_cast<matlab_mat *>(R);
    }
    if (mat_is_3d(m)) {
        const matlab_mat3 *M3 = reinterpret_cast<const matlab_mat3 *>(m);
        matlab_mat3 *o = mat3_alloc(M3->rows, M3->cols, M3->depth);
        int64_t t = M3->rows * M3->cols * M3->depth;
        if (t > 0) memcpy(o->data, M3->data, static_cast<size_t>(t) * sizeof(double));
        return reinterpret_cast<matlab_mat *>(o);
    }
    matlab_mat *o = mat_alloc(m->rows, m->cols);
    for (int64_t i = 0; i < m->rows * m->cols; ++i) o->data[i] = m->data[i];
    return o;
}

/* Total element count for any descriptor rank. */
inline int64_t nelem(const matlab_mat *m) {
    if (!m) return 0;
    if (mat_is_nd(m)) {
        const matlab_matN *Mn = reinterpret_cast<const matlab_matN *>(m);
        int64_t t = 1; for (uint32_t k = 0; k < Mn->ndims; ++k) t *= Mn->dims[k];
        return t;
    }
    if (mat_is_3d(m)) {
        const matlab_mat3 *M3 = reinterpret_cast<const matlab_mat3 *>(m);
        return M3->rows * M3->cols * M3->depth;
    }
    return m->rows * m->cols;
}

/* Allocate a zero-initialised descriptor with the same shape as `m`
 * (mat / mat3 / matN).  Cheaper than clone() — calloc skips the
 * memcpy — and the right primitive when building a fresh adjoint /
 * gradient buffer. */
inline matlab_mat *zero_clone(const matlab_mat *m) {
    if (!m) return mat_alloc(0, 0);
    if (mat_is_nd(m)) {
        const matlab_matN *Mn = reinterpret_cast<const matlab_matN *>(m);
        int64_t dims[16]; uint32_t lim = Mn->ndims < 16 ? Mn->ndims : 16;
        for (uint32_t k = 0; k < lim; ++k) dims[k] = Mn->dims[k];
        return reinterpret_cast<matlab_mat *>(matN_alloc(static_cast<int>(lim), dims));
    }
    if (mat_is_3d(m)) {
        const matlab_mat3 *M3 = reinterpret_cast<const matlab_mat3 *>(m);
        return reinterpret_cast<matlab_mat *>(mat3_alloc(M3->rows, M3->cols, M3->depth));
    }
    return mat_alloc(m->rows, m->cols);
}

/* Recover the (H, W, C, N) shape + per-axis stride tuple for any rank
 * (matN / mat3 / mat).  Used by every batched pool / conv forward to
 * walk the rank-agnostic input via a single index expression. */
struct Shape4 { int64_t H, W, C, N; int64_t s0, s1, s2, s3; const double *data; };
inline Shape4 shape4(const matlab_mat *m) {
    Shape4 s{};
    if (!m) return s;
    if (mat_is_nd(m)) {
        const matlab_matN *Mn = reinterpret_cast<const matlab_matN *>(m);
        s.H = Mn->dims[0]; s.W = Mn->dims[1];
        s.C = Mn->ndims >= 3 ? Mn->dims[2] : 1;
        s.N = Mn->ndims >= 4 ? Mn->dims[3] : 1;
        s.s0 = Mn->strides[0]; s.s1 = Mn->strides[1];
        s.s2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
        s.s3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
        s.data = Mn->data;
    } else if (mat_is_3d(m)) {
        const matlab_mat3 *M3 = reinterpret_cast<const matlab_mat3 *>(m);
        s.H = M3->rows; s.W = M3->cols; s.C = M3->depth; s.N = 1;
        s.s0 = M3->cols; s.s1 = 1; s.s2 = M3->rows * M3->cols; s.s3 = 0;
        s.data = M3->data;
    } else {
        s.H = m->rows; s.W = m->cols; s.C = 1; s.N = 1;
        s.s0 = m->cols; s.s1 = 1; s.s2 = 0; s.s3 = 0;
        s.data = m->data;
    }
    return s;
}

/* Flat-buffer view: returns the data pointer regardless of descriptor rank.
 * Useful for elementwise tape ops that don't care about shape. */
inline double *flatdata(matlab_mat *m) {
    if (!m) return nullptr;
    if (mat_is_nd(m)) return reinterpret_cast<matlab_matN *>(m)->data;
    if (mat_is_3d(m)) return reinterpret_cast<matlab_mat3 *>(m)->data;
    return m->data;
}
inline const double *flatdata(const matlab_mat *m) {
    if (!m) return nullptr;
    if (mat_is_nd(m)) return reinterpret_cast<const matlab_matN *>(m)->data;
    if (mat_is_3d(m)) return reinterpret_cast<const matlab_mat3 *>(m)->data;
    return m->data;
}

// ---- reverse-mode tape ----------------------------------------------------
// Opcodes.
enum { OP_LEAF, OP_ADD, OP_SUB, OP_MTIMES, OP_TIMES, OP_RELU, OP_SIGMOID,
       OP_TANH, OP_SOFTMAX, OP_SUM, OP_MEAN, OP_LOG, OP_EXP, OP_CE, OP_MSE,
       OP_LSTM, OP_TRANSPOSE, OP_EMBED, OP_GRU, OP_BILSTM, OP_LSTMP,
       /* Phase 1 — small additional ops. */
       OP_RDIV, OP_SQRT, OP_MEAN_DIM, OP_LEAKY_RELU, OP_GELU, OP_SWISH,
       OP_SOFTPLUS, OP_ELU,
       /* Tier C: rank-4 batched 2-D convolution (X*W with im2col+GEMM). */
       OP_CONV2D_BATCH,
       /* Shape / pool ops over matN tensors. */
       OP_RESHAPE, OP_MAXPOOL2D, OP_AVGPOOL2D,
       /* BatchNorm + conv-with-bias-pad-stride + axis-aware reductions. */
       OP_BATCHNORM, OP_CONV2D_FULL, OP_SOFTMAX_DIM, OP_MEAN_DIM_ND,
       /* LayerNorm (axis-aware) + GroupNorm + EMA-tracked BN training +
        * InstanceNorm + RMSNorm. */
       OP_LAYERNORM, OP_GROUPNORM, OP_BATCHNORM_TRAIN,
       OP_INSTANCENORM, OP_RMSNORM,
       /* Tape-tracked concatenation: vertcat (along rows, axis=1) and
        * horzcat (along cols, axis=2).  Two-input forms suffice for
        * the cat+single-Wo head-merge pattern in multi-head attention. */
       OP_VERTCAT, OP_HORZCAT };

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
// Handles broadcasting axes by reducing the contribution to the accumulator's
// shape: scalar / row / col / matrix.  When the accumulator is already
// allocated and shorter than `contrib`, we sum-reduce over the broadcast axes.
inline void accum(int id, const matlab_mat *contrib) {
    if (id < 0 || !contrib) return;
    Node &n = g_tape[id];
    if (!n.adj) {
        // Allocate the adj at the parent value's shape (sized by the
        // original leaf), not the contribution's — this ensures repeated
        // accumulations from broadcast results reduce to the correct
        // operand shape.  Shape-preserving clone-then-zero handles every
        // rank (matN / mat3 / mat).
        const matlab_mat *V = n.val ? n.val : contrib;
        n.adj = clone(V);
        double *ad = flatdata(n.adj);
        int64_t ne = nelem(n.adj);
        for (int64_t i = 0; i < ne; ++i) ad[i] = 0;
    }
    /* matN / mat3 elementwise fast path: same-shape buffers add through
     * the flat view; mismatch falls through to the 2-D broadcast logic
     * below (which is exclusively for 2-D row/col/scalar reductions —
     * matN/mat3 mismatch is a bug at this layer so silently drop). */
    if (mat_is_nd(n.adj) || mat_is_nd(contrib) ||
        mat_is_3d(n.adj) || mat_is_3d(contrib)) {
        int64_t aN_ = nelem(n.adj), cN_ = nelem(contrib);
        if (aN_ == cN_) {
            double *ad = flatdata(n.adj);
            const double *cd = flatdata(contrib);
            for (int64_t i = 0; i < aN_; ++i) ad[i] += cd[i];
        }
        return;
    }
    int64_t aM = n.adj->rows, aN = n.adj->cols;
    int64_t cM = contrib->rows, cN = contrib->cols;
    if (aM == cM && aN == cN) {
        // Strict element-wise.
        for (int64_t i = 0; i < aM*aN; ++i) n.adj->data[i] += contrib->data[i];
    } else if (aM == 1 && aN == 1) {
        // Scalar accumulator — sum the entire contribution.
        double s = 0; for (int64_t i = 0; i < cM*cN; ++i) s += contrib->data[i];
        n.adj->data[0] += s;
    } else if (aM == cM && aN == 1) {
        // Column accumulator — sum across cols.
        for (int64_t r = 0; r < cM; ++r) {
            double s = 0; for (int64_t c = 0; c < cN; ++c) s += contrib->data[r*cN + c];
            n.adj->data[r] += s;
        }
    } else if (aM == 1 && aN == cN) {
        // Row accumulator — sum across rows.
        for (int64_t c = 0; c < cN; ++c) {
            double s = 0; for (int64_t r = 0; r < cM; ++r) s += contrib->data[r*cN + c];
            n.adj->data[c] += s;
        }
    } else if (aM*aN == cM*cN) {
        // Flat length matches — element-wise (shape transposed but same nel).
        for (int64_t i = 0; i < aM*aN; ++i) n.adj->data[i] += contrib->data[i];
    }
    // Other mismatches: silently drop (defensive — backward of unsupported
    // broadcast shape).
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
// Numpy-style broadcasting on rank-2 lanes:
//   * scalar  (1x1)   — replicated everywhere
//   * row     (1xN)   — replicated across rows
//   * col     (Mx1)   — replicated across cols
//   * matrix  (MxN)   — strict elementwise (must match)
// The result keeps the larger shape (A's shape when broadcasting B).
static matlab_mat *bin_forward(int op, matlab_mat *A, matlab_mat *B) {
    int64_t m = A->rows, n = A->cols;
    if (op == dlnet::OP_MTIMES) {
        /* Dispatch through dlnet_gemm — calls matlab_gpu_gemm when the
         * GPU training toggle is on (Metal-accelerated above 128³),
         * matlab_matmul_mm (BLAS dgemm) otherwise.  Both yield identical
         * results to the naive triple-loop within double-precision
         * rounding. */
        return dlnet_gemm(A, B);
    }
    // Decide result shape — A defines it when A is "bigger".  Pure scalar
    // A is also handled (swap so the bigger side leads).
    int64_t aM = A->rows, aN = A->cols, bM = B->rows, bN = B->cols;
    int64_t oM = std::max(aM, bM), oN = std::max(aN, bN);
    matlab_mat *C = mat_alloc(oM, oN);
    for (int64_t r = 0; r < oM; ++r) {
        for (int64_t c = 0; c < oN; ++c) {
            int64_t ar = (aM == 1) ? 0 : r;
            int64_t ac = (aN == 1) ? 0 : c;
            int64_t br = (bM == 1) ? 0 : r;
            int64_t bc = (bN == 1) ? 0 : c;
            double a = A->data[ar*aN + ac];
            double b = B->data[br*bN + bc];
            double v = 0;
            if (op == dlnet::OP_ADD)   v = a + b;
            if (op == dlnet::OP_SUB)   v = a - b;
            if (op == dlnet::OP_TIMES) v = a * b;
            C->data[r*oN + c] = v;
        }
    }
    (void)m; (void)n;  // legacy guard names
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

/* ---- vertcat / horzcat (tape-tracked) ----------------------------------- *
 * Two-input shape concat.  Backward slices the adjoint back to the
 * appropriate row / col ranges.  Restricted to plain 2-D matlab_mat
 * operands (rank ≥ 3 cat is its own carve-down via cat(dim,…)). */
static matlab_mat *vertcat_forward(matlab_mat *A, matlab_mat *B) {
    if (!A || !B || A->cols != B->cols) return mat_alloc(0, 0);
    int64_t Ar = A->rows, Br = B->rows, C = A->cols;
    matlab_mat *Y = mat_alloc(Ar + Br, C);
    for (int64_t i = 0; i < Ar; ++i)
        for (int64_t j = 0; j < C; ++j) Y->data[i * C + j] = A->data[i * C + j];
    for (int64_t i = 0; i < Br; ++i)
        for (int64_t j = 0; j < C; ++j) Y->data[(Ar + i) * C + j] = B->data[i * C + j];
    return Y;
}
static matlab_mat *horzcat_forward(matlab_mat *A, matlab_mat *B) {
    if (!A || !B || A->rows != B->rows) return mat_alloc(0, 0);
    int64_t R = A->rows, Ac = A->cols, Bc = B->cols;
    matlab_mat *Y = mat_alloc(R, Ac + Bc);
    int64_t Yc = Ac + Bc;
    for (int64_t i = 0; i < R; ++i) {
        for (int64_t j = 0; j < Ac; ++j) Y->data[i * Yc + j] = A->data[i * Ac + j];
        for (int64_t j = 0; j < Bc; ++j) Y->data[i * Yc + (Ac + j)] = B->data[i * Bc + j];
    }
    return Y;
}
matlab_mat *matlab_dlnet_vertcat(void *r, void *a, void *b) {
    using namespace dlnet;
    matlab_mat *A = get_data(a), *B = get_data(b);
    matlab_mat *Y = vertcat_forward(A, B);
    set_data(r, Y);
    set_id(r, record(OP_VERTCAT, get_id(a), get_id(b), Y));
    return mat_alloc(0, 0);
}
matlab_mat *matlab_dlnet_horzcat(void *r, void *a, void *b) {
    using namespace dlnet;
    matlab_mat *A = get_data(a), *B = get_data(b);
    matlab_mat *Y = horzcat_forward(A, B);
    set_data(r, Y);
    set_id(r, record(OP_HORZCAT, get_id(a), get_id(b), Y));
    return mat_alloc(0, 0);
}

// ---- unary / activation ops -----------------------------------------------
static matlab_mat *un_forward(int op, matlab_mat *X) {
    /* Rank-agnostic: walks via nelem/flatdata so matN-valued inputs
     * (e.g. relu over a 4-D conv output) don't read the magic word as
     * rows/cols.  Softmax stays 2-D since it's a per-column reduction;
     * matN softmax-along-axis is carved as OP_SOFTMAX_DIM. */
    using namespace dlnet;
    int64_t ne = nelem(X);
    const double *Xd = flatdata(X);
    if (op == dlnet::OP_SUM)  {
        matlab_mat *o = mat_alloc(1, 1);
        double s = 0; for (int64_t i = 0; i < ne; ++i) s += Xd[i];
        o->data[0] = s; return o;
    }
    if (op == dlnet::OP_MEAN) {
        matlab_mat *o = mat_alloc(1, 1);
        double s = 0; for (int64_t i = 0; i < ne; ++i) s += Xd[i];
        o->data[0] = ne ? s / ne : 0; return o;
    }
    if (op == dlnet::OP_SOFTMAX) {
        /* Strictly 2-D: matN softmax requires an axis parameter (carved). */
        int64_t m = X->rows, n = X->cols;
        if (mat_is_nd(X) || mat_is_3d(X)) {
            /* Defensive — softmax on rank>2 returns empty, caller should
             * have reshape'd to 2-D first. */
            return mat_alloc(0, 0);
        }
        matlab_mat *Y = mat_alloc(m, n);
        for (int64_t c = 0; c < n; ++c) {
            double mx = -1e300; for (int64_t r=0;r<m;++r) mx = std::max(mx, X->data[r*n+c]);
            double sm = 0; for (int64_t r=0;r<m;++r){ double e=std::exp(X->data[r*n+c]-mx); Y->data[r*n+c]=e; sm+=e; }
            for (int64_t r=0;r<m;++r) Y->data[r*n+c] /= (sm>0?sm:1);
        }
        return Y;
    }
    /* Elementwise unary: rank-preserving via zero_clone+overwrite. */
    matlab_mat *Y = zero_clone(X);
    double *Yd = flatdata(Y);
    for (int64_t i = 0; i < ne; ++i) {
        double x = Xd[i], y = x;
        if (op == dlnet::OP_RELU)    y = x > 0 ? x : 0;
        if (op == dlnet::OP_SIGMOID) y = 1.0/(1.0+std::exp(-x));
        if (op == dlnet::OP_TANH)    y = std::tanh(x);
        if (op == dlnet::OP_LOG)     y = std::log(x);
        if (op == dlnet::OP_EXP)     y = std::exp(x);
        Yd[i] = y;
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
    using namespace dlnet;
    matlab_mat *Y = get_data(yv), *T = get_data(tv);
    /* Rank-agnostic: matN / mat3 / mat all expose a contiguous flat
     * buffer via flatdata, total elements via nelem.  Without this the
     * direct Y->rows / Y->cols reads garbage from the matN's magic word. */
    int64_t nel = nelem(Y);
    if (nelem(T) != nel) { matlab_mat *L = mat_alloc(1,1); L->data[0] = 0;
                           set_data(r, L);
                           set_id(r, record(OP_MSE, get_id(yv), get_id(tv), L));
                           return mat_alloc(0, 0); }
    const double *Yd = flatdata(Y), *Td = flatdata(T);
    double s = 0;
    for (int64_t i = 0; i < nel; ++i) { double d = Yd[i] - Td[i]; s += d*d; }
    matlab_mat *L = mat_alloc(1, 1); L->data[0] = nel ? s / nel : 0;
    set_data(r, L);
    set_id(r, record(OP_MSE, get_id(yv), get_id(tv), L));
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

// ---- Phase 1 small ops: rdivide / sqrt / mean(dim) / extra activations ----

// Element-wise A ./ B with numpy-style broadcasting (scalar / row / col / mat).
matlab_mat *matlab_dlnet_rdivide(void *r, void *a, void *b) {
    using namespace dlnet;
    matlab_mat *A = get_data(a), *B = get_data(b);
    int64_t aM = A->rows, aN = A->cols, bM = B->rows, bN = B->cols;
    int64_t oM = std::max(aM, bM), oN = std::max(aN, bN);
    matlab_mat *C = mat_alloc(oM, oN);
    for (int64_t r2 = 0; r2 < oM; ++r2) {
        for (int64_t c = 0; c < oN; ++c) {
            int64_t ar = (aM == 1) ? 0 : r2;
            int64_t ac = (aN == 1) ? 0 : c;
            int64_t br = (bM == 1) ? 0 : r2;
            int64_t bc = (bN == 1) ? 0 : c;
            double av = A->data[ar*aN + ac];
            double bv = B->data[br*bN + bc];
            C->data[r2*oN + c] = (std::fabs(bv) > 1e-30) ? av / bv : 0.0;
        }
    }
    set_data(r, C);
    set_id(r, record(OP_RDIV, get_id(a), get_id(b), C));
    return mat_alloc(0, 0);
}

// sqrt(X)
matlab_mat *matlab_dlnet_sqrt(void *r, void *x) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows*X->cols; ++i) {
        double v = X->data[i];
        Y->data[i] = (v > 0) ? std::sqrt(v) : 0.0;
    }
    set_data(r, Y);
    set_id(r, record(OP_SQRT, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

// mean(X, dim) — dim = 1 → row-vector (1 × cols); dim = 2 → col-vector (rows × 1).
// Records the dim in the val matrix's leading byte position via aux data — but
// since OP_MEAN_DIM's pullback can read dim from V's shape vs P0's shape, we
// don't need an explicit attribute.
matlab_mat *matlab_dlnet_mean_dim(void *r, void *x, double dimd) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    int dim = static_cast<int>(dimd);
    /* matN path: walk every (axis-collapsed) cell.  Output has the same
     * rank with `dim` replaced by 1; matN_alloc may collapse trailing
     * singletons back to mat3 / mat. */
    if (mat_is_nd(X) || mat_is_3d(X)) {
        int nd; int64_t dims[16], strides[16];
        const double *Xd;
        if (mat_is_nd(X)) {
            matlab_matN *Mn = reinterpret_cast<matlab_matN *>(X);
            nd = static_cast<int>(Mn->ndims);
            if (nd > 16) nd = 16;
            for (int k = 0; k < nd; ++k) {
                dims[k] = Mn->dims[k]; strides[k] = Mn->strides[k];
            }
            Xd = Mn->data;
        } else {
            matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(X);
            nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
            strides[0] = M3->cols; strides[1] = 1; strides[2] = M3->rows * M3->cols;
            Xd = M3->data;
        }
        if (dim < 1 || dim > nd) {
            matlab_mat *Y2 = mat_alloc(0, 0);
            set_data(r, Y2);
            set_id(r, record(OP_MEAN_DIM_ND, get_id(x), -1, Y2));
            return mat_alloc(0, 0);
        }
        int axis = dim - 1;
        int64_t outDims[16];
        for (int k = 0; k < nd; ++k) outDims[k] = (k == axis) ? 1 : dims[k];
        void *Rv = matN_alloc(nd, outDims);
        double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
                  : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                                  : reinterpret_cast<matlab_mat *>(Rv)->data;
        int64_t outerN = 1;
        for (int k = 0; k < nd; ++k) outerN *= outDims[k];
        int64_t reduceLen = dims[axis];
        int64_t idx[16] = {0};
        for (int64_t oo = 0; oo < outerN; ++oo) {
            int64_t srcBase = 0;
            for (int k = 0; k < nd; ++k) srcBase += idx[k] * strides[k];
            double s = 0;
            for (int64_t a = 0; a < reduceLen; ++a)
                s += Xd[srcBase + a * strides[axis]];
            Yd[oo] = reduceLen > 0 ? s / static_cast<double>(reduceLen) : 0;
            /* Advance idx over outDims (the reduce axis stays 0). */
            for (int k = nd - 1; k >= 0; --k) {
                if (++idx[k] < outDims[k]) break;
                idx[k] = 0;
            }
        }
        matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
        set_data(r, Y);
        int id = record(OP_MEAN_DIM_ND, get_id(x), -1, Y);
        matlab_mat *dm = mat_alloc(1, 1); dm->data[0] = static_cast<double>(dim);
        g_tape[id].auxData.push_back(dm);
        set_id(r, id);
        return mat_alloc(0, 0);
    }
    /* Plain 2-D path (legacy). */
    int64_t M = X->rows, N = X->cols;
    matlab_mat *Y;
    if (dim == 1) {
        Y = mat_alloc(1, N);
        for (int64_t c = 0; c < N; ++c) {
            double s = 0; for (int64_t rr = 0; rr < M; ++rr) s += X->data[rr*N + c];
            Y->data[c] = (M > 0) ? s / M : 0;
        }
    } else {
        // default dim = 2 (mean across columns -> column vector)
        Y = mat_alloc(M, 1);
        for (int64_t rr = 0; rr < M; ++rr) {
            double s = 0; for (int64_t c = 0; c < N; ++c) s += X->data[rr*N + c];
            Y->data[rr] = (N > 0) ? s / N : 0;
        }
    }
    set_data(r, Y);
    set_id(r, record(OP_MEAN_DIM, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

// leakyrelu(X) — alpha = 0.01 (standard).
matlab_mat *matlab_dlnet_leakyrelu(void *r, void *x) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows*X->cols; ++i) {
        double v = X->data[i];
        Y->data[i] = (v > 0) ? v : 0.01 * v;
    }
    set_data(r, Y);
    set_id(r, record(OP_LEAKY_RELU, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

// gelu(X) — Hendrycks' fast approximation:  x * sigmoid(1.702*x).
//  Pulls: d/dx [x·σ(αx)] = σ(αx) + α·x·σ(αx)·(1-σ(αx))  with α = 1.702.
matlab_mat *matlab_dlnet_gelu(void *r, void *x) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows*X->cols; ++i) {
        double v = X->data[i];
        double s = 1.0 / (1.0 + std::exp(-1.702 * v));
        Y->data[i] = v * s;
    }
    set_data(r, Y);
    set_id(r, record(OP_GELU, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

// swish(X) = x * sigmoid(x).  Pullback: σ(x) + x·σ(x)·(1-σ(x)) = σ(x) + y·(1-σ(x)).
matlab_mat *matlab_dlnet_swish(void *r, void *x) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows*X->cols; ++i) {
        double v = X->data[i];
        double s = 1.0 / (1.0 + std::exp(-v));
        Y->data[i] = v * s;
    }
    set_data(r, Y);
    set_id(r, record(OP_SWISH, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

// softplus(X) = log(1 + exp(X)).  Numerically stable for large |x|.
matlab_mat *matlab_dlnet_softplus(void *r, void *x) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows*X->cols; ++i) {
        double v = X->data[i];
        Y->data[i] = (v > 20.0) ? v : (v < -20.0 ? std::exp(v) : std::log1p(std::exp(v)));
    }
    set_data(r, Y);
    set_id(r, record(OP_SOFTPLUS, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

// elu(X) — alpha = 1.0.  y = x for x>0, y = α*(exp(x)-1) for x≤0.
matlab_mat *matlab_dlnet_elu(void *r, void *x) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows*X->cols; ++i) {
        double v = X->data[i];
        Y->data[i] = (v > 0) ? v : (std::exp(v) - 1.0);
    }
    set_data(r, Y);
    set_id(r, record(OP_ELU, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

/* conv2d_batch(X, W) — batched 2-D conv with autodiff support.
 *
 * Forward: defers to the GEMM-based matlab_conv2d_batch in the core
 * runtime (matN X * matN W -> matN Y).
 *
 * Backward (computed on the reverse sweep below):
 *   dX = transposed conv of dY against W   (col2im of W^T * dY_2d)
 *   dW = correlation of X against dY        (dY_2d * X_col^T)
 * Implemented via the same im2col-matmul pattern as the forward, using
 * matlab_matmul_mm (BLAS-accelerated when available).
 *
 * Records the OP_CONV2D_BATCH node with X as p0 and W as p1; the saved
 * forward values are the actual X / W tape-node values (no need to dup).
 */
matlab_mat *matlab_dlnet_conv2d_batch(void *r, void *x, void *w) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *W = get_data(w);
    matlab_mat *Y =
        reinterpret_cast<matlab_mat *>(matlab_conv2d_batch(X, W));
    set_data(r, Y);
    set_id(r, record(OP_CONV2D_BATCH, get_id(x), get_id(w), Y));
    return mat_alloc(0, 0);
}

/* reshape(X, m, n) — rank-agnostic 2-D output.  Numel must match.
 * Forward: copy the flat buffer into a fresh 2-D mat.
 * Backward: copy the adjoint back into a buffer with X's original
 * shape (matN / mat3 / mat preserved by clone). */
matlab_mat *matlab_dlnet_reshape2(void *r, void *x, double m, double n) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    int64_t M = static_cast<int64_t>(m), N = static_cast<int64_t>(n);
    int64_t total = nelem(X);
    matlab_mat *Y;
    if (M <= 0 || N <= 0 || M * N != total) {
        Y = mat_alloc(0, 0);
    } else {
        Y = mat_alloc(M, N);
        const double *Xd = flatdata(X);
        if (total > 0)
            memcpy(Y->data, Xd, static_cast<size_t>(total) * sizeof(double));
    }
    set_data(r, Y);
    set_id(r, record(OP_RESHAPE, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

/* reshape(X, d1, d2, d3, d4) — 4-D / matN output. */
matlab_mat *matlab_dlnet_reshape4(void *r, void *x,
                                  double d1, double d2, double d3, double d4) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    int64_t dims[4] = {static_cast<int64_t>(d1), static_cast<int64_t>(d2),
                       static_cast<int64_t>(d3), static_cast<int64_t>(d4)};
    int64_t total = nelem(X);
    int64_t outTotal = dims[0] * dims[1] * dims[2] * dims[3];
    matlab_mat *Y;
    if (outTotal != total || outTotal <= 0) {
        Y = mat_alloc(0, 0);
    } else {
        void *Rv = matN_alloc(4, dims);
        double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
                  : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                                  : reinterpret_cast<matlab_mat *>(Rv)->data;
        const double *Xd = flatdata(X);
        memcpy(Yd, Xd, static_cast<size_t>(total) * sizeof(double));
        Y = reinterpret_cast<matlab_mat *>(Rv);
    }
    set_data(r, Y);
    set_id(r, record(OP_RESHAPE, get_id(x), -1, Y));
    return mat_alloc(0, 0);
}

/* maxpool2d(X, kH, kW) — non-overlapping pool, stride = kernel.
 *
 *   X : H x W x C x N (or trailing-singleton-dropped variants)
 *   Y : (H/kH) x (W/kW) x C x N
 *
 * Backward routes the upstream gradient back to the arg-max cell
 * within each window.  argmax positions saved in auxData[1] as a
 * flat int matrix; auxData[0] carries (kH, kW). */
matlab_mat *matlab_dlnet_maxpool2d(void *r, void *x, double kHd, double kWd) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    int64_t kH = static_cast<int64_t>(kHd), kW = static_cast<int64_t>(kWd);
    if (kH <= 0 || kW <= 0) { matlab_mat *Y = mat_alloc(0, 0);
                              set_data(r, Y);
                              set_id(r, record(OP_MAXPOOL2D, get_id(x), -1, Y));
                              return mat_alloc(0, 0); }
    Shape4 S = shape4(X);
    int64_t Hout = S.H / kH, Wout = S.W / kW;
    int64_t outDims[4] = {Hout, Wout, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    /* argmax: flat row vector indexed by ((n*C + c)*Hout + hOut)*Wout + wOut.
     * Value stored as the *encoded* (hi * S.W + wi) so backward can
     * recover the (hi, wi) input position via div / mod. */
    int64_t outN = Hout * Wout * S.C * S.N;
    matlab_mat *argmax = mat_alloc(1, outN > 0 ? outN : 1);
    for (int64_t n = 0; n < S.N; ++n)
        for (int64_t c = 0; c < S.C; ++c)
            for (int64_t hOut = 0; hOut < Hout; ++hOut)
                for (int64_t wOut = 0; wOut < Wout; ++wOut) {
                    double m_val = -1e300;
                    int64_t m_hi = hOut * kH, m_wi = wOut * kW;
                    for (int64_t kh = 0; kh < kH; ++kh)
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t hi = hOut * kH + kh, wi = wOut * kW + kw;
                            double v = S.data[hi*S.s0 + wi*S.s1 + c*S.s2 + n*S.s3];
                            if (v > m_val) { m_val = v; m_hi = hi; m_wi = wi; }
                        }
                    Yd[hOut*Ys[0] + wOut*Ys[1] + c*Ys[2] + n*Ys[3]] = m_val;
                    int64_t fl = ((n * S.C + c) * Hout + hOut) * Wout + wOut;
                    argmax->data[fl] = static_cast<double>(m_hi * S.W + m_wi);
                }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_MAXPOOL2D, get_id(x), -1, Y);
    /* auxData: [0]=kHkW (1x2), [1]=argmax (1 x outN). */
    matlab_mat *kK = mat_alloc(1, 2);
    kK->data[0] = static_cast<double>(kH); kK->data[1] = static_cast<double>(kW);
    g_tape[id].auxData.push_back(kK);
    g_tape[id].auxData.push_back(argmax);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* avgpool2d(X, kH, kW) — non-overlapping pool, average instead of max.
 * Backward spreads dY uniformly over each window cell, scaled by 1/(kH*kW). */
matlab_mat *matlab_dlnet_avgpool2d(void *r, void *x, double kHd, double kWd) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    int64_t kH = static_cast<int64_t>(kHd), kW = static_cast<int64_t>(kWd);
    if (kH <= 0 || kW <= 0) { matlab_mat *Y = mat_alloc(0, 0);
                              set_data(r, Y);
                              set_id(r, record(OP_AVGPOOL2D, get_id(x), -1, Y));
                              return mat_alloc(0, 0); }
    Shape4 S = shape4(X);
    int64_t Hout = S.H / kH, Wout = S.W / kW;
    int64_t outDims[4] = {Hout, Wout, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    double inv = 1.0 / static_cast<double>(kH * kW);
    for (int64_t n = 0; n < S.N; ++n)
        for (int64_t c = 0; c < S.C; ++c)
            for (int64_t hOut = 0; hOut < Hout; ++hOut)
                for (int64_t wOut = 0; wOut < Wout; ++wOut) {
                    double acc = 0;
                    for (int64_t kh = 0; kh < kH; ++kh)
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t hi = hOut * kH + kh, wi = wOut * kW + kw;
                            acc += S.data[hi*S.s0 + wi*S.s1 + c*S.s2 + n*S.s3];
                        }
                    Yd[hOut*Ys[0] + wOut*Ys[1] + c*Ys[2] + n*Ys[3]] = acc * inv;
                }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_AVGPOOL2D, get_id(x), -1, Y);
    /* auxData: [0]=kHkW (1x2). */
    matlab_mat *kK = mat_alloc(1, 2);
    kK->data[0] = static_cast<double>(kH); kK->data[1] = static_cast<double>(kW);
    g_tape[id].auxData.push_back(kK);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* conv2d_batch_full(X, W, b, pad_h, pad_w, stride_h, stride_w) — autodiff-
 * tracked conv with optional bias, zero-padding, and stride.  Forward
 * delegates to the matlab_conv2d_batch_full core; backward walks an
 * explicit set of (h_out, w_out) windows so padding + stride compose
 * cleanly.  Records OP_CONV2D_FULL with X as p0, W as p1, b as auxParents[0],
 * (pad_h, pad_w, stride_h, stride_w) as auxData[0]. */
extern "C" void *matlab_conv2d_batch_full(void *X, void *W, void *b,
                                          double pad_h, double pad_w,
                                          double stride_h, double stride_w);
matlab_mat *matlab_dlnet_conv2d_full(void *r, void *x, void *w, void *b,
                                     double pad_h, double pad_w,
                                     double stride_h, double stride_w) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *Ww = get_data(w);
    matlab_mat *B  = get_data(b);
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(
        matlab_conv2d_batch_full(X, Ww, B, pad_h, pad_w, stride_h, stride_w));
    set_data(r, Y);
    int id = record(OP_CONV2D_FULL, get_id(x), get_id(w), Y);
    g_tape[id].auxParents.push_back(get_id(b));
    matlab_mat *cfg = mat_alloc(1, 4);
    cfg->data[0] = pad_h; cfg->data[1] = pad_w;
    cfg->data[2] = stride_h; cfg->data[3] = stride_w;
    g_tape[id].auxData.push_back(cfg);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* batchnorm(X, gamma, beta) — per-channel normalization over (H, W, N)
 * of a rank-4 X (H × W × C × N).  gamma, beta are length-C vectors.
 *
 *   μ_c    = (1/M) Σ_{h,w,n} X[h,w,c,n]      where M = H*W*N
 *   σ²_c   = (1/M) Σ_{h,w,n} (X - μ_c)²
 *   xhat   = (X - μ_c) / √(σ²_c + ε)
 *   Y      = γ_c * xhat + β_c
 *
 * Saves x_hat (full flat buffer) + σ_c (length-C vector) in auxData
 * for the backward pass.  No running stats / inference mode — this is
 * the training-time form; inference-only BN would freeze (μ, σ) and
 * is a documented carve-down. */
matlab_mat *matlab_dlnet_batchnorm(void *r, void *x, void *gv, void *bv) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    matlab_mat *B = get_data(bv);
    Shape4 S = shape4(X);
    int64_t M = S.H * S.W * S.N;
    if (M <= 0 || S.C <= 0) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_BATCHNORM, get_id(x), get_id(gv), Y));
        return mat_alloc(0, 0);
    }
    const double eps = 1e-5;
    std::vector<double> mu(static_cast<size_t>(S.C), 0.0);
    std::vector<double> sig(static_cast<size_t>(S.C), 0.0);
    /* μ_c */
    for (int64_t c = 0; c < S.C; ++c) {
        double s = 0;
        for (int64_t n = 0; n < S.N; ++n)
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww)
                    s += S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3];
        mu[static_cast<size_t>(c)] = s / static_cast<double>(M);
    }
    /* σ_c = sqrt(var + ε) */
    for (int64_t c = 0; c < S.C; ++c) {
        double v = 0;
        for (int64_t n = 0; n < S.N; ++n)
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    double d = S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3]
                             - mu[static_cast<size_t>(c)];
                    v += d * d;
                }
        sig[static_cast<size_t>(c)] = std::sqrt(v / static_cast<double>(M) + eps);
    }
    const double *Gd = flatdata(G);
    const double *Bd = flatdata(B);
    int64_t outDims[4] = {S.H, S.W, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    /* xhat saved as a flat row vector indexed by ((n*C + c)*H + h)*W + w. */
    int64_t xhatN = M * S.C;
    matlab_mat *xhat = mat_alloc(1, xhatN > 0 ? xhatN : 1);
    for (int64_t n = 0; n < S.N; ++n)
        for (int64_t c = 0; c < S.C; ++c) {
            double inv = 1.0 / sig[static_cast<size_t>(c)];
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    int64_t fl = ((n * S.C + c) * S.H + h) * S.W + ww;
                    double xh = (S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3]
                                 - mu[static_cast<size_t>(c)]) * inv;
                    xhat->data[fl] = xh;
                    Yd[h*Ys[0] + ww*Ys[1] + c*Ys[2] + n*Ys[3]] = Gd[c] * xh + Bd[c];
                }
        }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_BATCHNORM, get_id(x), get_id(gv), Y);
    g_tape[id].auxParents.push_back(get_id(bv));
    g_tape[id].auxData.push_back(xhat);
    matlab_mat *sigvec = mat_alloc(1, S.C);
    for (int64_t c = 0; c < S.C; ++c) sigvec->data[c] = sig[static_cast<size_t>(c)];
    g_tape[id].auxData.push_back(sigvec);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* softmax(X, dim) — softmax along the given axis on a matN / mat3 / mat.
 * For 2-D X with dim=1 this matches the existing column-wise softmax. */
matlab_mat *matlab_dlnet_softmax_dim(void *r, void *x, double dim_d) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    int dim = static_cast<int>(dim_d);
    /* For matN, walk per-axis-slice; for mat3 promote to matN view; for
     * 2-D delegate to the existing softmax forward then record under
     * OP_SOFTMAX_DIM (so backward goes through the dim-aware path). */
    int nd; int64_t dims[16]; int64_t strides[16];
    const double *Xd;
    if (mat_is_nd(X)) {
        matlab_matN *Mn = reinterpret_cast<matlab_matN *>(X);
        nd = static_cast<int>(Mn->ndims);
        if (nd > 16) nd = 16;
        for (int k = 0; k < nd; ++k) {
            dims[k] = Mn->dims[k]; strides[k] = Mn->strides[k];
        }
        Xd = Mn->data;
    } else if (mat_is_3d(X)) {
        matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(X);
        nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
        strides[0] = M3->cols; strides[1] = 1; strides[2] = M3->rows * M3->cols;
        Xd = M3->data;
    } else {
        nd = 2; dims[0] = X->rows; dims[1] = X->cols;
        strides[0] = X->cols; strides[1] = 1;
        Xd = X->data;
    }
    if (dim < 1 || dim > nd) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_SOFTMAX_DIM, get_id(x), -1, Y));
        return mat_alloc(0, 0);
    }
    int axis = dim - 1;
    void *Rv;
    if (nd == 2) Rv = mat_alloc(dims[0], dims[1]);
    else if (nd == 3) Rv = mat3_alloc(dims[0], dims[1], dims[2]);
    else            Rv = matN_alloc(nd, dims);
    double *Yd; int64_t Ys[16];
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        Yd = Rn->data;
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Yd = R3->data;
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Yd = R2->data;
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    int64_t axisLen = dims[axis];
    int64_t outerN = 1;
    for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
    int64_t idx[16] = {0};
    for (int64_t oo = 0; oo < outerN; ++oo) {
        int64_t baseSrc = 0, baseDst = 0;
        for (int k = 0; k < nd; ++k) {
            if (k == axis) continue;
            baseSrc += idx[k] * strides[k];
            baseDst += idx[k] * Ys[k];
        }
        double mx = -1e300;
        for (int64_t a = 0; a < axisLen; ++a) {
            double v = Xd[baseSrc + a * strides[axis]];
            if (v > mx) mx = v;
        }
        double sm = 0;
        for (int64_t a = 0; a < axisLen; ++a) {
            double e = std::exp(Xd[baseSrc + a * strides[axis]] - mx);
            Yd[baseDst + a * Ys[axis]] = e;
            sm += e;
        }
        double inv = (sm > 0) ? 1.0 / sm : 0.0;
        for (int64_t a = 0; a < axisLen; ++a)
            Yd[baseDst + a * Ys[axis]] *= inv;
        /* Advance idx (skip axis). */
        for (int k = nd - 1; k >= 0; --k) {
            if (k == axis) continue;
            if (++idx[k] < dims[k]) break;
            idx[k] = 0;
        }
    }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_SOFTMAX_DIM, get_id(x), -1, Y);
    matlab_mat *dm = mat_alloc(1, 1); dm->data[0] = static_cast<double>(dim);
    g_tape[id].auxData.push_back(dm);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* layernorm(X, gamma, beta, dim) — normalize X along a single feature
 * axis.  γ, β are length-K vectors (K = size(X, dim)) applied per-axis-
 * position via broadcast over every other axis.  Unlike BatchNorm, the
 * mean/variance are computed PER (non-dim) position rather than per
 * channel — the canonical formulation for Transformer / RMSNorm-class
 * stacks where each token has its own normalization.
 *
 * Forward (for one slice of length K along dim):
 *   μ        = (1/K) Σ_i x_i
 *   σ²       = (1/K) Σ_i (x_i - μ)²
 *   xhat_i   = (x_i - μ) / √(σ² + ε)
 *   y_i      = γ_i * xhat_i + β_i
 *
 * Saves xhat (full flat) + σ_per_slice (1 × outerN) in auxData for
 * backward.  Records OP_LAYERNORM with X as p0, γ as p1, β as
 * auxParents[0], and dim in auxData[2]. */
matlab_mat *matlab_dlnet_layernorm(void *r, void *x, void *gv, void *bv,
                                   double dim_d) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    matlab_mat *B = get_data(bv);
    int dim = static_cast<int>(dim_d);
    int nd; int64_t dims[16], strides[16];
    const double *Xd;
    if (mat_is_nd(X)) {
        matlab_matN *Mn = reinterpret_cast<matlab_matN *>(X);
        nd = static_cast<int>(Mn->ndims); if (nd > 16) nd = 16;
        for (int k = 0; k < nd; ++k) { dims[k] = Mn->dims[k]; strides[k] = Mn->strides[k]; }
        Xd = Mn->data;
    } else if (mat_is_3d(X)) {
        matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(X);
        nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
        strides[0] = M3->cols; strides[1] = 1; strides[2] = M3->rows * M3->cols;
        Xd = M3->data;
    } else {
        nd = 2; dims[0] = X->rows; dims[1] = X->cols;
        strides[0] = X->cols; strides[1] = 1;
        Xd = X->data;
    }
    if (dim < 1 || dim > nd) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_LAYERNORM, get_id(x), get_id(gv), Y));
        return mat_alloc(0, 0);
    }
    int axis = dim - 1;
    int64_t K = dims[axis];
    int64_t outerN = 1;
    for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
    const double eps = 1e-5;
    /* Allocate Y at same rank as X. */
    int64_t outDims[16]; for (int k = 0; k < nd; ++k) outDims[k] = dims[k];
    void *Rv = matN_alloc(nd, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[16];
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    const double *Gd = flatdata(G);
    const double *Bd = flatdata(B);
    /* xhat: 1 × (outerN * K), laid out by outer-major / axis-minor. */
    matlab_mat *xhat = mat_alloc(1, outerN * K);
    matlab_mat *sigvec = mat_alloc(1, outerN);   /* σ per slice */
    int64_t idx[16] = {0};
    for (int64_t oo = 0; oo < outerN; ++oo) {
        int64_t baseSrc = 0, baseDst = 0;
        for (int k = 0; k < nd; ++k) {
            if (k == axis) continue;
            baseSrc += idx[k] * strides[k];
            baseDst += idx[k] * Ys[k];
        }
        /* μ, σ over this slice. */
        double mu = 0;
        for (int64_t a = 0; a < K; ++a) mu += Xd[baseSrc + a * strides[axis]];
        mu /= static_cast<double>(K > 0 ? K : 1);
        double vs = 0;
        for (int64_t a = 0; a < K; ++a) {
            double d = Xd[baseSrc + a * strides[axis]] - mu;
            vs += d * d;
        }
        double sig = std::sqrt(vs / static_cast<double>(K > 0 ? K : 1) + eps);
        sigvec->data[oo] = sig;
        for (int64_t a = 0; a < K; ++a) {
            double xh = (Xd[baseSrc + a * strides[axis]] - mu) / sig;
            xhat->data[oo * K + a] = xh;
            Yd[baseDst + a * Ys[axis]] = Gd[a] * xh + Bd[a];
        }
        /* Advance idx (skip axis). */
        for (int k = nd - 1; k >= 0; --k) {
            if (k == axis) continue;
            if (++idx[k] < dims[k]) break;
            idx[k] = 0;
        }
    }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_LAYERNORM, get_id(x), get_id(gv), Y);
    g_tape[id].auxParents.push_back(get_id(bv));
    g_tape[id].auxData.push_back(xhat);
    g_tape[id].auxData.push_back(sigvec);
    matlab_mat *dm = mat_alloc(1, 1); dm->data[0] = static_cast<double>(dim);
    g_tape[id].auxData.push_back(dm);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* batchnorm_eval(X, gamma, beta, running_mean, running_var) — frozen-
 * statistics BN forward.  No autodiff backward (inference-only); the
 * tape node is OP_LEAF so dlgradient skips it.  Returns
 *   Y[h,w,c,n] = γ_c * (X[h,w,c,n] - μ_c) / √(σ²_c + ε) + β_c
 * with μ, σ² taken from the supplied running-stat vectors. */
matlab_mat *matlab_dlnet_batchnorm_eval(void *r, void *x, void *gv, void *bv,
                                        void *muv, void *varv) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    matlab_mat *B = get_data(bv);
    matlab_mat *MU = get_data(muv);
    matlab_mat *VR = get_data(varv);
    Shape4 S = shape4(X);
    if (S.C <= 0) { matlab_mat *Y = mat_alloc(0, 0);
                    set_data(r, Y);
                    set_id(r, record(OP_LEAF, -1, -1, Y));
                    return mat_alloc(0, 0); }
    const double eps = 1e-5;
    int64_t outDims[4] = {S.H, S.W, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    const double *Gd = flatdata(G), *Bd = flatdata(B);
    const double *MUd = flatdata(MU), *VRd = flatdata(VR);
    for (int64_t c = 0; c < S.C; ++c) {
        double sig = std::sqrt(VRd[c] + eps);
        double inv = 1.0 / sig;
        double gc = Gd[c], bc = Bd[c], muc = MUd[c];
        for (int64_t n = 0; n < S.N; ++n)
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    double x = S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3];
                    Yd[h*Ys[0] + ww*Ys[1] + c*Ys[2] + n*Ys[3]] = gc * (x - muc) * inv + bc;
                }
    }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    /* Record as a leaf so backward sweep treats it as a constant -- no
     * upstream gradient flows through inference-mode BN. */
    set_id(r, record(OP_LEAF, -1, -1, Y));
    return mat_alloc(0, 0);
}

/* groupnorm(X, gamma, beta, G) — split the channel axis into G groups,
 * compute (μ, σ²) per (group, sample) over (H, W, C/G), apply γ_c, β_c.
 *
 *   X : H × W × C × N  (C must be divisible by G)
 *   γ, β : length-C vectors
 *
 *   For each (g, n) with M = H*W*(C/G):
 *     μ_{g,n} = (1/M) Σ_{h,w,c∈group g} x[h,w,c,n]
 *     σ²_{g,n} = (1/M) Σ_{h,w,c∈group g} (x - μ)²
 *     xhat[h,w,c,n] = (x[h,w,c,n] - μ_{g,n}) / √(σ² + ε)
 *     y[h,w,c,n] = γ_c · xhat[h,w,c,n] + β_c
 *
 * Backward (per group, sample) is the standard 3-term form:
 *   dxhat[h,w,c,n] = dy[h,w,c,n] · γ_c
 *   dx = (1/(M·σ)) · (M·dxhat − Σdxhat − xhat·Σ(dxhat·xhat))
 *   dγ_c += Σ_{h,w,n} dy[h,w,c,n] · xhat[h,w,c,n]
 *   dβ_c += Σ_{h,w,n} dy[h,w,c,n]
 *
 * Saves xhat (full flat) + σ_per_(group,sample) (1 × G·N) + G in
 * auxData for the backward pass. */
matlab_mat *matlab_dlnet_groupnorm(void *r, void *x, void *gv, void *bv,
                                   double G_d) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    matlab_mat *B = get_data(bv);
    int64_t Gn = static_cast<int64_t>(G_d);
    Shape4 S = shape4(X);
    if (S.C <= 0 || Gn <= 0 || (S.C % Gn) != 0) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_GROUPNORM, get_id(x), get_id(gv), Y));
        return mat_alloc(0, 0);
    }
    int64_t Cpg = S.C / Gn;      /* channels per group */
    int64_t M = S.H * S.W * Cpg; /* reduction count per (group, sample) */
    const double eps = 1e-5;

    int64_t outDims[4] = {S.H, S.W, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    const double *Gd = flatdata(G);
    const double *Bd = flatdata(B);

    /* xhat: flat 1 × (H·W·C·N).  Indexed by ((n·C + c)·H + h)·W + w
     * to match the BN convention.  σ_per_gn: 1 × (G·N). */
    int64_t xhatN = S.H * S.W * S.C * S.N;
    matlab_mat *xhat = mat_alloc(1, xhatN > 0 ? xhatN : 1);
    matlab_mat *sigvec = mat_alloc(1, Gn * S.N > 0 ? Gn * S.N : 1);

    for (int64_t n = 0; n < S.N; ++n) {
        for (int64_t g = 0; g < Gn; ++g) {
            int64_t c_start = g * Cpg, c_end = c_start + Cpg;
            /* μ_{g,n} */
            double s = 0;
            for (int64_t c = c_start; c < c_end; ++c)
                for (int64_t h = 0; h < S.H; ++h)
                    for (int64_t ww = 0; ww < S.W; ++ww)
                        s += S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3];
            double mu = s / static_cast<double>(M);
            /* σ²_{g,n} */
            double vs = 0;
            for (int64_t c = c_start; c < c_end; ++c)
                for (int64_t h = 0; h < S.H; ++h)
                    for (int64_t ww = 0; ww < S.W; ++ww) {
                        double d = S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3] - mu;
                        vs += d * d;
                    }
            double sig = std::sqrt(vs / static_cast<double>(M) + eps);
            sigvec->data[n * Gn + g] = sig;
            double inv_sig = 1.0 / sig;
            for (int64_t c = c_start; c < c_end; ++c) {
                double gc = Gd[c], bc = Bd[c];
                for (int64_t h = 0; h < S.H; ++h)
                    for (int64_t ww = 0; ww < S.W; ++ww) {
                        int64_t fl = ((n * S.C + c) * S.H + h) * S.W + ww;
                        double xh = (S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3] - mu) * inv_sig;
                        xhat->data[fl] = xh;
                        Yd[h*Ys[0] + ww*Ys[1] + c*Ys[2] + n*Ys[3]] = gc * xh + bc;
                    }
            }
        }
    }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_GROUPNORM, get_id(x), get_id(gv), Y);
    g_tape[id].auxParents.push_back(get_id(bv));
    g_tape[id].auxData.push_back(xhat);
    g_tape[id].auxData.push_back(sigvec);
    matlab_mat *gG = mat_alloc(1, 1); gG->data[0] = static_cast<double>(Gn);
    g_tape[id].auxData.push_back(gG);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* batchnorm_train(X, gamma, beta, run_mean, run_var, momentum) —
 * training-mode BN that ALSO updates the supplied running-stat buffers
 * in place via an exponential moving average:
 *
 *   run_mean = (1 - α) · run_mean + α · batch_mean
 *   run_var  = (1 - α) · run_var  + α · batch_var
 *
 * The forward output uses the BATCH statistics (so the backward is the
 * standard 3-term BN form); the running stats are a side effect for
 * later inference-mode use via batchnorm_eval.
 *
 * Records under OP_BATCHNORM (the backward case handles either training
 * variant — same saved xhat + σ).  Caller maintains run_mean / run_var
 * as plain matlab_mat buffers (or as dlarrays whose extractdata they
 * thread back through the loop). */
matlab_mat *matlab_dlnet_batchnorm_train(void *r, void *x, void *gv, void *bv,
                                         void *muv, void *varv, double mom_d) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    matlab_mat *B = get_data(bv);
    matlab_mat *MU = get_data(muv);
    matlab_mat *VR = get_data(varv);
    double mom = mom_d;
    if (mom < 0.0 || mom > 1.0) mom = 0.1;
    Shape4 S = shape4(X);
    int64_t M = S.H * S.W * S.N;
    if (M <= 0 || S.C <= 0) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_BATCHNORM_TRAIN, get_id(x), get_id(gv), Y));
        return mat_alloc(0, 0);
    }
    const double eps = 1e-5;
    std::vector<double> mu(static_cast<size_t>(S.C), 0.0);
    std::vector<double> sig(static_cast<size_t>(S.C), 0.0);
    for (int64_t c = 0; c < S.C; ++c) {
        double s = 0;
        for (int64_t n = 0; n < S.N; ++n)
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww)
                    s += S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3];
        mu[static_cast<size_t>(c)] = s / static_cast<double>(M);
    }
    for (int64_t c = 0; c < S.C; ++c) {
        double v = 0;
        for (int64_t n = 0; n < S.N; ++n)
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    double d = S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3]
                             - mu[static_cast<size_t>(c)];
                    v += d * d;
                }
        sig[static_cast<size_t>(c)] = std::sqrt(v / static_cast<double>(M) + eps);
    }
    /* EMA update of running stats — IN PLACE writes to MU, VR. */
    if (MU && MU->data) {
        int64_t lim = std::min<int64_t>(MU->rows * MU->cols, S.C);
        for (int64_t c = 0; c < lim; ++c) {
            double bm = mu[static_cast<size_t>(c)];
            MU->data[c] = (1.0 - mom) * MU->data[c] + mom * bm;
        }
    }
    if (VR && VR->data) {
        int64_t lim = std::min<int64_t>(VR->rows * VR->cols, S.C);
        for (int64_t c = 0; c < lim; ++c) {
            double s = sig[static_cast<size_t>(c)];
            double bv = s * s - eps;          /* recover batch var */
            VR->data[c] = (1.0 - mom) * VR->data[c] + mom * bv;
        }
    }
    /* Forward output uses BATCH stats (so backward is correct). */
    const double *Gd = flatdata(G);
    const double *Bd = flatdata(B);
    int64_t outDims[4] = {S.H, S.W, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    int64_t xhatN = M * S.C;
    matlab_mat *xhat = mat_alloc(1, xhatN > 0 ? xhatN : 1);
    for (int64_t n = 0; n < S.N; ++n)
        for (int64_t c = 0; c < S.C; ++c) {
            double inv = 1.0 / sig[static_cast<size_t>(c)];
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    int64_t fl = ((n * S.C + c) * S.H + h) * S.W + ww;
                    double xh = (S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3]
                                 - mu[static_cast<size_t>(c)]) * inv;
                    xhat->data[fl] = xh;
                    Yd[h*Ys[0] + ww*Ys[1] + c*Ys[2] + n*Ys[3]] = Gd[c] * xh + Bd[c];
                }
        }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    /* Reuse OP_BATCHNORM's backward — same xhat / σ contract. */
    int id = record(OP_BATCHNORM, get_id(x), get_id(gv), Y);
    g_tape[id].auxParents.push_back(get_id(bv));
    g_tape[id].auxData.push_back(xhat);
    matlab_mat *sigvec = mat_alloc(1, S.C);
    for (int64_t c = 0; c < S.C; ++c) sigvec->data[c] = sig[static_cast<size_t>(c)];
    g_tape[id].auxData.push_back(sigvec);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* instancenorm(X, gamma, beta) — per-(channel, sample) normalization
 * over (H, W).  Equivalent to GroupNorm with G = C (each channel is its
 * own group).  Standard formulation:
 *   M       = H * W
 *   μ_{c,n} = (1/M) Σ_{h,w} x[h,w,c,n]
 *   σ²      = (1/M) Σ (x - μ)²
 *   xhat    = (x - μ) / √(σ² + ε)
 *   y       = γ_c · xhat + β_c
 *
 * Saves xhat + σ_per_(c,n) (1 × C·N) in auxData. */
matlab_mat *matlab_dlnet_instancenorm(void *r, void *x, void *gv, void *bv) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    matlab_mat *B = get_data(bv);
    Shape4 S = shape4(X);
    int64_t M = S.H * S.W;
    if (M <= 0 || S.C <= 0) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_INSTANCENORM, get_id(x), get_id(gv), Y));
        return mat_alloc(0, 0);
    }
    const double eps = 1e-5;
    int64_t outDims[4] = {S.H, S.W, S.C, S.N};
    void *Rv = matN_alloc(4, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[4] = {0,0,0,0};
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    const double *Gd = flatdata(G);
    const double *Bd = flatdata(B);
    int64_t xhatN = S.H * S.W * S.C * S.N;
    matlab_mat *xhat = mat_alloc(1, xhatN > 0 ? xhatN : 1);
    matlab_mat *sigvec = mat_alloc(1, S.C * S.N > 0 ? S.C * S.N : 1);
    for (int64_t n = 0; n < S.N; ++n) {
        for (int64_t c = 0; c < S.C; ++c) {
            /* μ over (H, W) for this (c, n). */
            double s = 0;
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww)
                    s += S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3];
            double mu = s / static_cast<double>(M);
            double vs = 0;
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    double d = S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3] - mu;
                    vs += d * d;
                }
            double sig = std::sqrt(vs / static_cast<double>(M) + eps);
            sigvec->data[n * S.C + c] = sig;
            double inv = 1.0 / sig;
            for (int64_t h = 0; h < S.H; ++h)
                for (int64_t ww = 0; ww < S.W; ++ww) {
                    int64_t fl = ((n * S.C + c) * S.H + h) * S.W + ww;
                    double xh = (S.data[h*S.s0 + ww*S.s1 + c*S.s2 + n*S.s3]
                                 - mu) * inv;
                    xhat->data[fl] = xh;
                    Yd[h*Ys[0] + ww*Ys[1] + c*Ys[2] + n*Ys[3]] = Gd[c] * xh + Bd[c];
                }
        }
    }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_INSTANCENORM, get_id(x), get_id(gv), Y);
    g_tape[id].auxParents.push_back(get_id(bv));
    g_tape[id].auxData.push_back(xhat);
    g_tape[id].auxData.push_back(sigvec);
    set_id(r, id);
    return mat_alloc(0, 0);
}

/* rmsnorm(X, gamma, dim) — simplified LayerNorm without mean subtraction.
 *   K       = size(X, dim)
 *   rms_i   = √( (1/K) Σ_{j∈axis} x_j² + ε )
 *   xhat    = x_i / rms_i
 *   y       = γ_i · xhat       (no β by convention)
 *
 * Saves xhat + rms_per_slice (1 × outerN) + dim in auxData. */
matlab_mat *matlab_dlnet_rmsnorm(void *r, void *x, void *gv, double dim_d) {
    using namespace dlnet;
    matlab_mat *X = get_data(x);
    matlab_mat *G = get_data(gv);
    int dim = static_cast<int>(dim_d);
    int nd; int64_t dims[16], strides[16];
    const double *Xd;
    if (mat_is_nd(X)) {
        matlab_matN *Mn = reinterpret_cast<matlab_matN *>(X);
        nd = static_cast<int>(Mn->ndims); if (nd > 16) nd = 16;
        for (int k = 0; k < nd; ++k) { dims[k] = Mn->dims[k]; strides[k] = Mn->strides[k]; }
        Xd = Mn->data;
    } else if (mat_is_3d(X)) {
        matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(X);
        nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
        strides[0] = M3->cols; strides[1] = 1; strides[2] = M3->rows * M3->cols;
        Xd = M3->data;
    } else {
        nd = 2; dims[0] = X->rows; dims[1] = X->cols;
        strides[0] = X->cols; strides[1] = 1;
        Xd = X->data;
    }
    if (dim < 1 || dim > nd) {
        matlab_mat *Y = mat_alloc(0, 0);
        set_data(r, Y);
        set_id(r, record(OP_RMSNORM, get_id(x), get_id(gv), Y));
        return mat_alloc(0, 0);
    }
    int axis = dim - 1;
    int64_t K = dims[axis];
    int64_t outerN = 1;
    for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
    const double eps = 1e-5;
    int64_t outDims[16]; for (int k = 0; k < nd; ++k) outDims[k] = dims[k];
    void *Rv = matN_alloc(nd, outDims);
    double *Yd = mat_is_nd(Rv) ? reinterpret_cast<matlab_matN *>(Rv)->data
              : mat_is_3d(Rv) ? reinterpret_cast<matlab_mat3 *>(Rv)->data
                              : reinterpret_cast<matlab_mat *>(Rv)->data;
    int64_t Ys[16];
    if (mat_is_nd(Rv)) {
        matlab_matN *Rn = reinterpret_cast<matlab_matN *>(Rv);
        for (uint32_t k = 0; k < Rn->ndims; ++k) Ys[k] = Rn->strides[k];
    } else if (mat_is_3d(Rv)) {
        matlab_mat3 *R3 = reinterpret_cast<matlab_mat3 *>(Rv);
        Ys[0] = R3->cols; Ys[1] = 1; Ys[2] = R3->rows * R3->cols;
    } else {
        matlab_mat *R2 = reinterpret_cast<matlab_mat *>(Rv);
        Ys[0] = R2->cols; Ys[1] = 1;
    }
    const double *Gd = flatdata(G);
    matlab_mat *xhat = mat_alloc(1, outerN * K);
    matlab_mat *rmsvec = mat_alloc(1, outerN);
    int64_t idx[16] = {0};
    for (int64_t oo = 0; oo < outerN; ++oo) {
        int64_t baseSrc = 0, baseDst = 0;
        for (int k = 0; k < nd; ++k) {
            if (k == axis) continue;
            baseSrc += idx[k] * strides[k];
            baseDst += idx[k] * Ys[k];
        }
        double ss = 0;
        for (int64_t a = 0; a < K; ++a) {
            double v = Xd[baseSrc + a * strides[axis]];
            ss += v * v;
        }
        double rms = std::sqrt(ss / static_cast<double>(K > 0 ? K : 1) + eps);
        rmsvec->data[oo] = rms;
        double inv = 1.0 / rms;
        for (int64_t a = 0; a < K; ++a) {
            double v = Xd[baseSrc + a * strides[axis]];
            double xh = v * inv;
            xhat->data[oo * K + a] = xh;
            Yd[baseDst + a * Ys[axis]] = Gd[a] * xh;
        }
        for (int k = nd - 1; k >= 0; --k) {
            if (k == axis) continue;
            if (++idx[k] < dims[k]) break;
            idx[k] = 0;
        }
    }
    matlab_mat *Y = reinterpret_cast<matlab_mat *>(Rv);
    set_data(r, Y);
    int id = record(OP_RMSNORM, get_id(x), get_id(gv), Y);
    g_tape[id].auxData.push_back(xhat);
    g_tape[id].auxData.push_back(rmsvec);
    matlab_mat *dm = mat_alloc(1, 1); dm->data[0] = static_cast<double>(dim);
    g_tape[id].auxData.push_back(dm);
    set_id(r, id);
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
    // seed — uses nelem/flatdata so a matN-valued loss seeds correctly.
    { matlab_mat *seed = clone(g_tape[lossId].val);
      double *sd = flatdata(seed); int64_t sn = nelem(seed);
      for (int64_t i = 0; i < sn; ++i) sd[i] = 1.0;
      accum(lossId, seed); }
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
            // Broadcast-aware: walks A's shape, reads P0/P1 via the
            // broadcasted address; accum then reduces to operand shape.
            if (P0 && P1) {
                int64_t oM = A->rows, oN = A->cols;
                int64_t pM = P0->rows, pN = P0->cols, qM = P1->rows, qN = P1->cols;
                matlab_mat *g0 = mat_alloc(oM, oN);
                matlab_mat *g1 = mat_alloc(oM, oN);
                for (int64_t r = 0; r < oM; ++r) {
                    for (int64_t c = 0; c < oN; ++c) {
                        int64_t pr = (pM == 1) ? 0 : r, pc = (pN == 1) ? 0 : c;
                        int64_t qr = (qM == 1) ? 0 : r, qc = (qN == 1) ? 0 : c;
                        double adv = A->data[r*oN + c];
                        g0->data[r*oN + c] = adv * P1->data[qr*qN + qc];
                        g1->data[r*oN + c] = adv * P0->data[pr*pN + pc];
                    }
                }
                accum(n.p0, g0);
                accum(n.p1, g1);
            }
        } break;
        case OP_MTIMES: {
            // C = P0 * P1 ; gP0 = A * P1' ; gP1 = P0' * A
            if (P0 && P1) {
                /* dlnet_gemm_ABt / _AtB go through the same GPU-aware
                 * dispatcher as the forward pass, so training with
                 * `dlnetGpu(1)` accelerates both forward AND backward
                 * MTIMES on the Metal lane (above 128³).  Below the
                 * threshold each call falls through to BLAS dgemm /
                 * the naive triple-loop without overhead. */
                accum(n.p0, dlnet_gemm_ABt(A,  P1));
                accum(n.p1, dlnet_gemm_AtB(P0, A));
            }
        } break;
        case OP_RELU:    { matlab_mat *g=zero_clone(V); double *gd=flatdata(g); const double *Vd=flatdata(V), *Ad=flatdata(A); int64_t ne=nelem(V); for(int64_t k=0;k<ne;++k) gd[k]= (Vd[k]>0)?Ad[k]:0; accum(n.p0,g);} break;
        case OP_SIGMOID: { matlab_mat *g=zero_clone(V); double *gd=flatdata(g); const double *Vd=flatdata(V), *Ad=flatdata(A); int64_t ne=nelem(V); for(int64_t k=0;k<ne;++k){double y=Vd[k]; gd[k]=Ad[k]*y*(1-y);} accum(n.p0,g);} break;
        case OP_TANH:    { matlab_mat *g=zero_clone(V); double *gd=flatdata(g); const double *Vd=flatdata(V), *Ad=flatdata(A); int64_t ne=nelem(V); for(int64_t k=0;k<ne;++k){double y=Vd[k]; gd[k]=Ad[k]*(1-y*y);} accum(n.p0,g);} break;
        case OP_EXP:     { matlab_mat *g=zero_clone(V); double *gd=flatdata(g); const double *Vd=flatdata(V), *Ad=flatdata(A); int64_t ne=nelem(V); for(int64_t k=0;k<ne;++k) gd[k]=Ad[k]*Vd[k]; accum(n.p0,g);} break;
        case OP_LOG:     { matlab_mat *g=zero_clone(V); double *gd=flatdata(g); const double *Pd=flatdata(P0), *Ad=flatdata(A); int64_t ne=nelem(V); for(int64_t k=0;k<ne;++k) gd[k]=Ad[k]/Pd[k]; accum(n.p0,g);} break;
        case OP_SOFTMAX: {
            // gx = y .* (adj - sum(adj.*y, over rows)) per column
            matlab_mat *g=mat_alloc(V->rows,V->cols);
            for (int64_t c=0;c<V->cols;++c){ double dot=0; for(int64_t r=0;r<V->rows;++r) dot+=A->data[r*A->cols+c]*V->data[r*V->cols+c];
                for(int64_t r=0;r<V->rows;++r){ double y=V->data[r*V->cols+c]; g->data[r*g->cols+c]=y*(A->data[r*A->cols+c]-dot);} }
            accum(n.p0,g);
        } break;
        case OP_SUM:  { matlab_mat *g=zero_clone(P0); double *gd=flatdata(g); int64_t ne=nelem(P0); for(int64_t k=0;k<ne;++k) gd[k]=A->data[0]; accum(n.p0,g);} break;
        case OP_MEAN: { int64_t nel=P0->rows*P0->cols; matlab_mat *g=mat_alloc(P0->rows,P0->cols); for(int64_t k=0;k<nel;++k) g->data[k]=A->data[0]/(nel?nel:1); accum(n.p0,g);} break;
        case OP_CE: {
            // L = -sum(T.*log(Y))/N ; gY = -(T./Y)/N * adj
            if (P0 && P1) { int64_t Nb = P0->cols>0?P0->cols:1; matlab_mat *g=mat_alloc(P0->rows,P0->cols);
                for(int64_t k=0;k<P0->rows*P0->cols;++k){ double y=P0->data[k]; g->data[k]= -A->data[0]*(P1->data[k]/(y>1e-12?y:1e-12))/Nb; } accum(n.p0,g); }
        } break;
        case OP_MSE: {
            /* Rank-agnostic: walks via nelem/flatdata so matN-valued
             * forward (e.g. conv output vs target rank-4 tensor) back-
             * props correctly without crashing on the magic word. */
            if (P0 && P1) {
                int64_t nel = nelem(P0);
                if (nelem(P1) != nel) break;
                matlab_mat *g = clone(P0);
                double *gd = flatdata(g);
                const double *Pd = flatdata(P0), *Qd = flatdata(P1);
                double scale = A->data[0] * 2.0 / (nel ? nel : 1);
                for (int64_t k = 0; k < nel; ++k)
                    gd[k] = scale * (Pd[k] - Qd[k]);
                accum(n.p0, g);
            }
        } break;
        case OP_RDIV: {
            // C = A ./ B ;  gA = adj / B ;  gB = -adj * A / B^2 ;
            // walks the result shape (A's shape == adj shape) and indexes
            // operands with broadcasted addresses so 1×N/M×1/1×1 operands
            // produce the right per-cell gradient before accum reduces.
            if (P0 && P1) {
                int64_t oM = A->rows, oN = A->cols;
                int64_t pM = P0->rows, pN = P0->cols;
                int64_t qM = P1->rows, qN = P1->cols;
                matlab_mat *gA = mat_alloc(oM, oN);
                matlab_mat *gB = mat_alloc(oM, oN);
                for (int64_t r = 0; r < oM; ++r) {
                    for (int64_t c = 0; c < oN; ++c) {
                        int64_t pr = (pM == 1) ? 0 : r, pc = (pN == 1) ? 0 : c;
                        int64_t qr = (qM == 1) ? 0 : r, qc = (qN == 1) ? 0 : c;
                        double a = P0->data[pr*pN + pc];
                        double b = P1->data[qr*qN + qc];
                        double bs = (std::fabs(b) > 1e-30) ? b : (b < 0 ? -1e-30 : 1e-30);
                        double adv = A->data[r*oN + c];
                        gA->data[r*oN + c] =  adv / bs;
                        gB->data[r*oN + c] = -adv * a / (bs*bs);
                    }
                }
                accum(n.p0, gA);
                accum(n.p1, gB);
            }
        } break;
        case OP_SQRT: {
            // gx = adj / (2 * sqrt(x))  = adj / (2*V)
            matlab_mat *g = mat_alloc(V->rows, V->cols);
            for (int64_t k = 0; k < V->rows*V->cols; ++k) {
                double y = V->data[k];
                g->data[k] = (y > 1e-30) ? A->data[k] / (2.0*y) : 0.0;
            }
            accum(n.p0, g);
        } break;
        case OP_MEAN_DIM: {
            // V is 1xN (dim=1) or Mx1 (dim=2); spread adj evenly over the
            // reduced axis of P0.
            if (P0) {
                matlab_mat *g = mat_alloc(P0->rows, P0->cols);
                int64_t M = P0->rows, N = P0->cols;
                if (V->rows == 1) {
                    // mean over rows -> each input col scaled by adj[c]/M.
                    for (int64_t r = 0; r < M; ++r)
                        for (int64_t c = 0; c < N; ++c)
                            g->data[r*N + c] = A->data[c] / (M > 0 ? M : 1);
                } else {
                    // mean over cols -> each input row scaled by adj[r]/N.
                    for (int64_t r = 0; r < M; ++r)
                        for (int64_t c = 0; c < N; ++c)
                            g->data[r*N + c] = A->data[r] / (N > 0 ? N : 1);
                }
                accum(n.p0, g);
            }
        } break;
        case OP_LEAKY_RELU: {
            matlab_mat *g = mat_alloc(V->rows, V->cols);
            // y = x for x>0 else 0.01*x; dy/dx = 1 if x>0 else 0.01.
            // V holds y; pre-relu sign = sign of y (positive y came from positive x).
            for (int64_t k = 0; k < V->rows*V->cols; ++k)
                g->data[k] = A->data[k] * ((V->data[k] > 0) ? 1.0 : 0.01);
            accum(n.p0, g);
        } break;
        case OP_GELU: {
            // y = x * sigmoid(1.702*x).  Need x (P0) to recompute σ.
            matlab_mat *g = mat_alloc(V->rows, V->cols);
            if (P0) {
                for (int64_t k = 0; k < V->rows*V->cols; ++k) {
                    double x = P0->data[k];
                    double s = 1.0 / (1.0 + std::exp(-1.702 * x));
                    double dydx = s + 1.702 * x * s * (1.0 - s);
                    g->data[k] = A->data[k] * dydx;
                }
            }
            accum(n.p0, g);
        } break;
        case OP_SWISH: {
            // y = x * sigmoid(x); dy/dx = σ(x) + y * (1 - σ(x)).
            matlab_mat *g = mat_alloc(V->rows, V->cols);
            if (P0) {
                for (int64_t k = 0; k < V->rows*V->cols; ++k) {
                    double x = P0->data[k];
                    double s = 1.0 / (1.0 + std::exp(-x));
                    double dydx = s + V->data[k] * (1.0 - s);
                    g->data[k] = A->data[k] * dydx;
                }
            }
            accum(n.p0, g);
        } break;
        case OP_SOFTPLUS: {
            // y = log(1+exp(x)) ; dy/dx = σ(x).
            matlab_mat *g = mat_alloc(V->rows, V->cols);
            if (P0) {
                for (int64_t k = 0; k < V->rows*V->cols; ++k) {
                    double x = P0->data[k];
                    double s = 1.0 / (1.0 + std::exp(-x));
                    g->data[k] = A->data[k] * s;
                }
            }
            accum(n.p0, g);
        } break;
        case OP_ELU: {
            // dy/dx = 1 if x>0 else α*exp(x) = y+α (with α=1).  Use P0 for branch.
            matlab_mat *g = mat_alloc(V->rows, V->cols);
            if (P0) {
                for (int64_t k = 0; k < V->rows*V->cols; ++k) {
                    double x = P0->data[k];
                    g->data[k] = A->data[k] * ((x > 0) ? 1.0 : (V->data[k] + 1.0));
                }
            }
            accum(n.p0, g);
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
        case OP_CONV2D_BATCH: {
            /* Forward: Y = conv2d_batch(X, W)
             *   X : H x W x C x N (matN; may be mat3 when N==1 or mat when C=N=1)
             *   W : kH x kW x C x K (matN; may be mat3/mat)
             *   Y : Hout x Wout x K x N
             * Backward (dY = A, the adj of Y):
             *   dW[kh,kw,c,k] = sum_{h,w,n} A[h,w,k,n] * X[h+kh, w+kw, c, n]
             *   dX[h+kh, w+kw, c, n] = sum_{kh,kw,k} A[h,w,k,n] * W[kh,kw,c,k]
             * Both expressed as GEMM via the im2col layout (same shape used
             * by the forward path), reusing matlab_im2col_2d + matlab_matmul_mm
             * for the BLAS-accelerated reduction. */
            if (!P0 || !P1) break;
            /* Recover shape from operand descriptors (X, W). */
            auto getShape4 = [](const matlab_mat *m, int64_t &d0, int64_t &d1,
                                                       int64_t &d2, int64_t &d3) {
                if (mat_is_nd(m)) {
                    const matlab_matN *Mn = reinterpret_cast<const matlab_matN *>(m);
                    d0 = Mn->dims[0];
                    d1 = Mn->dims[1];
                    d2 = Mn->ndims >= 3 ? Mn->dims[2] : 1;
                    d3 = Mn->ndims >= 4 ? Mn->dims[3] : 1;
                } else if (mat_is_3d(m)) {
                    const matlab_mat3 *M3 = reinterpret_cast<const matlab_mat3 *>(m);
                    d0 = M3->rows; d1 = M3->cols; d2 = M3->depth; d3 = 1;
                } else {
                    d0 = m->rows; d1 = m->cols; d2 = 1; d3 = 1;
                }
            };
            int64_t H, Wd, C, N, kH, kW, Cw, K;
            getShape4(P0, H, Wd, C,  N);
            getShape4(P1, kH, kW, Cw, K);
            if (Cw != C) break;
            int64_t Hout = H - kH + 1, Wout = Wd - kW + 1;
            int64_t inner = kH * kW * C;
            int64_t hwn = Hout * Wout * N;

            /* Lay out A as dY_2d (K x Hout*Wout*N) so the row-k slice
             * indexes A[h, w, k, n] at column n*Hout*Wout + h*Wout + w. */
            matlab_mat *dY_2d = mat_alloc(K, hwn);
            {
                /* A may be matN / mat3 / mat depending on trailing-singleton
                 * drop; recover its strides per descriptor rank. */
                int64_t As0, As1, As2, As3; const double *Ad;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                    Ad = Mn->data;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows * M3->cols; As3 = 0;
                    Ad = M3->data;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                    Ad = A->data;
                }
                for (int64_t k = 0; k < K; ++k)
                    for (int64_t n = 0; n < N; ++n)
                        for (int64_t h = 0; h < Hout; ++h)
                            for (int64_t w = 0; w < Wout; ++w) {
                                int64_t col = n * Hout * Wout + h * Wout + w;
                                dY_2d->data[k * hwn + col] =
                                    Ad[h*As0 + w*As1 + k*As2 + n*As3];
                            }
            }

            /* dW path: X_col (inner x hwn) via shared im2col helper. */
            matlab_mat *X_col = reinterpret_cast<matlab_mat *>(
                matlab_im2col_2d(P0, static_cast<double>(kH), static_cast<double>(kW)));
            /* X_col^T (hwn x inner) so the matmul gives the right shape. */
            matlab_mat *XcolT = mat_alloc(hwn, inner);
            for (int64_t r2 = 0; r2 < inner; ++r2)
                for (int64_t c2 = 0; c2 < hwn; ++c2)
                    XcolT->data[c2 * inner + r2] = X_col->data[r2 * hwn + c2];
            free(X_col->data); free(X_col);
            /* dW_2d (K x inner) = dY_2d * XcolT. */
            matlab_mat *dW_2d = matlab_matmul_mm(dY_2d, XcolT);
            free(XcolT->data); free(XcolT);
            /* Scatter dW_2d into a parent-shape-matching descriptor for W. */
            matlab_mat *dW_full = clone(P1);   /* same shape, will be overwritten */
            {
                /* Zero out. */
                double *dst = flatdata(dW_full);
                int64_t nn = nelem(dW_full);
                for (int64_t i = 0; i < nn; ++i) dst[i] = 0.0;
                int64_t Ws0, Ws1, Ws2, Ws3;
                if (mat_is_nd(dW_full)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(dW_full);
                    Ws0 = Mn->strides[0]; Ws1 = Mn->strides[1];
                    Ws2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    Ws3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(dW_full)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(dW_full);
                    Ws0 = M3->cols; Ws1 = 1; Ws2 = M3->rows * M3->cols; Ws3 = 0;
                } else {
                    Ws0 = dW_full->cols; Ws1 = 1; Ws2 = 0; Ws3 = 0;
                }
                for (int64_t k = 0; k < K; ++k)
                    for (int64_t c = 0; c < C; ++c)
                        for (int64_t kh = 0; kh < kH; ++kh)
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t col = c * kH * kW + kh * kW + kw;
                                dst[kh*Ws0 + kw*Ws1 + c*Ws2 + k*Ws3] =
                                    dW_2d->data[k * inner + col];
                            }
            }
            free(dW_2d->data); free(dW_2d);
            accum(n.p1, dW_full);
            /* Free our temporary (accum cloned the contribution). */
            { double *dst = flatdata(dW_full); (void)dst;
              if (mat_is_nd(dW_full)) {
                  /* matN_alloc allocated the descriptor + dims/strides in one
                   * block; free the data + the descriptor. */
                  free(reinterpret_cast<matlab_matN *>(dW_full)->data);
                  free(dW_full);
              } else if (mat_is_3d(dW_full)) {
                  free(reinterpret_cast<matlab_mat3 *>(dW_full)->data);
                  free(dW_full);
              } else {
                  free(dW_full->data); free(dW_full);
              }
            }

            /* dX path: W_2d (K x inner) -> W_2d^T (inner x K) -> col_grad
             * (inner x hwn) = W_2d^T * dY_2d.  Then col2im scatters to dX. */
            matlab_mat *W2d_T = mat_alloc(inner, K);
            {
                int64_t Ws0, Ws1, Ws2, Ws3;
                if (mat_is_nd(P1)) {
                    const matlab_matN *Mn = reinterpret_cast<const matlab_matN *>(P1);
                    Ws0 = Mn->strides[0]; Ws1 = Mn->strides[1];
                    Ws2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    Ws3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(P1)) {
                    const matlab_mat3 *M3 = reinterpret_cast<const matlab_mat3 *>(P1);
                    Ws0 = M3->cols; Ws1 = 1; Ws2 = M3->rows * M3->cols; Ws3 = 0;
                } else {
                    Ws0 = P1->cols; Ws1 = 1; Ws2 = 0; Ws3 = 0;
                }
                const double *Wd2 = flatdata(P1);
                for (int64_t k = 0; k < K; ++k)
                    for (int64_t c = 0; c < C; ++c)
                        for (int64_t kh = 0; kh < kH; ++kh)
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t row = c * kH * kW + kh * kW + kw;
                                W2d_T->data[row * K + k] =
                                    Wd2[kh*Ws0 + kw*Ws1 + c*Ws2 + k*Ws3];
                            }
            }
            matlab_mat *col_grad = matlab_matmul_mm(W2d_T, dY_2d);
            free(W2d_T->data); free(W2d_T);
            free(dY_2d->data); free(dY_2d);

            /* col2im: scatter (inner x hwn) back to dX (H x Wd x C x N),
             * summing overlapping patches. */
            matlab_mat *dX_full = clone(P0);
            {
                double *dst = flatdata(dX_full);
                int64_t nn = nelem(dX_full);
                for (int64_t i = 0; i < nn; ++i) dst[i] = 0.0;
                int64_t Xs0, Xs1, Xs2, Xs3;
                if (mat_is_nd(dX_full)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(dX_full);
                    Xs0 = Mn->strides[0]; Xs1 = Mn->strides[1];
                    Xs2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    Xs3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(dX_full)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(dX_full);
                    Xs0 = M3->cols; Xs1 = 1; Xs2 = M3->rows * M3->cols; Xs3 = 0;
                } else {
                    Xs0 = dX_full->cols; Xs1 = 1; Xs2 = 0; Xs3 = 0;
                }
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t h = 0; h < Hout; ++h)
                        for (int64_t w = 0; w < Wout; ++w) {
                            int64_t col = n * Hout * Wout + h * Wout + w;
                            for (int64_t c = 0; c < C; ++c)
                                for (int64_t kh = 0; kh < kH; ++kh)
                                    for (int64_t kw = 0; kw < kW; ++kw) {
                                        int64_t row = c * kH * kW + kh * kW + kw;
                                        dst[(h + kh)*Xs0 + (w + kw)*Xs1
                                            + c*Xs2 + n*Xs3] +=
                                            col_grad->data[row * hwn + col];
                                    }
                        }
            }
            free(col_grad->data); free(col_grad);
            accum(n.p0, dX_full);
            if (mat_is_nd(dX_full)) {
                free(reinterpret_cast<matlab_matN *>(dX_full)->data);
                free(dX_full);
            } else if (mat_is_3d(dX_full)) {
                free(reinterpret_cast<matlab_mat3 *>(dX_full)->data);
                free(dX_full);
            } else {
                free(dX_full->data); free(dX_full);
            }
        } break;
        case OP_RESHAPE: {
            /* Reshape is an identity on the flat buffer; the gradient
             * just needs to flow back at P0's original rank. */
            if (P0) {
                matlab_mat *g = zero_clone(P0);
                double *gd = flatdata(g);
                const double *Ad = flatdata(A);
                int64_t ne = nelem(g);
                if (nelem(A) == ne && Ad && gd)
                    memcpy(gd, Ad, static_cast<size_t>(ne) * sizeof(double));
                accum(n.p0, g);
            }
        } break;
        case OP_MAXPOOL2D: {
            /* Route dY back to the arg-max input cell within each window.
             * Shape recovery + stride lookup mirror the forward path. */
            if (P0 && n.auxData.size() >= 2) {
                matlab_mat *kK = n.auxData[0];
                matlab_mat *argmax = n.auxData[1];
                int64_t kH = static_cast<int64_t>(kK->data[0]);
                int64_t kW = static_cast<int64_t>(kK->data[1]);
                Shape4 SX = shape4(P0);
                int64_t Hout = SX.H / kH, Wout = SX.W / kW;
                /* dY (A) strides — same pattern as Y. */
                int64_t As0, As1, As2, As3;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows*M3->cols; As3 = 0;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                }
                const double *Ad = flatdata(A);
                matlab_mat *g = zero_clone(P0);
                double *gd = flatdata(g);
                /* dX strides come from g (same shape as P0). */
                int64_t gs0, gs1, gs2, gs3;
                if (mat_is_nd(g)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(g);
                    gs0 = Mn->strides[0]; gs1 = Mn->strides[1];
                    gs2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    gs3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(g)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(g);
                    gs0 = M3->cols; gs1 = 1; gs2 = M3->rows*M3->cols; gs3 = 0;
                } else {
                    gs0 = g->cols; gs1 = 1; gs2 = 0; gs3 = 0;
                }
                for (int64_t nn = 0; nn < SX.N; ++nn)
                    for (int64_t c = 0; c < SX.C; ++c)
                        for (int64_t hO = 0; hO < Hout; ++hO)
                            for (int64_t wO = 0; wO < Wout; ++wO) {
                                int64_t fl = ((nn * SX.C + c) * Hout + hO) * Wout + wO;
                                int64_t enc = static_cast<int64_t>(argmax->data[fl]);
                                int64_t hi = enc / SX.W;
                                int64_t wi = enc - hi * SX.W;
                                double dY = Ad[hO*As0 + wO*As1 + c*As2 + nn*As3];
                                gd[hi*gs0 + wi*gs1 + c*gs2 + nn*gs3] += dY;
                            }
                accum(n.p0, g);
            }
        } break;
        case OP_AVGPOOL2D: {
            /* Spread dY uniformly across each window cell. */
            if (P0 && n.auxData.size() >= 1) {
                matlab_mat *kK = n.auxData[0];
                int64_t kH = static_cast<int64_t>(kK->data[0]);
                int64_t kW = static_cast<int64_t>(kK->data[1]);
                Shape4 SX = shape4(P0);
                int64_t Hout = SX.H / kH, Wout = SX.W / kW;
                int64_t As0, As1, As2, As3;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows*M3->cols; As3 = 0;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                }
                const double *Ad = flatdata(A);
                matlab_mat *g = zero_clone(P0);
                double *gd = flatdata(g);
                int64_t gs0, gs1, gs2, gs3;
                if (mat_is_nd(g)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(g);
                    gs0 = Mn->strides[0]; gs1 = Mn->strides[1];
                    gs2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    gs3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(g)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(g);
                    gs0 = M3->cols; gs1 = 1; gs2 = M3->rows*M3->cols; gs3 = 0;
                } else {
                    gs0 = g->cols; gs1 = 1; gs2 = 0; gs3 = 0;
                }
                double inv = 1.0 / static_cast<double>(kH * kW);
                for (int64_t nn = 0; nn < SX.N; ++nn)
                    for (int64_t c = 0; c < SX.C; ++c)
                        for (int64_t hO = 0; hO < Hout; ++hO)
                            for (int64_t wO = 0; wO < Wout; ++wO) {
                                double dY = Ad[hO*As0 + wO*As1 + c*As2 + nn*As3];
                                double share = dY * inv;
                                for (int64_t kh = 0; kh < kH; ++kh)
                                    for (int64_t kw = 0; kw < kW; ++kw) {
                                        int64_t hi = hO * kH + kh, wi = wO * kW + kw;
                                        gd[hi*gs0 + wi*gs1 + c*gs2 + nn*gs3] += share;
                                    }
                            }
                accum(n.p0, g);
            }
        } break;
        case OP_BATCHNORM: {
            /* P0 = X, P1 = γ, auxParents[0] = β.
             * auxData[0] = xhat (flat 1 × (H*W*C*N)), auxData[1] = σ (1 × C). */
            if (P0 && P1 && n.auxParents.size() >= 1 && n.auxData.size() >= 2) {
                matlab_mat *xhat = n.auxData[0];
                matlab_mat *sigvec = n.auxData[1];
                Shape4 SX = shape4(P0);
                int64_t M = SX.H * SX.W * SX.N;
                if (M <= 0 || SX.C <= 0) break;
                int64_t As0, As1, As2, As3;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows * M3->cols; As3 = 0;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                }
                const double *Ad = flatdata(A);
                const double *Gd = flatdata(P1);
                /* dγ_c = Σ dy·xhat ; dβ_c = Σ dy. */
                matlab_mat *dG = zero_clone(P1);
                matlab_mat *dB = zero_clone(g_tape[n.auxParents[0]].val);
                double *dGd = flatdata(dG);
                double *dBd = flatdata(dB);
                for (int64_t c = 0; c < SX.C; ++c) {
                    double sDY = 0, sDYxh = 0;
                    for (int64_t nn = 0; nn < SX.N; ++nn)
                        for (int64_t h = 0; h < SX.H; ++h)
                            for (int64_t ww = 0; ww < SX.W; ++ww) {
                                int64_t fl = ((nn * SX.C + c) * SX.H + h) * SX.W + ww;
                                double dy = Ad[h*As0 + ww*As1 + c*As2 + nn*As3];
                                sDY += dy;
                                sDYxh += dy * xhat->data[fl];
                            }
                    dGd[c] = sDYxh;
                    dBd[c] = sDY;
                }
                accum(n.p1, dG);
                accum(n.auxParents[0], dB);
                /* dX = (γ_c / (M·σ_c)) · (M·dy − Σdy − xhat · Σ(dy·xhat)). */
                matlab_mat *dX = zero_clone(P0);
                double *dXd = flatdata(dX);
                int64_t Xs0, Xs1, Xs2, Xs3;
                if (mat_is_nd(dX)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(dX);
                    Xs0 = Mn->strides[0]; Xs1 = Mn->strides[1];
                    Xs2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    Xs3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(dX)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(dX);
                    Xs0 = M3->cols; Xs1 = 1; Xs2 = M3->rows * M3->cols; Xs3 = 0;
                } else {
                    Xs0 = dX->cols; Xs1 = 1; Xs2 = 0; Xs3 = 0;
                }
                for (int64_t c = 0; c < SX.C; ++c) {
                    double sDY = dBd[c];
                    double sDYxh = dGd[c];
                    double sig = sigvec->data[c];
                    double scale = Gd[c] / (static_cast<double>(M) * sig);
                    for (int64_t nn = 0; nn < SX.N; ++nn)
                        for (int64_t h = 0; h < SX.H; ++h)
                            for (int64_t ww = 0; ww < SX.W; ++ww) {
                                int64_t fl = ((nn * SX.C + c) * SX.H + h) * SX.W + ww;
                                double dy = Ad[h*As0 + ww*As1 + c*As2 + nn*As3];
                                double xh = xhat->data[fl];
                                dXd[h*Xs0 + ww*Xs1 + c*Xs2 + nn*Xs3] =
                                    scale * (static_cast<double>(M) * dy - sDY - xh * sDYxh);
                            }
                }
                accum(n.p0, dX);
            }
        } break;
        case OP_CONV2D_FULL: {
            /* P0 = X, P1 = W, auxParents[0] = b.
             * auxData[0] = (pad_h, pad_w, stride_h, stride_w).
             *
             * im2col + GEMM backward (matches the forward path):
             *   X_col   : (kH·kW·C) x (Hout·Wout·N)   via matlab_im2col_2d_pad
             *   dY_2d   : K x (Hout·Wout·N)
             *   dW_2d   : K x (kH·kW·C)               = dY_2d · X_col^T
             *   col_grad: (kH·kW·C) x (Hout·Wout·N)   = W_2d^T · dY_2d
             *   dX      : col2im_pad(col_grad)        scatter back with overlap
             *   db_k    : Σ_{n,h,w} dy[h,w,k,n]
             */
            if (P0 && P1 && n.auxData.size() >= 1) {
                matlab_mat *cfg = n.auxData[0];
                int64_t pad_h = static_cast<int64_t>(cfg->data[0]);
                int64_t pad_w = static_cast<int64_t>(cfg->data[1]);
                int64_t stride_h = static_cast<int64_t>(cfg->data[2]);
                int64_t stride_w = static_cast<int64_t>(cfg->data[3]);
                if (stride_h <= 0) stride_h = 1;
                if (stride_w <= 0) stride_w = 1;
                Shape4 SX = shape4(P0);
                Shape4 SW = shape4(P1);
                int64_t kH = SW.H, kW = SW.W, K = SW.N;
                int64_t Hout = (SX.H + 2*pad_h - kH) / stride_h + 1;
                int64_t Wout = (SX.W + 2*pad_w - kW) / stride_w + 1;
                int64_t inner = kH * kW * SX.C;
                int64_t hwn = Hout * Wout * SX.N;
                if (Hout <= 0 || Wout <= 0 || inner <= 0 || hwn <= 0) break;

                /* dY_2d (K x hwn): repack A using its own strides. */
                int64_t As0, As1, As2, As3;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows * M3->cols; As3 = 0;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                }
                const double *Ad = flatdata(A);
                matlab_mat *dY_2d = mat_alloc(K, hwn);
                for (int64_t k = 0; k < K; ++k)
                    for (int64_t nn = 0; nn < SX.N; ++nn)
                        for (int64_t hO = 0; hO < Hout; ++hO)
                            for (int64_t wO = 0; wO < Wout; ++wO) {
                                int64_t col = nn * Hout * Wout + hO * Wout + wO;
                                dY_2d->data[k * hwn + col] =
                                    Ad[hO*As0 + wO*As1 + k*As2 + nn*As3];
                            }

                /* db_k = sum across columns. */
                matlab_mat *db = zero_clone(g_tape[n.auxParents[0]].val);
                double *dbd = flatdata(db);
                int64_t dbN = nelem(db);
                for (int64_t k = 0; k < K; ++k) {
                    double s = 0;
                    for (int64_t col = 0; col < hwn; ++col)
                        s += dY_2d->data[k * hwn + col];
                    if (k < dbN) dbd[k] += s;
                }
                accum(n.auxParents[0], db);

                /* X_col via im2col_2d_pad. */
                matlab_mat *X_col = matlab_im2col_2d_pad(P0,
                    static_cast<double>(kH), static_cast<double>(kW),
                    static_cast<double>(pad_h), static_cast<double>(pad_w),
                    static_cast<double>(stride_h), static_cast<double>(stride_w));
                /* X_col^T (hwn x inner). */
                matlab_mat *XcolT = mat_alloc(hwn, inner);
                for (int64_t r2 = 0; r2 < inner; ++r2)
                    for (int64_t c2 = 0; c2 < hwn; ++c2)
                        XcolT->data[c2 * inner + r2] = X_col->data[r2 * hwn + c2];
                /* dW_2d (K x inner) = dY_2d * X_col^T. */
                matlab_mat *dW_2d = matlab_matmul_mm(dY_2d, XcolT);
                free(XcolT->data); free(XcolT);

                /* Scatter dW_2d into W-shaped dW (matN / mat3 / mat). */
                matlab_mat *dW = zero_clone(P1);
                {
                    double *dWd = flatdata(dW);
                    int64_t Ws0, Ws1, Ws2, Ws3;
                    if (mat_is_nd(dW)) {
                        matlab_matN *Mn = reinterpret_cast<matlab_matN *>(dW);
                        Ws0 = Mn->strides[0]; Ws1 = Mn->strides[1];
                        Ws2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                        Ws3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                    } else if (mat_is_3d(dW)) {
                        matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(dW);
                        Ws0 = M3->cols; Ws1 = 1; Ws2 = M3->rows * M3->cols; Ws3 = 0;
                    } else {
                        Ws0 = dW->cols; Ws1 = 1; Ws2 = 0; Ws3 = 0;
                    }
                    for (int64_t k = 0; k < K; ++k)
                        for (int64_t c = 0; c < SX.C; ++c)
                            for (int64_t kh = 0; kh < kH; ++kh)
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t col = c * kH * kW + kh * kW + kw;
                                    dWd[kh*Ws0 + kw*Ws1 + c*Ws2 + k*Ws3] =
                                        dW_2d->data[k * inner + col];
                                }
                }
                free(dW_2d->data); free(dW_2d);

                /* W_2d^T (inner x K). */
                matlab_mat *W2d_T = mat_alloc(inner, K);
                {
                    int64_t Ws0 = SW.s0, Ws1 = SW.s1, Ws2 = SW.s2, Ws3 = SW.s3;
                    const double *Wd_ = flatdata(P1);
                    for (int64_t k = 0; k < K; ++k)
                        for (int64_t c = 0; c < SX.C; ++c)
                            for (int64_t kh = 0; kh < kH; ++kh)
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t row = c * kH * kW + kh * kW + kw;
                                    W2d_T->data[row * K + k] =
                                        Wd_[kh*Ws0 + kw*Ws1 + c*Ws2 + k*Ws3];
                                }
                }
                /* col_grad (inner x hwn) = W_2d^T * dY_2d. */
                matlab_mat *col_grad = matlab_matmul_mm(W2d_T, dY_2d);
                free(W2d_T->data); free(W2d_T);
                free(dY_2d->data); free(dY_2d);
                free(X_col->data); free(X_col);

                /* col2im-pad: scatter col_grad back to dX with stride+padding,
                 * accumulating overlapping patches. */
                matlab_mat *dX = zero_clone(P0);
                {
                    double *dXd = flatdata(dX);
                    int64_t Xs0 = SX.s0, Xs1 = SX.s1, Xs2 = SX.s2, Xs3 = SX.s3;
                    for (int64_t nn = 0; nn < SX.N; ++nn)
                        for (int64_t hO = 0; hO < Hout; ++hO)
                            for (int64_t wO = 0; wO < Wout; ++wO) {
                                int64_t col = nn * Hout * Wout + hO * Wout + wO;
                                for (int64_t c = 0; c < SX.C; ++c)
                                    for (int64_t kh = 0; kh < kH; ++kh) {
                                        int64_t hi = hO * stride_h - pad_h + kh;
                                        if (hi < 0 || hi >= SX.H) continue;
                                        for (int64_t kw = 0; kw < kW; ++kw) {
                                            int64_t wi = wO * stride_w - pad_w + kw;
                                            if (wi < 0 || wi >= SX.W) continue;
                                            int64_t row = c * kH * kW + kh * kW + kw;
                                            dXd[hi*Xs0 + wi*Xs1 + c*Xs2 + nn*Xs3]
                                                += col_grad->data[row * hwn + col];
                                        }
                                    }
                            }
                }
                free(col_grad->data); free(col_grad);
                accum(n.p0, dX);
                accum(n.p1, dW);
            }
        } break;
        case OP_SOFTMAX_DIM: {
            /* dxhat[i] = y[i] * (dy[i] - Σ_j(dy[j] * y[j]))  along axis. */
            if (P0 && n.auxData.size() >= 1) {
                int dim = static_cast<int>(n.auxData[0]->data[0]);
                int nd; int64_t dims[16], Vstr[16], Pstr[16];
                const double *Vd = flatdata(V);
                if (mat_is_nd(V)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(V);
                    nd = static_cast<int>(Mn->ndims); if (nd > 16) nd = 16;
                    for (int k = 0; k < nd; ++k) {
                        dims[k] = Mn->dims[k]; Vstr[k] = Mn->strides[k];
                    }
                } else if (mat_is_3d(V)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(V);
                    nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
                    Vstr[0] = M3->cols; Vstr[1] = 1; Vstr[2] = M3->rows*M3->cols;
                } else {
                    nd = 2; dims[0] = V->rows; dims[1] = V->cols;
                    Vstr[0] = V->cols; Vstr[1] = 1;
                }
                /* P0 strides (may differ if it's a different rank — but
                 * here softmax preserves shape, so P0 has same rank). */
                if (mat_is_nd(P0)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(P0);
                    for (uint32_t k = 0; k < Mn->ndims && static_cast<int>(k) < 16; ++k) Pstr[k] = Mn->strides[k];
                } else if (mat_is_3d(P0)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(P0);
                    Pstr[0] = M3->cols; Pstr[1] = 1; Pstr[2] = M3->rows*M3->cols;
                } else {
                    Pstr[0] = P0->cols; Pstr[1] = 1;
                }
                int axis = dim - 1;
                if (axis < 0 || axis >= nd) break;
                int64_t axisLen = dims[axis];
                int64_t outerN = 1;
                for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
                const double *Ad = flatdata(A);
                matlab_mat *g = zero_clone(P0);
                double *gd = flatdata(g);
                int64_t Astr[16];
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    for (uint32_t k = 0; k < Mn->ndims && static_cast<int>(k) < 16; ++k) Astr[k] = Mn->strides[k];
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    Astr[0] = M3->cols; Astr[1] = 1; Astr[2] = M3->rows*M3->cols;
                } else {
                    Astr[0] = A->cols; Astr[1] = 1;
                }
                int64_t idx[16] = {0};
                for (int64_t oo = 0; oo < outerN; ++oo) {
                    int64_t baseV = 0, baseA = 0, baseG = 0;
                    for (int k = 0; k < nd; ++k) {
                        if (k == axis) continue;
                        baseV += idx[k] * Vstr[k];
                        baseA += idx[k] * Astr[k];
                        baseG += idx[k] * Pstr[k];
                    }
                    /* dot = Σ_j dy_j · y_j along axis. */
                    double dot = 0;
                    for (int64_t a = 0; a < axisLen; ++a) {
                        double y = Vd[baseV + a * Vstr[axis]];
                        double dy = Ad[baseA + a * Astr[axis]];
                        dot += dy * y;
                    }
                    for (int64_t a = 0; a < axisLen; ++a) {
                        double y = Vd[baseV + a * Vstr[axis]];
                        double dy = Ad[baseA + a * Astr[axis]];
                        gd[baseG + a * Pstr[axis]] = y * (dy - dot);
                    }
                    for (int k = nd - 1; k >= 0; --k) {
                        if (k == axis) continue;
                        if (++idx[k] < dims[k]) break;
                        idx[k] = 0;
                    }
                }
                accum(n.p0, g);
            }
        } break;
        case OP_MEAN_DIM_ND: {
            /* dX[..,i,..] += dY[..] / axisLen along axis. */
            if (P0 && n.auxData.size() >= 1) {
                int dim = static_cast<int>(n.auxData[0]->data[0]);
                int nd; int64_t dims[16], Xstr[16];
                if (mat_is_nd(P0)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(P0);
                    nd = static_cast<int>(Mn->ndims); if (nd > 16) nd = 16;
                    for (int k = 0; k < nd; ++k) {
                        dims[k] = Mn->dims[k]; Xstr[k] = Mn->strides[k];
                    }
                } else if (mat_is_3d(P0)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(P0);
                    nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
                    Xstr[0] = M3->cols; Xstr[1] = 1; Xstr[2] = M3->rows*M3->cols;
                } else {
                    nd = 2; dims[0] = P0->rows; dims[1] = P0->cols;
                    Xstr[0] = P0->cols; Xstr[1] = 1;
                }
                int axis = dim - 1;
                if (axis < 0 || axis >= nd) break;
                int64_t axisLen = dims[axis];
                /* A has output shape (axis collapsed to 1). */
                int64_t Astr[16];
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    for (uint32_t k = 0; k < Mn->ndims && static_cast<int>(k) < 16; ++k) Astr[k] = Mn->strides[k];
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    Astr[0] = M3->cols; Astr[1] = 1; Astr[2] = M3->rows*M3->cols;
                } else {
                    Astr[0] = A->cols; Astr[1] = 1;
                }
                const double *Ad = flatdata(A);
                matlab_mat *g = zero_clone(P0);
                double *gd = flatdata(g);
                int64_t outerN = 1;
                for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
                double inv = axisLen > 0 ? 1.0 / static_cast<double>(axisLen) : 0;
                int64_t idx[16] = {0};
                for (int64_t oo = 0; oo < outerN; ++oo) {
                    /* For A (axis collapsed), the address along that axis is 0. */
                    int64_t baseA = 0;
                    for (int k = 0; k < nd; ++k) {
                        if (k == axis) continue;
                        baseA += idx[k] * Astr[k];
                    }
                    double dy = Ad[baseA];
                    int64_t baseX = 0;
                    for (int k = 0; k < nd; ++k) {
                        if (k == axis) continue;
                        baseX += idx[k] * Xstr[k];
                    }
                    for (int64_t a = 0; a < axisLen; ++a)
                        gd[baseX + a * Xstr[axis]] = dy * inv;
                    for (int k = nd - 1; k >= 0; --k) {
                        if (k == axis) continue;
                        if (++idx[k] < dims[k]) break;
                        idx[k] = 0;
                    }
                }
                accum(n.p0, g);
            }
        } break;
        case OP_LAYERNORM: {
            /* P0 = X, P1 = γ, auxParents[0] = β.
             * auxData[0] = xhat (1 × outerN*K), [1] = σ (1 × outerN), [2] = dim.
             *
             * Per-slice (length K) backward (LayerNorm has dγ/dβ summed
             * across non-axis positions but per-axis-i; dx uses the σ
             * for THAT slice, not a per-channel sigma like BN):
             *   dxhat_i  = dy_i * γ_i
             *   dx_i     = (1/σ) * (dxhat_i - mean(dxhat) - xhat_i * mean(dxhat*xhat))
             *   dγ_i    += Σ_outer  dy_i * xhat_i_at_that_outer
             *   dβ_i    += Σ_outer  dy_i
             */
            if (P0 && P1 && n.auxParents.size() >= 1 && n.auxData.size() >= 3) {
                matlab_mat *xhat = n.auxData[0];
                matlab_mat *sigvec = n.auxData[1];
                int dim = static_cast<int>(n.auxData[2]->data[0]);
                int nd; int64_t dims[16], Xstr[16];
                if (mat_is_nd(P0)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(P0);
                    nd = static_cast<int>(Mn->ndims); if (nd > 16) nd = 16;
                    for (int k = 0; k < nd; ++k) {
                        dims[k] = Mn->dims[k]; Xstr[k] = Mn->strides[k];
                    }
                } else if (mat_is_3d(P0)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(P0);
                    nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
                    Xstr[0] = M3->cols; Xstr[1] = 1; Xstr[2] = M3->rows * M3->cols;
                } else {
                    nd = 2; dims[0] = P0->rows; dims[1] = P0->cols;
                    Xstr[0] = P0->cols; Xstr[1] = 1;
                }
                int axis = dim - 1;
                if (axis < 0 || axis >= nd) break;
                int64_t K = dims[axis];
                int64_t outerN = 1;
                for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
                /* Adjoint A strides. */
                int64_t As0[16];
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    for (uint32_t k = 0; k < Mn->ndims && static_cast<int>(k) < 16; ++k) As0[k] = Mn->strides[k];
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0[0] = M3->cols; As0[1] = 1; As0[2] = M3->rows*M3->cols;
                } else {
                    As0[0] = A->cols; As0[1] = 1;
                }
                const double *Ad = flatdata(A);
                const double *Gd = flatdata(P1);
                /* dG, dB are length-K vectors (1×K shape from γ/β). */
                matlab_mat *dG = zero_clone(P1);
                matlab_mat *dB = zero_clone(g_tape[n.auxParents[0]].val);
                double *dGd = flatdata(dG);
                double *dBd = flatdata(dB);
                matlab_mat *dX = zero_clone(P0);
                double *dXd = flatdata(dX);
                int64_t idx[16] = {0};
                for (int64_t oo = 0; oo < outerN; ++oo) {
                    int64_t baseA = 0, baseX = 0;
                    for (int k = 0; k < nd; ++k) {
                        if (k == axis) continue;
                        baseA += idx[k] * As0[k];
                        baseX += idx[k] * Xstr[k];
                    }
                    double sig = sigvec->data[oo];
                    /* Compute dxhat_i = dy_i * γ_i, plus the running means. */
                    double mean_dxh = 0, mean_dxh_xh = 0;
                    /* First pass: build dxhat means; per-i dγ/dβ. */
                    for (int64_t a = 0; a < K; ++a) {
                        double dy = Ad[baseA + a * As0[axis]];
                        double xh = xhat->data[oo * K + a];
                        double dxhat = dy * Gd[a];
                        mean_dxh    += dxhat;
                        mean_dxh_xh += dxhat * xh;
                        dGd[a] += dy * xh;
                        dBd[a] += dy;
                    }
                    mean_dxh    /= static_cast<double>(K);
                    mean_dxh_xh /= static_cast<double>(K);
                    /* Second pass: dx_i. */
                    double inv_sig = 1.0 / sig;
                    for (int64_t a = 0; a < K; ++a) {
                        double dy = Ad[baseA + a * As0[axis]];
                        double xh = xhat->data[oo * K + a];
                        double dxhat = dy * Gd[a];
                        double dx = (dxhat - mean_dxh - xh * mean_dxh_xh) * inv_sig;
                        dXd[baseX + a * Xstr[axis]] = dx;
                    }
                    /* Advance idx (skip axis). */
                    for (int k = nd - 1; k >= 0; --k) {
                        if (k == axis) continue;
                        if (++idx[k] < dims[k]) break;
                        idx[k] = 0;
                    }
                }
                accum(n.p1, dG);
                accum(n.auxParents[0], dB);
                accum(n.p0, dX);
            }
        } break;
        case OP_GROUPNORM: {
            /* P0 = X, P1 = γ, auxParents[0] = β.
             * auxData[0] = xhat (1 × H·W·C·N), [1] = σ (1 × G·N), [2] = G.
             * Per (group, sample): M = H*W*(C/G).
             *   dxhat[h,w,c,n] = dy · γ_c
             *   dx = (1/(M·σ_{g,n})) · (M·dxhat − ΣdxhatGroup − xhat·Σ(dxhat·xhat)Group)
             *   dγ_c += Σ dy · xhat (over h,w,n)
             *   dβ_c += Σ dy        (over h,w,n)
             */
            if (P0 && P1 && n.auxParents.size() >= 1 && n.auxData.size() >= 3) {
                matlab_mat *xhat = n.auxData[0];
                matlab_mat *sigvec = n.auxData[1];
                int64_t Gn = static_cast<int64_t>(n.auxData[2]->data[0]);
                Shape4 SX = shape4(P0);
                if (Gn <= 0 || (SX.C % Gn) != 0) break;
                int64_t Cpg = SX.C / Gn;
                int64_t M = SX.H * SX.W * Cpg;
                if (M <= 0) break;
                int64_t As0, As1, As2, As3;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows * M3->cols; As3 = 0;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                }
                const double *Ad = flatdata(A);
                const double *Gd = flatdata(P1);
                /* dγ_c, dβ_c — per-channel sums over (h, w, n). */
                matlab_mat *dG = zero_clone(P1);
                matlab_mat *dB = zero_clone(g_tape[n.auxParents[0]].val);
                double *dGd = flatdata(dG);
                double *dBd = flatdata(dB);
                /* Per-(group, sample): cache Σdxhat and Σ(dxhat·xhat). */
                std::vector<double> sumDxh(static_cast<size_t>(Gn * SX.N), 0.0);
                std::vector<double> sumDxhXh(static_cast<size_t>(Gn * SX.N), 0.0);
                for (int64_t nn = 0; nn < SX.N; ++nn) {
                    for (int64_t g = 0; g < Gn; ++g) {
                        double s_dxh = 0, s_dxh_xh = 0;
                        int64_t c_start = g * Cpg, c_end = c_start + Cpg;
                        for (int64_t c = c_start; c < c_end; ++c)
                            for (int64_t h = 0; h < SX.H; ++h)
                                for (int64_t ww = 0; ww < SX.W; ++ww) {
                                    int64_t fl = ((nn * SX.C + c) * SX.H + h) * SX.W + ww;
                                    double dy = Ad[h*As0 + ww*As1 + c*As2 + nn*As3];
                                    double xh = xhat->data[fl];
                                    double dxh = dy * Gd[c];
                                    s_dxh    += dxh;
                                    s_dxh_xh += dxh * xh;
                                    dGd[c] += dy * xh;
                                    dBd[c] += dy;
                                }
                        sumDxh   [static_cast<size_t>(nn * Gn + g)] = s_dxh;
                        sumDxhXh [static_cast<size_t>(nn * Gn + g)] = s_dxh_xh;
                    }
                }
                accum(n.p1, dG);
                accum(n.auxParents[0], dB);
                /* dX */
                matlab_mat *dX = zero_clone(P0);
                double *dXd = flatdata(dX);
                int64_t Xs0, Xs1, Xs2, Xs3;
                if (mat_is_nd(dX)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(dX);
                    Xs0 = Mn->strides[0]; Xs1 = Mn->strides[1];
                    Xs2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    Xs3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(dX)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(dX);
                    Xs0 = M3->cols; Xs1 = 1; Xs2 = M3->rows * M3->cols; Xs3 = 0;
                } else {
                    Xs0 = dX->cols; Xs1 = 1; Xs2 = 0; Xs3 = 0;
                }
                for (int64_t nn = 0; nn < SX.N; ++nn) {
                    for (int64_t g = 0; g < Gn; ++g) {
                        double sig = sigvec->data[nn * Gn + g];
                        double inv_M_sig = 1.0 / (static_cast<double>(M) * sig);
                        double s_dxh = sumDxh   [static_cast<size_t>(nn * Gn + g)];
                        double s_dxh_xh = sumDxhXh[static_cast<size_t>(nn * Gn + g)];
                        int64_t c_start = g * Cpg, c_end = c_start + Cpg;
                        for (int64_t c = c_start; c < c_end; ++c)
                            for (int64_t h = 0; h < SX.H; ++h)
                                for (int64_t ww = 0; ww < SX.W; ++ww) {
                                    int64_t fl = ((nn * SX.C + c) * SX.H + h) * SX.W + ww;
                                    double dy = Ad[h*As0 + ww*As1 + c*As2 + nn*As3];
                                    double xh = xhat->data[fl];
                                    double dxh = dy * Gd[c];
                                    double dx = inv_M_sig * (static_cast<double>(M) * dxh
                                                             - s_dxh - xh * s_dxh_xh);
                                    dXd[h*Xs0 + ww*Xs1 + c*Xs2 + nn*Xs3] = dx;
                                }
                    }
                }
                accum(n.p0, dX);
            }
        } break;
        case OP_INSTANCENORM: {
            /* P0 = X, P1 = γ, auxParents[0] = β.
             * auxData[0] = xhat (1 × H·W·C·N), [1] = σ (1 × C·N).
             * Per (c, n): M = H*W, 3-term BN-style backward. */
            if (P0 && P1 && n.auxParents.size() >= 1 && n.auxData.size() >= 2) {
                matlab_mat *xhat = n.auxData[0];
                matlab_mat *sigvec = n.auxData[1];
                Shape4 SX = shape4(P0);
                int64_t M = SX.H * SX.W;
                if (M <= 0 || SX.C <= 0) break;
                int64_t As0, As1, As2, As3;
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    As0 = Mn->strides[0]; As1 = Mn->strides[1];
                    As2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    As3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0 = M3->cols; As1 = 1; As2 = M3->rows * M3->cols; As3 = 0;
                } else {
                    As0 = A->cols; As1 = 1; As2 = 0; As3 = 0;
                }
                const double *Ad = flatdata(A);
                const double *Gd = flatdata(P1);
                matlab_mat *dG = zero_clone(P1);
                matlab_mat *dB = zero_clone(g_tape[n.auxParents[0]].val);
                double *dGd = flatdata(dG);
                double *dBd = flatdata(dB);
                /* Cache Σdxhat, Σ(dxhat·xhat) per (c, n). */
                std::vector<double> sDxh(static_cast<size_t>(SX.C * SX.N), 0.0);
                std::vector<double> sDxhXh(static_cast<size_t>(SX.C * SX.N), 0.0);
                for (int64_t nn = 0; nn < SX.N; ++nn)
                    for (int64_t c = 0; c < SX.C; ++c) {
                        double s_dxh = 0, s_dxh_xh = 0;
                        for (int64_t h = 0; h < SX.H; ++h)
                            for (int64_t ww = 0; ww < SX.W; ++ww) {
                                int64_t fl = ((nn * SX.C + c) * SX.H + h) * SX.W + ww;
                                double dy = Ad[h*As0 + ww*As1 + c*As2 + nn*As3];
                                double xh = xhat->data[fl];
                                double dxh = dy * Gd[c];
                                s_dxh    += dxh;
                                s_dxh_xh += dxh * xh;
                                dGd[c] += dy * xh;
                                dBd[c] += dy;
                            }
                        sDxh   [static_cast<size_t>(nn * SX.C + c)] = s_dxh;
                        sDxhXh [static_cast<size_t>(nn * SX.C + c)] = s_dxh_xh;
                    }
                accum(n.p1, dG);
                accum(n.auxParents[0], dB);
                matlab_mat *dX = zero_clone(P0);
                double *dXd = flatdata(dX);
                int64_t Xs0, Xs1, Xs2, Xs3;
                if (mat_is_nd(dX)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(dX);
                    Xs0 = Mn->strides[0]; Xs1 = Mn->strides[1];
                    Xs2 = Mn->ndims >= 3 ? Mn->strides[2] : 0;
                    Xs3 = Mn->ndims >= 4 ? Mn->strides[3] : 0;
                } else if (mat_is_3d(dX)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(dX);
                    Xs0 = M3->cols; Xs1 = 1; Xs2 = M3->rows * M3->cols; Xs3 = 0;
                } else {
                    Xs0 = dX->cols; Xs1 = 1; Xs2 = 0; Xs3 = 0;
                }
                for (int64_t nn = 0; nn < SX.N; ++nn)
                    for (int64_t c = 0; c < SX.C; ++c) {
                        double sig = sigvec->data[nn * SX.C + c];
                        double inv_M_sig = 1.0 / (static_cast<double>(M) * sig);
                        double s_dxh   = sDxh   [static_cast<size_t>(nn * SX.C + c)];
                        double s_dxh_xh = sDxhXh[static_cast<size_t>(nn * SX.C + c)];
                        for (int64_t h = 0; h < SX.H; ++h)
                            for (int64_t ww = 0; ww < SX.W; ++ww) {
                                int64_t fl = ((nn * SX.C + c) * SX.H + h) * SX.W + ww;
                                double dy = Ad[h*As0 + ww*As1 + c*As2 + nn*As3];
                                double xh = xhat->data[fl];
                                double dxh = dy * Gd[c];
                                double dx = inv_M_sig * (static_cast<double>(M) * dxh
                                                         - s_dxh - xh * s_dxh_xh);
                                dXd[h*Xs0 + ww*Xs1 + c*Xs2 + nn*Xs3] = dx;
                            }
                    }
                accum(n.p0, dX);
            }
        } break;
        case OP_RMSNORM: {
            /* P0 = X, P1 = γ.  auxData[0] = xhat (1 × outerN·K),
             *                  auxData[1] = rms_per_slice (1 × outerN),
             *                  auxData[2] = dim.
             * Per-slice (length K):
             *   dxhat_i = dy_i · γ_i
             *   dx_i = (1/rms) · (dxhat_i − xhat_i · mean(dxhat · xhat))
             *   dγ_i += Σ_outer dy_i · xhat_i
             */
            if (P0 && P1 && n.auxData.size() >= 3) {
                matlab_mat *xhat = n.auxData[0];
                matlab_mat *rmsvec = n.auxData[1];
                int dim = static_cast<int>(n.auxData[2]->data[0]);
                int nd; int64_t dims[16], Xstr[16];
                if (mat_is_nd(P0)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(P0);
                    nd = static_cast<int>(Mn->ndims); if (nd > 16) nd = 16;
                    for (int k = 0; k < nd; ++k) {
                        dims[k] = Mn->dims[k]; Xstr[k] = Mn->strides[k];
                    }
                } else if (mat_is_3d(P0)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(P0);
                    nd = 3; dims[0] = M3->rows; dims[1] = M3->cols; dims[2] = M3->depth;
                    Xstr[0] = M3->cols; Xstr[1] = 1; Xstr[2] = M3->rows * M3->cols;
                } else {
                    nd = 2; dims[0] = P0->rows; dims[1] = P0->cols;
                    Xstr[0] = P0->cols; Xstr[1] = 1;
                }
                int axis = dim - 1;
                if (axis < 0 || axis >= nd) break;
                int64_t K = dims[axis];
                int64_t outerN = 1;
                for (int k = 0; k < nd; ++k) if (k != axis) outerN *= dims[k];
                int64_t As0_[16];
                if (mat_is_nd(A)) {
                    matlab_matN *Mn = reinterpret_cast<matlab_matN *>(A);
                    for (uint32_t k = 0; k < Mn->ndims && static_cast<int>(k) < 16; ++k) As0_[k] = Mn->strides[k];
                } else if (mat_is_3d(A)) {
                    matlab_mat3 *M3 = reinterpret_cast<matlab_mat3 *>(A);
                    As0_[0] = M3->cols; As0_[1] = 1; As0_[2] = M3->rows*M3->cols;
                } else {
                    As0_[0] = A->cols; As0_[1] = 1;
                }
                const double *Ad = flatdata(A);
                const double *Gd = flatdata(P1);
                matlab_mat *dG = zero_clone(P1);
                double *dGd = flatdata(dG);
                matlab_mat *dX = zero_clone(P0);
                double *dXd = flatdata(dX);
                int64_t idx[16] = {0};
                for (int64_t oo = 0; oo < outerN; ++oo) {
                    int64_t baseA = 0, baseX = 0;
                    for (int k = 0; k < nd; ++k) {
                        if (k == axis) continue;
                        baseA += idx[k] * As0_[k];
                        baseX += idx[k] * Xstr[k];
                    }
                    double rms = rmsvec->data[oo];
                    /* First pass: dγ, mean(dxhat · xhat). */
                    double sum_dxh_xh = 0;
                    for (int64_t a = 0; a < K; ++a) {
                        double dy = Ad[baseA + a * As0_[axis]];
                        double xh = xhat->data[oo * K + a];
                        double dxh = dy * Gd[a];
                        sum_dxh_xh += dxh * xh;
                        dGd[a] += dy * xh;
                    }
                    double mean_dxh_xh = sum_dxh_xh / static_cast<double>(K);
                    /* Second pass: dx_i. */
                    double inv_rms = 1.0 / rms;
                    for (int64_t a = 0; a < K; ++a) {
                        double dy = Ad[baseA + a * As0_[axis]];
                        double xh = xhat->data[oo * K + a];
                        double dxh = dy * Gd[a];
                        double dx = inv_rms * (dxh - xh * mean_dxh_xh);
                        dXd[baseX + a * Xstr[axis]] = dx;
                    }
                    for (int k = nd - 1; k >= 0; --k) {
                        if (k == axis) continue;
                        if (++idx[k] < dims[k]) break;
                        idx[k] = 0;
                    }
                }
                accum(n.p1, dG);
                accum(n.p0, dX);
            }
        } break;
        case OP_VERTCAT: {
            /* Forward: Y = [P0; P1] (Ar+Br rows, C cols).
             * Backward: dP0 = A(1:Ar, :);  dP1 = A(Ar+1:end, :). */
            if (P0 && P1 && A) {
                int64_t Ar = P0->rows, Br = P1->rows, C = P0->cols;
                if (A->cols == C && A->rows == Ar + Br) {
                    matlab_mat *dA = mat_alloc(Ar, C);
                    matlab_mat *dB = mat_alloc(Br, C);
                    for (int64_t i = 0; i < Ar; ++i)
                        for (int64_t j = 0; j < C; ++j) dA->data[i * C + j] = A->data[i * C + j];
                    for (int64_t i = 0; i < Br; ++i)
                        for (int64_t j = 0; j < C; ++j) dB->data[i * C + j] = A->data[(Ar + i) * C + j];
                    accum(n.p0, dA);
                    accum(n.p1, dB);
                }
            }
        } break;
        case OP_HORZCAT: {
            /* Forward: Y = [P0  P1] (R rows, Ac+Bc cols).
             * Backward: dP0 = A(:, 1:Ac);  dP1 = A(:, Ac+1:end). */
            if (P0 && P1 && A) {
                int64_t R = P0->rows, Ac = P0->cols, Bc = P1->cols;
                int64_t Yc = Ac + Bc;
                if (A->rows == R && A->cols == Yc) {
                    matlab_mat *dA = mat_alloc(R, Ac);
                    matlab_mat *dB = mat_alloc(R, Bc);
                    for (int64_t i = 0; i < R; ++i) {
                        for (int64_t j = 0; j < Ac; ++j) dA->data[i * Ac + j] = A->data[i * Yc + j];
                        for (int64_t j = 0; j < Bc; ++j) dB->data[i * Bc + j] = A->data[i * Yc + (Ac + j)];
                    }
                    accum(n.p0, dA);
                    accum(n.p1, dB);
                }
            }
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
matlab_mat *matlab_dlnet_reset0(void) { dlnet::g_tape.clear(); return mat_alloc(0,0); }

// ---- Tape-scoping primitives ----------------------------------------------
// Without explicit scoping, the dlnet tape grows monotonically across
// training iterations — each forward records new nodes that survive
// until the program exits.  For long-running loops this means:
//   (a) memory grows ~linearly with iter count, and
//   (b) every dlgradient walks all prior nodes (zero-init adj for
//       each), so per-iter wall time grows too.
//
// The recommended pattern is:
//
//   for k = 1:N
//       dlreset();           % truncate tape back to zero
//       Y = forward(...);    % build a fresh subgraph
//       loss = ...;
//       g    = dlgradient(loss, w);
//       w    = update(w, g);
//   end
//
// dltape_size() returns the current tape node count (debug + tests).
// dltape_truncate(n) drops everything past index n — lets a caller
// "checkpoint + restore" for partial reuse (e.g. shared encoder, fresh
// classifier head per minibatch).
double matlab_dltape_size(double dummy) {
    (void)dummy;
    return static_cast<double>(dlnet::g_tape.size());
}
matlab_mat *matlab_dltape_truncate(double n) {
    int64_t target = static_cast<int64_t>(n);
    if (target < 0) target = 0;
    if (target > static_cast<int64_t>(dlnet::g_tape.size()))
        target = static_cast<int64_t>(dlnet::g_tape.size());
    dlnet::g_tape.resize(static_cast<size_t>(target));
    return mat_alloc(0, 0);
}

/* ---- Functional optimizers ------------------------------------------------
 *
 * MATLAB exposes a small family of "update" functions that take the current
 * parameter, its gradient, and the optimiser-state buffers; return the new
 * parameter while updating the state buffers in place.  Same shape as the
 * BatchNorm-train EMA pattern (running stats threaded through dlarrays).
 *
 * Each helper is rank-agnostic — walks via nelem/flatdata so matN / mat3 /
 * mat all work.  Caller maintains the state buffers as plain numeric matrices
 * (or dlarrays whose extractdata they thread back through the loop). */

/* SGD with momentum:
 *   v_new   = β·v + (1 − β)·g       (heavy-ball variant)
 *   w_new   = w − lr·v_new
 * `v` is updated in place. */
matlab_mat *matlab_dlnet_sgdmupdate(matlab_mat *W, matlab_mat *G,
                                    matlab_mat *V,
                                    double lr, double beta) {
    using namespace dlnet;
    if (!W || !G || !V) return mat_alloc(0, 0);
    int64_t nw = nelem(W), ng = nelem(G), nv = nelem(V);
    if (nw != ng || nw != nv) return mat_alloc(0, 0);
    double *Wd = flatdata(W);
    const double *Gd = flatdata(G);
    double *Vd = flatdata(V);
    matlab_mat *Wn = clone(W);
    double *Wnd = flatdata(Wn);
    double oneMinusB = 1.0 - beta;
    for (int64_t i = 0; i < nw; ++i) {
        double vnew = beta * Vd[i] + oneMinusB * Gd[i];
        Vd[i] = vnew;
        Wnd[i] = Wd[i] - lr * vnew;
    }
    return Wn;
}

/* Adam (Kingma & Ba 2014):
 *   m_new   = β1·m + (1 − β1)·g
 *   v_new   = β2·v + (1 − β2)·g²
 *   m_hat   = m_new / (1 − β1^t)
 *   v_hat   = v_new / (1 − β2^t)
 *   w_new   = w − lr · m_hat / (√v_hat + ε)
 * `m` and `v` are updated in place; `t` is the 1-based step counter
 * (caller increments per call). */
matlab_mat *matlab_dlnet_adamupdate(matlab_mat *W, matlab_mat *G,
                                    matlab_mat *M, matlab_mat *V,
                                    double t_d,
                                    double lr, double b1, double b2, double eps) {
    using namespace dlnet;
    if (!W || !G || !M || !V) return mat_alloc(0, 0);
    int64_t nw = nelem(W);
    if (nelem(G) != nw || nelem(M) != nw || nelem(V) != nw) return mat_alloc(0, 0);
    double *Wd = flatdata(W);
    const double *Gd = flatdata(G);
    double *Md = flatdata(M);
    double *Vd = flatdata(V);
    matlab_mat *Wn = clone(W);
    double *Wnd = flatdata(Wn);
    int64_t t = static_cast<int64_t>(t_d);
    if (t < 1) t = 1;
    double bc1 = 1.0 - std::pow(b1, static_cast<double>(t));
    double bc2 = 1.0 - std::pow(b2, static_cast<double>(t));
    if (bc1 <= 0) bc1 = 1e-12;
    if (bc2 <= 0) bc2 = 1e-12;
    double inv_bc1 = 1.0 / bc1;
    double inv_bc2 = 1.0 / bc2;
    double oneMinusB1 = 1.0 - b1;
    double oneMinusB2 = 1.0 - b2;
    for (int64_t i = 0; i < nw; ++i) {
        double g = Gd[i];
        double mnew = b1 * Md[i] + oneMinusB1 * g;
        double vnew = b2 * Vd[i] + oneMinusB2 * g * g;
        Md[i] = mnew;
        Vd[i] = vnew;
        double mhat = mnew * inv_bc1;
        double vhat = vnew * inv_bc2;
        Wnd[i] = Wd[i] - lr * mhat / (std::sqrt(vhat) + eps);
    }
    return Wn;
}

/* RMSProp (Hinton):
 *   v_new = γ·v + (1 − γ)·g²
 *   w_new = w − lr · g / (√v_new + ε)
 * `v` (running mean-square) is updated in place. */
matlab_mat *matlab_dlnet_rmspropupdate(matlab_mat *W, matlab_mat *G,
                                       matlab_mat *V,
                                       double lr, double gamma, double eps) {
    using namespace dlnet;
    if (!W || !G || !V) return mat_alloc(0, 0);
    int64_t nw = nelem(W);
    if (nelem(G) != nw || nelem(V) != nw) return mat_alloc(0, 0);
    double *Wd = flatdata(W);
    const double *Gd = flatdata(G);
    double *Vd = flatdata(V);
    matlab_mat *Wn = clone(W);
    double *Wnd = flatdata(Wn);
    double oneMinusG = 1.0 - gamma;
    for (int64_t i = 0; i < nw; ++i) {
        double g = Gd[i];
        double vnew = gamma * Vd[i] + oneMinusG * g * g;
        Vd[i] = vnew;
        Wnd[i] = Wd[i] - lr * g / (std::sqrt(vnew) + eps);
    }
    return Wn;
}

/* Magnitude-based pruning mask.
 * Returns a 0/1 matrix the same shape as W whose entries are 0 wherever
 * |W| falls below the (frac*100)-th percentile of |W|, 1 otherwise.
 * Applying the mask as `W .* M` zeros the bottom-frac of weights so the
 * remaining sparse tensor still fits the H2 SV/FPGA datapath. */
matlab_mat *matlab_dlnet_prune_mask(matlab_mat *W, double frac) {
    using namespace dlnet;
    if (!W) return mat_alloc(0, 0);
    int64_t n = nelem(W);
    matlab_mat *M = clone(W);
    double *Md = flatdata(M);
    if (n == 0) return M;
    if (frac <= 0.0) {
        for (int64_t i = 0; i < n; ++i) Md[i] = 1.0;
        return M;
    }
    if (frac >= 1.0) {
        for (int64_t i = 0; i < n; ++i) Md[i] = 0.0;
        return M;
    }
    const double *Wd = flatdata(W);
    std::vector<double> absV(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) absV[static_cast<size_t>(i)] = std::fabs(Wd[i]);
    /* k-th smallest |W| is the cutoff; weights with |w| <= cutoff are pruned. */
    int64_t k = static_cast<int64_t>(std::floor(frac * static_cast<double>(n)));
    if (k <= 0) k = 1;
    if (k > n) k = n;
    std::nth_element(absV.begin(), absV.begin() + (k - 1), absV.end());
    double cutoff = absV[static_cast<size_t>(k - 1)];
    for (int64_t i = 0; i < n; ++i)
        Md[i] = (std::fabs(Wd[i]) > cutoff) ? 1.0 : 0.0;
    return M;
}

/* Programmatic experiment-sweep harness.
 * runExperiment(@trialFn, Grid) — Grid is N x K (one row per trial, one
 * column per hyperparameter).  Calls trialFn(row_i_as_column) per row;
 * trialFn must take a K x 1 dlarray-or-matrix and return a scalar metric.
 * Returns an N x 1 column of metrics. */
typedef double (*dlnet_obj_fn)(matlab_mat *);

matlab_mat *matlab_dlnet_run_experiment(void *fn_p, matlab_mat *Gridm) {
    using namespace dlnet;
    if (!fn_p || !Gridm) return mat_alloc(0, 0);
    dlnet_obj_fn f = reinterpret_cast<dlnet_obj_fn>(fn_p);
    int64_t N = Gridm->rows, K = Gridm->cols;
    if (N <= 0 || K <= 0) return mat_alloc(0, 0);
    matlab_mat *Out = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) {
        matlab_mat *row = mat_alloc(K, 1);
        for (int64_t j = 0; j < K; ++j) {
            /* matlab_mat is row-major: elem(r, c) = data[r * cols + c]. */
            row->data[j] = Gridm->data[i * K + j];
        }
        Out->data[i] = f(row);
    }
    return Out;
}

/* Sparsity of a 0/1 mask: fraction of entries equal to zero. */
double matlab_dlnet_mask_sparsity(matlab_mat *M) {
    using namespace dlnet;
    if (!M) return 0.0;
    int64_t n = nelem(M);
    if (n == 0) return 0.0;
    const double *Md = flatdata(M);
    int64_t zeros = 0;
    for (int64_t i = 0; i < n; ++i)
        if (Md[i] == 0.0) ++zeros;
    return static_cast<double>(zeros) / static_cast<double>(n);
}

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

/* ---- DL HDL Tier H1/T6.5 — quantize to an externally-provided scale ----
 * dlqclip(X, scale) projects every element of X onto the int8 lattice
 * { -127*scale, ..., 0, ..., 127*scale } using round-to-nearest with
 * symmetric clipping.  Companion to `dlquantize` (which uses the
 * tensor's own max-abs); used when calibration data picks the scale
 * separately (e.g. activations whose run-time range is wider than any
 * single batch). */
matlab_mat *matlab_dlnet_qclip(matlab_mat *X, matlab_mat *Sm) {
    if (!X || X->rows*X->cols == 0) return mat_alloc(0, 0);
    double scale = (Sm && Sm->rows*Sm->cols > 0) ? Sm->data[0] : 1.0;
    if (scale <= 0) scale = 1.0;
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    int64_t nel = X->rows * X->cols;
    for (int64_t i = 0; i < nel; ++i) {
        double q = std::round(X->data[i] / scale);
        if (q >  127.0) q =  127.0;
        if (q < -127.0) q = -127.0;
        Y->data[i] = q * scale;
    }
    return Y;
}

/* dlqcalibrate(X, runningMaxAbs) -> max(runningMaxAbs, max(abs(X))).
 * Drives the calibration pass: invoked once per calibration batch, the
 * caller threads the running maximum through and divides by 127 once at
 * the end to get the int8 scale.  Both args are scalars except for X
 * which can be any-shape matrix. */
matlab_mat *matlab_dlnet_qcalibrate(matlab_mat *X, matlab_mat *Rm) {
    double running = (Rm && Rm->rows*Rm->cols > 0) ? Rm->data[0] : 0.0;
    if (!X || X->rows*X->cols == 0) { matlab_mat *o = mat_alloc(1,1); o->data[0] = running; return o; }
    int64_t nel = X->rows * X->cols;
    for (int64_t i = 0; i < nel; ++i) {
        double a = std::fabs(X->data[i]);
        if (a > running) running = a;
    }
    matlab_mat *o = mat_alloc(1, 1);
    o->data[0] = running;
    return o;
}

}  // extern "C"

/* ===== T1.8 — image-data plumbing ======================================= *
 * imageDatastore('folder','LabelSource','foldernames') walks `folder`, finds
 * image files in each immediate subdirectory, and treats the subdirectory
 * name as the label.  countEachLabel returns per-label counts (sorted by
 * label name for determinism); splitEachLabel keeps the first `p*count`
 * entries of each label group (deterministic since entries are sorted).
 *
 * State is a single thread-local datastore — multiple imageDatastore()
 * calls reset it.  This matches the project's "one global at a time"
 * pattern (cf. ident's lsqnonlin ctx).
 *
 * mkdir(path) is a sibling helper so tests can synthesise a class-folder
 * layout via imwrite — no external setup scripts. */
namespace dlnet_imds {
struct Entry { std::string path; std::string label; };
static std::vector<Entry> g_entries;
static std::vector<Entry> g_kept;       /* current view (post-split) */
static std::vector<std::string> g_labels;
}  /* namespace dlnet_imds */

namespace {
struct dlnet_string_s { char *data; int64_t len; };
}

extern "C" {

matlab_mat *matlab_dlnet_mkdir(void *path_s) {
    auto *p = reinterpret_cast<const dlnet_string_s *>(path_s);
    matlab_mat *out = mat_alloc(0, 0);
    if (!p || !p->data || p->len <= 0) return out;
    std::string path(p->data, p->data + p->len);
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return out;
}

matlab_mat *matlab_dlnet_imds_load(void *folder_s) {
    using namespace dlnet_imds;
    g_entries.clear();
    g_kept.clear();
    g_labels.clear();
    auto *p = reinterpret_cast<const dlnet_string_s *>(folder_s);
    matlab_mat *handle = mat_alloc(1, 1);
    handle->data[0] = 0.0;
    if (!p || !p->data || p->len <= 0) return handle;
    namespace fs = std::filesystem;
    std::string root(p->data, p->data + p->len);
    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) return handle;
    std::set<std::string> labelSet;
    for (auto &sub : fs::directory_iterator(root, ec)) {
        if (!sub.is_directory(ec)) continue;
        std::string label = sub.path().filename().string();
        labelSet.insert(label);
        for (auto &f : fs::directory_iterator(sub.path(), ec)) {
            if (!f.is_regular_file(ec)) continue;
            std::string ext = f.path().extension().string();
            for (auto &c : ext) c = static_cast<char>(std::tolower(c));
            if (ext == ".pgm" || ext == ".ppm" || ext == ".bmp" ||
                ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                g_entries.push_back({f.path().string(), label});
            }
        }
    }
    g_labels.assign(labelSet.begin(), labelSet.end());
    std::sort(g_labels.begin(), g_labels.end());
    std::sort(g_entries.begin(), g_entries.end(),
              [](const Entry &a, const Entry &b) {
                  if (a.label != b.label) return a.label < b.label;
                  return a.path < b.path;
              });
    g_kept = g_entries;
    handle->data[0] = 1.0;
    return handle;
}

matlab_mat *matlab_dlnet_imds_count(matlab_mat *) {
    using namespace dlnet_imds;
    int64_t N = static_cast<int64_t>(g_labels.size());
    matlab_mat *out = mat_alloc(N, 1);
    if (N == 0) return out;
    std::map<std::string, int64_t> cnt;
    for (auto &e : g_kept) cnt[e.label]++;
    for (int64_t i = 0; i < N; ++i) {
        auto it = cnt.find(g_labels[static_cast<size_t>(i)]);
        out->data[i] = (it != cnt.end()) ? static_cast<double>(it->second) : 0.0;
    }
    return out;
}

double matlab_dlnet_imds_numfiles(matlab_mat *) {
    using namespace dlnet_imds;
    return static_cast<double>(g_kept.size());
}

/* T3.4b — augmentedImageDatastore's per-batch transform.
 * Apply ONE random rotate (uniform in [-ang_max, ang_max] deg) + scale
 * (uniform in [scale_min, scale_max]) + translation (uniform in
 * [-tx_max, tx_max] x [-ty_max, ty_max] px) to the input.  Output is
 * resized back to the original image size so it's drop-in for the
 * input layer.
 *
 * Reuses runtime_images.cpp's imrotate / imresize / imtranslate.  The
 * randomness pulls from matlab_rand() so seeds are reproducible.  Input
 * I can be M×N grayscale or M×N×3 RGB; we operate per slice. */
extern matlab_mat *matlab_image_imrotate(matlab_mat *, matlab_mat *, void *, void *);
extern matlab_mat *matlab_image_imresize(matlab_mat *, matlab_mat *, void *);
extern matlab_mat *matlab_image_imtranslate(matlab_mat *, matlab_mat *);
extern "C" matlab_mat *matlab_rand(double, double);

matlab_mat *matlab_dlnet_augment_image(matlab_mat *I,
                                       double ang_max,
                                       double scale_min, double scale_max,
                                       double tx_max, double ty_max) {
    if (!I) return mat_alloc(0, 0);
    int64_t H = I->rows, W = I->cols;

    /* Sample three uniform random scalars. */
    matlab_mat *u = matlab_rand(3, 1);
    double r0 = u->data[0], r1 = u->data[1], r2 = u->data[2];
    double r3 = 0.0, r4 = 0.0;
    matlab_mat *u2 = matlab_rand(2, 1);
    r3 = u2->data[0]; r4 = u2->data[1];

    double ang = (2.0 * r0 - 1.0) * ang_max;                     /* [-ang_max, ang_max] */
    double scl = scale_min + r1 * (scale_max - scale_min);       /* [smin, smax] */
    double tx  = (2.0 * r2 - 1.0) * tx_max;
    double ty  = (2.0 * r3 - 1.0) * ty_max;
    (void)r4;

    /* 1. Rotate (crop bbox so size is preserved). */
    matlab_mat *am = mat_alloc(1, 1); am->data[0] = ang;
    matlab_mat *Ir = matlab_image_imrotate(I, am, nullptr, nullptr);

    /* 2. Scale (uniform).  imresize takes a scalar scale. */
    matlab_mat *sm = mat_alloc(1, 1); sm->data[0] = scl;
    matlab_mat *Is = matlab_image_imresize(Ir, sm, nullptr);

    /* 3. Resize back to the original (H, W) so the augmented sample is
     * drop-in for the input layer. */
    matlab_mat *sz = mat_alloc(2, 1);
    sz->data[0] = static_cast<double>(H);
    sz->data[1] = static_cast<double>(W);
    matlab_mat *Ifit = matlab_image_imresize(Is, sz, nullptr);

    /* 4. Translate. */
    matlab_mat *tm = mat_alloc(2, 1);
    tm->data[0] = tx; tm->data[1] = ty;
    matlab_mat *Ot = matlab_image_imtranslate(Ifit, tm);
    return Ot;
}

matlab_mat *matlab_dlnet_imds_split(matlab_mat *, double p) {
    using namespace dlnet_imds;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    std::map<std::string, std::vector<const Entry *>> by_label;
    for (auto &e : g_entries) by_label[e.label].push_back(&e);
    std::vector<Entry> kept_new;
    for (auto &kv : by_label) {
        int64_t total = static_cast<int64_t>(kv.second.size());
        int64_t k = static_cast<int64_t>(std::floor(p * static_cast<double>(total)));
        if (k < 1 && p > 0.0) k = 1;
        for (int64_t i = 0; i < k; ++i) kept_new.push_back(*kv.second[static_cast<size_t>(i)]);
    }
    g_kept = std::move(kept_new);
    matlab_mat *out = mat_alloc(1, 1);
    out->data[0] = static_cast<double>(g_kept.size());
    return out;
}

}  // extern "C"

/* =====================================================================
 * C: dlnetwork carrier — sequential layer-list driver
 *
 * MATLAB's `dlnetwork` is a classdef object that wraps an array of
 * layer objects (`imageInputLayer`, `fullyConnectedLayer`, etc.).
 * The "true" object-array surface is gated on classdef array literals
 * (a Sema feature, multi-week work).
 *
 * Pragmatic unlock: same user-facing pattern via a runtime-resident
 * sequential carrier.  The handle is a 1x1 mat whose data[0] is the
 * net index in g_nets.  Each net stores a vector of layer descriptors;
 * each layer is one of:
 *   - FC(W, b)            -- fully-connected: y = W*x + b
 *   - Relu / Sigmoid / Tanh / Softmax  -- elementwise
 *
 * forward(net, X) chains them.  train(net, X, Y_target, lr, n_iter)
 * trains end-to-end with Adam over the FC weights.  This is the
 * "dlnetwork + trainnet" carve-down unlock, minus the classdef array
 * literal syntax.
 * =================================================================== */
namespace dlnet_net {

enum LayerKind { L_FC = 1, L_RELU = 2, L_SIGMOID = 3, L_TANH = 4, L_SOFTMAX = 5 };

struct Layer {
    int kind;
    matlab_mat *W = nullptr;   /* FC: kxin */
    matlab_mat *b = nullptr;   /* FC: kx1  (bias column) */
    /* Adam state — owned by the layer, lazily initialised on first train. */
    matlab_mat *mW = nullptr;
    matlab_mat *vW = nullptr;
    matlab_mat *mb = nullptr;
    matlab_mat *vb = nullptr;
};

struct Net {
    std::vector<Layer> layers;
};

static std::vector<Net> g_nets;  /* indexed by handle.data[0] - 1 */

static int handle_to_idx(matlab_mat *h) {
    if (!h || h->rows * h->cols < 1) return -1;
    int i = static_cast<int>(h->data[0]) - 1;
    if (i < 0 || i >= static_cast<int>(g_nets.size())) return -1;
    return i;
}

static matlab_mat *mat_clone_d(const matlab_mat *src) {
    if (!src) return mat_alloc(0, 0);
    matlab_mat *o = mat_alloc(src->rows, src->cols);
    int64_t n = src->rows * src->cols;
    for (int64_t i = 0; i < n; ++i) o->data[i] = src->data[i];
    return o;
}

}  /* namespace dlnet_net */

extern "C" {

matlab_mat *matlab_dlnet_net_new(void) {
    dlnet_net::g_nets.emplace_back();
    matlab_mat *h = mat_alloc(1, 1);
    h->data[0] = static_cast<double>(dlnet_net::g_nets.size());
    return h;
}

matlab_mat *matlab_dlnet_net_add_fc(matlab_mat *h, matlab_mat *W, matlab_mat *b) {
    using namespace dlnet_net;
    int i = handle_to_idx(h);
    if (i < 0 || !W || !b) return h;
    Layer L;
    L.kind = L_FC;
    L.W = mat_clone_d(W);
    L.b = mat_clone_d(b);
    g_nets[static_cast<size_t>(i)].layers.push_back(std::move(L));
    return h;
}
matlab_mat *matlab_dlnet_net_add_relu   (matlab_mat *h) { using namespace dlnet_net; int i = handle_to_idx(h); if (i >= 0) g_nets[static_cast<size_t>(i)].layers.push_back({L_RELU}); return h; }
matlab_mat *matlab_dlnet_net_add_sigmoid(matlab_mat *h) { using namespace dlnet_net; int i = handle_to_idx(h); if (i >= 0) g_nets[static_cast<size_t>(i)].layers.push_back({L_SIGMOID}); return h; }
matlab_mat *matlab_dlnet_net_add_tanh   (matlab_mat *h) { using namespace dlnet_net; int i = handle_to_idx(h); if (i >= 0) g_nets[static_cast<size_t>(i)].layers.push_back({L_TANH}); return h; }
matlab_mat *matlab_dlnet_net_add_softmax(matlab_mat *h) { using namespace dlnet_net; int i = handle_to_idx(h); if (i >= 0) g_nets[static_cast<size_t>(i)].layers.push_back({L_SOFTMAX}); return h; }

double matlab_dlnet_net_num_layers(matlab_mat *h) {
    using namespace dlnet_net;
    int i = handle_to_idx(h);
    if (i < 0) return 0.0;
    return static_cast<double>(g_nets[static_cast<size_t>(i)].layers.size());
}

/* Apply a single layer: y_out = layer(x_in).  Allocates a fresh y. */
static matlab_mat *apply_layer(const dlnet_net::Layer &L, matlab_mat *X) {
    using namespace dlnet_net;
    if (L.kind == L_FC) {
        /* Y = W * X + b  (broadcast bias across columns). */
        matlab_mat *WX = matlab_matmul_mm(L.W, X);
        int64_t r = WX->rows, c = WX->cols;
        for (int64_t i = 0; i < r; ++i)
            for (int64_t j = 0; j < c; ++j) WX->data[i * c + j] += L.b->data[i];
        return WX;
    }
    int64_t r = X->rows, c = X->cols;
    matlab_mat *Y = mat_alloc(r, c);
    int64_t n = r * c;
    if (L.kind == L_RELU)    { for (int64_t i = 0; i < n; ++i) Y->data[i] = X->data[i] > 0 ? X->data[i] : 0; }
    else if (L.kind == L_SIGMOID) { for (int64_t i = 0; i < n; ++i) Y->data[i] = 1.0 / (1.0 + std::exp(-X->data[i])); }
    else if (L.kind == L_TANH)    { for (int64_t i = 0; i < n; ++i) Y->data[i] = std::tanh(X->data[i]); }
    else if (L.kind == L_SOFTMAX) {
        for (int64_t j = 0; j < c; ++j) {
            double m = X->data[j];
            for (int64_t i = 1; i < r; ++i) if (X->data[i * c + j] > m) m = X->data[i * c + j];
            double s = 0; for (int64_t i = 0; i < r; ++i) { double e = std::exp(X->data[i * c + j] - m); Y->data[i * c + j] = e; s += e; }
            for (int64_t i = 0; i < r; ++i) Y->data[i * c + j] /= s;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) Y->data[i] = X->data[i];
    }
    return Y;
}

matlab_mat *matlab_dlnet_net_predict(matlab_mat *h, matlab_mat *X) {
    using namespace dlnet_net;
    int idx = handle_to_idx(h);
    if (idx < 0 || !X) return mat_alloc(0, 0);
    matlab_mat *cur = mat_clone_d(X);
    for (auto &L : g_nets[static_cast<size_t>(idx)].layers) {
        matlab_mat *nxt = apply_layer(L, cur);
        cur = nxt;
    }
    return cur;
}

/* train(handle, X, Y, lr, n_iter, b1=0.9, b2=0.999, eps=1e-8) -> final MSE.
 * Forward chains apply_layer; backward walks layers in reverse using the
 * stashed forward activations + analytic per-op gradient.  Adam per FC layer. */
double matlab_dlnet_net_train(matlab_mat *h, matlab_mat *X, matlab_mat *Y,
                               double lr, double n_iter) {
    using namespace dlnet_net;
    int idx = handle_to_idx(h);
    if (idx < 0 || !X || !Y) return 0.0;
    Net &net = g_nets[static_cast<size_t>(idx)];
    /* Initialise Adam state lazily. */
    for (auto &L : net.layers) {
        if (L.kind == L_FC && !L.mW) {
            int64_t kr = L.W->rows, kc = L.W->cols;
            L.mW = mat_alloc(kr, kc); for (int64_t i = 0; i < kr*kc; ++i) L.mW->data[i] = 0.0;
            L.vW = mat_alloc(kr, kc); for (int64_t i = 0; i < kr*kc; ++i) L.vW->data[i] = 0.0;
            int64_t br = L.b->rows, bc = L.b->cols;
            L.mb = mat_alloc(br, bc); for (int64_t i = 0; i < br*bc; ++i) L.mb->data[i] = 0.0;
            L.vb = mat_alloc(br, bc); for (int64_t i = 0; i < br*bc; ++i) L.vb->data[i] = 0.0;
        }
    }
    double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    double final_loss = 0.0;
    int64_t N = static_cast<int64_t>(n_iter);
    if (N < 1) N = 1;
    int64_t Xc = X->cols;
    for (int64_t t = 1; t <= N; ++t) {
        /* Forward — stash activations per layer. */
        std::vector<matlab_mat *> acts;
        acts.push_back(mat_clone_d(X));
        for (auto &L : net.layers) {
            matlab_mat *nxt = apply_layer(L, acts.back());
            acts.push_back(nxt);
        }
        matlab_mat *yhat = acts.back();
        /* MSE loss: 0.5 * sum((yhat - Y)^2) / Xc. */
        int64_t nE = yhat->rows * yhat->cols;
        double loss = 0;
        matlab_mat *grad = mat_alloc(yhat->rows, yhat->cols);
        for (int64_t i = 0; i < nE; ++i) {
            double d = yhat->data[i] - Y->data[i];
            loss += 0.5 * d * d;
            grad->data[i] = d / static_cast<double>(Xc);
        }
        final_loss = loss / static_cast<double>(Xc);
        /* Backward + per-FC-layer Adam. */
        for (int64_t li = static_cast<int64_t>(net.layers.size()) - 1; li >= 0; --li) {
            Layer &L = net.layers[static_cast<size_t>(li)];
            matlab_mat *x_in  = acts[static_cast<size_t>(li)];
            matlab_mat *y_out = acts[static_cast<size_t>(li + 1)];
            matlab_mat *next_grad = nullptr;
            if (L.kind == L_FC) {
                /* dW = grad * x_in^T;  db = sum(grad, dim=2);
                 * d_x = W^T * grad. */
                int64_t k = L.W->rows, in_n = L.W->cols, B = grad->cols;
                matlab_mat *dW = mat_alloc(k, in_n);
                for (int64_t i = 0; i < k; ++i)
                    for (int64_t j = 0; j < in_n; ++j) {
                        double s = 0;
                        for (int64_t r = 0; r < B; ++r) s += grad->data[i * B + r] * x_in->data[j * B + r];
                        dW->data[i * in_n + j] = s;
                    }
                matlab_mat *db = mat_alloc(L.b->rows, L.b->cols);
                for (int64_t i = 0; i < k; ++i) {
                    double s = 0;
                    for (int64_t r = 0; r < B; ++r) s += grad->data[i * B + r];
                    db->data[i] = s;
                }
                /* d_x = W^T * grad. */
                matlab_mat *Wt = mat_alloc(in_n, k);
                for (int64_t i = 0; i < k; ++i) for (int64_t j = 0; j < in_n; ++j) Wt->data[j * k + i] = L.W->data[i * in_n + j];
                next_grad = matlab_matmul_mm(Wt, grad);
                /* Adam on W. */
                double bc1 = 1.0 - std::pow(b1, static_cast<double>(t));
                double bc2 = 1.0 - std::pow(b2, static_cast<double>(t));
                for (int64_t i = 0; i < k * in_n; ++i) {
                    L.mW->data[i] = b1 * L.mW->data[i] + (1.0 - b1) * dW->data[i];
                    L.vW->data[i] = b2 * L.vW->data[i] + (1.0 - b2) * dW->data[i] * dW->data[i];
                    double mhat = L.mW->data[i] / bc1;
                    double vhat = L.vW->data[i] / bc2;
                    L.W->data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
                }
                for (int64_t i = 0; i < L.b->rows * L.b->cols; ++i) {
                    L.mb->data[i] = b1 * L.mb->data[i] + (1.0 - b1) * db->data[i];
                    L.vb->data[i] = b2 * L.vb->data[i] + (1.0 - b2) * db->data[i] * db->data[i];
                    double mhat = L.mb->data[i] / bc1;
                    double vhat = L.vb->data[i] / bc2;
                    L.b->data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
                }
            } else if (L.kind == L_RELU) {
                next_grad = mat_alloc(grad->rows, grad->cols);
                int64_t nn = grad->rows * grad->cols;
                for (int64_t i = 0; i < nn; ++i) next_grad->data[i] = (x_in->data[i] > 0) ? grad->data[i] : 0;
            } else if (L.kind == L_SIGMOID) {
                next_grad = mat_alloc(grad->rows, grad->cols);
                int64_t nn = grad->rows * grad->cols;
                for (int64_t i = 0; i < nn; ++i) {
                    double y = y_out->data[i];
                    next_grad->data[i] = grad->data[i] * y * (1.0 - y);
                }
            } else if (L.kind == L_TANH) {
                next_grad = mat_alloc(grad->rows, grad->cols);
                int64_t nn = grad->rows * grad->cols;
                for (int64_t i = 0; i < nn; ++i) {
                    double y = y_out->data[i];
                    next_grad->data[i] = grad->data[i] * (1.0 - y * y);
                }
            } else if (L.kind == L_SOFTMAX) {
                /* dx = y * (dy - sum(dy*y, dim=2_per_col)) */
                next_grad = mat_alloc(grad->rows, grad->cols);
                int64_t R = grad->rows, C = grad->cols;
                for (int64_t c = 0; c < C; ++c) {
                    double dot = 0; for (int64_t r = 0; r < R; ++r) dot += grad->data[r * C + c] * y_out->data[r * C + c];
                    for (int64_t r = 0; r < R; ++r) next_grad->data[r * C + c] = y_out->data[r * C + c] * (grad->data[r * C + c] - dot);
                }
            } else {
                next_grad = mat_clone_d(grad);
            }
            grad = next_grad;
        }
    }
    return final_loss;
}

}  // extern "C"
