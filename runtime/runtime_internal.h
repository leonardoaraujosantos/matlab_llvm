/* Internal header — shared between the runtime translation units.
 *
 * Exposes the private type layouts (matlab_mat, matlab_mat_c, matlab_mat3),
 * shared allocator helpers (mat_alloc et al.) and the global stdout mutex
 * across the .cpp files that result from the Phase-2 file split. Public
 * ABI clients still see only the opaque types in matlab_runtime.h — this
 * header is install-time NOT publicly distributed.
 *
 * The single extern "C" block opens once in this header so every TU that
 * #includes it sees the shared declarations with C linkage. Each runtime
 * .cpp adds its own extern "C" wrapper around its definitions; the two
 * are equivalent at the symbol-table level.
 */

#ifndef MATLAB_RUNTIME_INTERNAL_H
#define MATLAB_RUNTIME_INTERNAL_H

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Deliberately *not* #include "matlab_runtime.h" — the runtime TUs
 * use polymorphic-on-void* signatures for several macros (BINARY_MM,
 * MAT_C_BINARY, ...) that conflict with the public header's typed
 * declarations. The public-facing types are declared opaque-style
 * here for the runtime's internal use; downstream code that links
 * against the runtime gets the full ABI from matlab_runtime.h. */

typedef struct matlab_mat      matlab_mat;
typedef struct matlab_mat_c    matlab_mat_c;
typedef struct matlab_struct_s matlab_struct;

#ifdef __cplusplus
extern "C" {
#endif

/*--- Tagged-descriptor magic constants ------------------------------------
 * matlab_mat (real) is untagged — its first 8 bytes are a heap data pointer
 * whose low 32 bits are vanishingly unlikely to collide with a magic word.
 * matlab_mat_c and matlab_mat3 carry the magic at offset 0 so polymorphic
 * entries (matlab_disp_mat, matlab_conj_c, matlab_fft_c, ...) can sniff
 * the descriptor kind without a separate id field. */
#define MATLAB_MAT_C_MAGIC 0xC0FFEE01u
#define MATLAB_MAT3_MAGIC  0xC0FFEE03u

/*--- Real matrix descriptor ----------------------------------------------*/
struct matlab_mat {
    double *data;      /* row-major, rows*cols doubles (calloc'd) */
    int64_t rows;
    int64_t cols;
};

/*--- Complex matrix descriptor ------------------------------------------- */
struct matlab_mat_c {
    uint32_t magic;    /* MATLAB_MAT_C_MAGIC */
    uint32_t _pad;     /* keep re/im 8-byte aligned */
    double *re;        /* row-major, rows*cols doubles */
    double *im;        /* row-major, rows*cols doubles */
    int64_t rows;
    int64_t cols;
};

/*--- 3-D matrix descriptor ------------------------------------------------ */
struct matlab_mat3 {
    uint32_t magic;    /* MATLAB_MAT3_MAGIC */
    uint32_t _pad;
    double *data;      /* contiguous rows*cols*depth doubles, row-major */
    int64_t rows;
    int64_t cols;
    int64_t depth;
};
typedef struct matlab_mat3 matlab_mat3;

/*--- Magic-tag predicates -------------------------------------------------
 * Inline so they don't impose a TU boundary cost. Both accept a raw
 * void* — callers pass either descriptor kind interchangeably. */
static inline int mat_is_complex(const void *p) {
    if (!p) return 0;
    return *(const uint32_t *)p == MATLAB_MAT_C_MAGIC;
}
static inline int mat_is_3d(const void *p) {
    if (!p) return 0;
    return *(const uint32_t *)p == MATLAB_MAT3_MAGIC;
}

/*--- Allocator helpers ---------------------------------------------------
 * Defined once in runtime/matlab_runtime.cpp; declared here for the other
 * runtime TUs (runtime_debug.cpp etc.). Negative dims are clamped to 0 to
 * keep the runtime defensive against arithmetic underflow in the lowering. */
matlab_mat   *mat_alloc  (int64_t m, int64_t n);
matlab_mat_c *mat_c_alloc(int64_t m, int64_t n);
matlab_mat3  *mat3_alloc (int64_t m, int64_t n, int64_t p);

/*--- Shared global I/O mutex --------------------------------------------- */
extern pthread_mutex_t matlab_io_mutex;

/*--- Struct descriptor + helpers ----------------------------------------- *
 * Used by the workspace mirror (runtime_debug.cpp) to build a typed
 * variable view for the DAP server. The real-side runtime
 * (runtime/matlab_runtime.cpp) defines the helpers; this header makes
 * them visible across the split. */
struct matlab_struct_s {
    int32_t nfields;
    int32_t capacity;
    char **names;
    int32_t *kinds;       /* 0 = f64, 1 = matrix ptr, 2 = nested struct */
    double *f64_vals;
    void **ptr_vals;
};

int32_t struct_find_field(matlab_struct *s, const char *name, int32_t len);
int32_t struct_reserve   (matlab_struct *s, const char *name, int32_t len);

/*--- Error-message buffer ------------------------------------------------ *
 * `matlab_set_error_msg` writes to these; the DAP error-traceback path
 * reads them. Defined in matlab_runtime.cpp. */
extern char    matlab_error_msg[1024];
extern int64_t matlab_error_msg_len;
extern int32_t matlab_error_flag;

/*--- Class-tagged object descriptor -------------------------------------- *
 * matlab_obj is a struct-compatible record with a class_id appended;
 * the leading fields MUST match matlab_struct_s exactly so polymorphic
 * struct-getter helpers work on both. Used by the DAP for class
 * instance display. */
struct matlab_obj_s {
    /* matlab_struct prefix — MUST MATCH matlab_struct_s exactly. */
    int32_t nfields;
    int32_t capacity;
    char **names;
    int32_t *kinds;
    double *f64_vals;
    void **ptr_vals;
    int32_t class_id;
};
typedef struct matlab_obj_s matlab_obj;

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus

/*=========================================================================
 * Phase 4 — RAII helpers for the runtime's internal scratch state.
 *
 * Public ABI clients still see the opaque types declared above; these
 * smart-pointer aliases are TU-internal only. Their job is to take the
 * 178 manual malloc/free pairs in matlab_runtime.cpp down to ≤30, with
 * leak-by-construction-impossible guarantees on the routines that get
 * migrated.
 *
 * Pattern at public entry points:
 *
 *     extern "C" matlab_mat *matlab_inv(matlab_mat *A) {
 *         if (!A || A->rows != A->cols) return mat_alloc(0, 0);
 *         std::vector<double> LU(...), rhs(...), col(...);
 *         std::vector<int64_t> piv(...);
 *         matlab::runtime::MatPtr work(mat_alloc(n, n));
 *         // ... fill work->data ...
 *         return work.release();   // hand the ptr back over the C ABI
 *     }
 *
 * Internal helpers can hold MatPtr / MatCPtr through their lifetime
 * and simply let the destructor fire on early returns / exceptions.
 *=========================================================================*/
#include <memory>

/* Verbose form (instead of `namespace matlab::runtime`) so the runtime
 * still compiles when downstream consumers force a pre-C++17 standard
 * (e.g. the strict-mode emit-c test corpus). */
namespace matlab { namespace runtime {

/* mat_alloc allocates a descriptor *and* its data buffer; both must be
 * freed together. The runtime today leaks aggressively (programs are
 * assumed short-lived), but RAII removes the per-call leaks on hot
 * scratch matrices that get used and discarded inside one entry. */
struct MatDeleter {
    void operator()(matlab_mat *p) const noexcept {
        if (!p) return;
        free(p->data);
        free(p);
    }
};
using MatPtr = std::unique_ptr<matlab_mat, MatDeleter>;

struct MatCDeleter {
    void operator()(matlab_mat_c *p) const noexcept {
        if (!p) return;
        free(p->re);
        free(p->im);
        free(p);
    }
};
using MatCPtr = std::unique_ptr<matlab_mat_c, MatCDeleter>;

inline MatPtr  make_mat   (int64_t m, int64_t n) { return MatPtr (mat_alloc  (m, n)); }
inline MatCPtr make_mat_c (int64_t m, int64_t n) { return MatCPtr(mat_c_alloc(m, n)); }

/*--- Phase 5: shape-op template -----------------------------------------
 *
 * Collapses the "alloc m_out x n_out, double for-loop, one differing
 * index expression, return" pattern shared by transpose / fliplr /
 * flipud / flip / rot90 / diag (vector path) / repmat / permute /
 * squeeze into a one-liner. Each op provides only the per-cell index
 * expression as a lambda:
 *
 *     matlab_mat *matlab_fliplr(matlab_mat *A) {
 *         if (!A) return mat_alloc(0, 0);
 *         int64_t m = A->rows, n = A->cols;
 *         return matlab::runtime::shape_op(m, n, [&](int64_t i, int64_t j) {
 *             return A->data[i * n + (n - 1 - j)];
 *         }).release();
 *     }
 *
 * The lambda runs inside shape_op only — no escape — so [&] capture
 * of the input matrix and its dimensions is safe and zero-cost. */
template <class IndexFn>
inline MatPtr shape_op(int64_t m_out, int64_t n_out, IndexFn idx) {
    MatPtr R = make_mat(m_out, n_out);
    for (int64_t i = 0; i < m_out; ++i)
        for (int64_t j = 0; j < n_out; ++j)
            R->data[i * n_out + j] = idx(i, j);
    return R;
}

} }  /* namespace matlab::runtime */

#endif /* __cplusplus */

#endif /* MATLAB_RUNTIME_INTERNAL_H */
