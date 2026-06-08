/* runtime_debug.cpp — REPL workspace + DAP hook infrastructure.
 *
 * Extracted from runtime/matlab_runtime.cpp in Phase 2 of the runtime
 * port (docs/port_runtime_2_cpp.md). The body is byte-identical to the
 * original block at lines ~4149-6965; only the surrounding wrappers are
 * new — `runtime_internal.h` exposes the shared layouts and statics
 * (matlab_io_mutex, mat_alloc, mat_c_alloc) that this TU needs.
 *
 * Forward declarations match the post-debug section in the main TU so
 * matlab_struct_rmfield, matlab_disp_obj, matlab_dbg_class_name, and
 * matlab_struct_get_child_struct can be called from inside the debug
 * machinery without pulling in the public header (which has typed
 * signatures that conflict with several macros in matlab_runtime.cpp).
 */

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "runtime_internal.h"

#include <set>
#include <string>

/* Forward declarations for symbols defined elsewhere in the runtime
 * but called from inside the debug machinery. Mirrors the names the
 * public matlab_runtime.h header would otherwise provide. */
extern "C" {
matlab_struct *matlab_struct_new(void);
matlab_struct *matlab_struct_rmfield(matlab_struct *s, const char *name,
                                      int64_t len);
matlab_struct *matlab_struct_get_child_struct(matlab_struct *s,
                                              const char *name, int64_t len);
const char    *matlab_dbg_class_name(int32_t class_id, int64_t *len_out);
double         matlab_struct_has_field(matlab_struct *s,
                                       const char *name, int64_t len);
void           matlab_struct_set_f64(matlab_struct *s, const char *name,
                                      int64_t len, double v);
void           matlab_struct_set_mat(matlab_struct *s, const char *name,
                                      int64_t len, matlab_mat *m);
double         matlab_struct_get_f64(matlab_struct *s,
                                     const char *name, int64_t len);
matlab_mat    *matlab_struct_get_mat(matlab_struct *s,
                                     const char *name, int64_t len);
void           matlab_disp_obj(matlab_obj *o);
void           matlab_disp_mat(void *Aptr);
void           matlab_set_error_msg(const char *msg, int64_t n);
/* Phase 1.1.F: typed-int descriptor kind lookup (defined in
 * matlab_runtime.cpp). Returns -1 / 0 (u8) / 1 (i32). */
int            matlab_mat_intlane_kind(const void *p);
/* matlab_string accessors. The struct layout lives in matlab_runtime.cpp;
 * the runtime_debug TU only ever holds the pointer and reads bytes
 * through these helpers so the layout stays opaque across the file split. */
const char    *matlab_string_get_data(void *s, int64_t *len_out);
int64_t        matlab_string_get_len (void *s);

/* -------- REPL workspace -------------------------------------------------
 *
 * A single matlab_struct* holds every variable the user has assigned
 * across REPL inputs. Each JIT-compiled input uses matlab_ws_get_* /
 * matlab_ws_set_* in place of local slots so state persists across
 * invocations. Field names are the user-visible variable names.
 *
 * The runtime lazily allocates the workspace on first touch, which
 * means the normal AOT path (matlabc -emit-llvm / -emit-c / ...)
 * never pays for it — these symbols only get linked into a program
 * when the compiler emits references to them, which today only the
 * REPL mode does.
 *--------------------------------------------------------------------------*/

static matlab_struct *matlab_ws = NULL;

static void matlab_ws_init_if_needed(void) {
    if (!matlab_ws) matlab_ws = matlab_struct_new();
}

/* Forward-declared up here so the matlab_ws_get_* / matlab_ws_set_*
 * call sites below compile. The bodies live alongside the rest of
 * the dbg machinery further down where matlab_dbg state is in
 * scope. matlab_ws_lock / _unlock take/release matlab_dbg.mu —
 * sharing the dbg mutex with the workspace serialization avoids a
 * second mutex and keeps the lock-order graph trivial (the
 * undo-rewind path in matlab_dbg_step_back already mutates ws
 * while holding matlab_dbg.mu, so it gets the same protection
 * for free). */
static void matlab_ws_check_watch(const char *name, int64_t len);
static void matlab_ws_check_read_watch(const char *name, int64_t len);
static void matlab_ws_lock(void);
static void matlab_ws_unlock(void);

double matlab_ws_get_f64(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    double v = matlab_struct_get_f64(matlab_ws, name, len);
    matlab_ws_unlock();
    /* Read watchpoint check. Fast path: when n_wp is 0 (the
     * common case — no read-watches active) the entire body is a
     * single mutex-free load + compare and the JIT's REPL-mode
     * read sites pay no measurable cost. The full check fires
     * only once a read-watch is armed. */
    matlab_ws_check_read_watch(name, len);
    return v;
}

/* Forward decl for the locked undo helper. The matlab_ws_set_*
 * sites take the workspace lock around BOTH the undo capture and
 * the matlab_struct mutation — concurrent parfor workers used to
 * race here, with two threads in struct_grow_if_needed at once
 * corrupting matlab_ws->names and a third thread's strlen on a
 * stale pointer crashing. The returned record gets its new_*
 * fields filled in after the write so the redo path can replay
 * forward through the log without re-running the JIT. */
struct matlab_dbg_undo_rec;
static struct matlab_dbg_undo_rec *
matlab_ws_push_undo_locked(const char *name, int64_t len,
                            int kind_being_written);
static void matlab_dbg_undo_record_set_new_f64(
    struct matlab_dbg_undo_rec *r, double v);
static void matlab_dbg_undo_record_set_new_ptr(
    struct matlab_dbg_undo_rec *r, int new_kind, void *p);

void matlab_ws_set_f64(const char *name, int64_t len, double v) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/0);
    matlab_struct_set_f64(matlab_ws, name, len, v);
    matlab_dbg_undo_record_set_new_f64(r, v);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

matlab_mat *matlab_ws_get_mat(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    matlab_mat *m = matlab_struct_get_mat(matlab_ws, name, len);
    matlab_ws_unlock();
    matlab_ws_check_read_watch(name, len);
    return m;
}

/* Reads a string variable from the script workspace. Mirrors the
 * read shape of matlab_ws_get_mat — returns a borrowed pointer (the
 * workspace owns the string descriptor; no caller free) or NULL when
 * the binding doesn't exist or holds a non-string kind. The lowering
 * routes a script-level NameExpr read through this entry whenever
 * StringBindings (or Sema's StringArrayType) tagged the variable, so
 * `t` alone or `disp(t)` see the matlab_string* the assign stamped
 * with kind=3 instead of an empty 0x0 matlab_mat. */
void *matlab_ws_get_string(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    void *out = NULL;
    if (matlab_ws) {
        for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
            if (matlab_ws->kinds[i] != 3) continue;
            const char *fn = matlab_ws->names[i];
            if ((int64_t)strlen(fn) != len) continue;
            if (memcmp(fn, name, (size_t)len) != 0) continue;
            out = matlab_ws->ptr_vals[i];
            break;
        }
    }
    matlab_ws_unlock();
    matlab_ws_check_read_watch(name, len);
    return out;
}

void matlab_ws_set_mat(const char *name, int64_t len, matlab_mat *m) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    /* Phase 1.1.F: typed-int matrices (matlab_mat_u8 *, matlab_mat_i32 *)
     * arrive here through the same matlab_ws_set_mat entry the f64 lane
     * uses — the Lowering doesn't keep separate workspace setters per
     * dtype. Consult the intlane registry to recover the actual lane and
     * tag the workspace slot with a sub-kind (4 = u8 mat, 5 = i32 mat).
     * Resolver.cpp reads this kind on cross-REPL-input lookups and
     * stamps InferredType=Array(UInt8/Int32, ...), letting the next
     * input's BinaryOp emission pick the typed runtime entry points
     * instead of running f64 arith over typed-int storage. */
    int kind = 1;
    int intlane = matlab_mat_intlane_kind(m);
    if (intlane == 0) kind = 4;        /* matlab_mat_u8 */
    else if (intlane == 1) kind = 5;   /* matlab_mat_i32 */
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, kind);
    matlab_struct_set_mat(matlab_ws, name, len, m);
    /* matlab_struct_set_mat normalises kinds[idx] to 1 (mat); restore
     * the lane-aware kind for the typed-int case. */
    if (kind != 1) {
        int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
        matlab_ws->kinds[idx] = (uint8_t)kind;
    }
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/kind, m);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Plain struct (matlab_struct*) assignment to the workspace.
 * Stores the struct pointer with kind=12 so Resolver / Lowering can
 * tell it apart from a class instance (kind=2) and a real matrix
 * (kind=1) on the next REPL turn — field-access dispatch needs to
 * see the binding as struct, not matrix.  Layout-compatible with
 * matlab_obj* (same prefix), so the existing matlab_struct_get_*
 * helpers walk it correctly.
 *
 * Note: kinds 6/9/10/11 are reserved for table / categorical /
 * datetime / duration (see matlab_struct_get_mat); kind=12 is the
 * next free slot. */
void matlab_ws_set_struct(const char *name, int64_t len, matlab_struct *s) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/12);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 12;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = s;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/12, s);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Struct-array assignment to the script-level workspace (kind=14, #133).
 * `a(i).x = v` builds a matlab_struct_arr* in a local slot; without this
 * store the array is discarded at end of REPL turn and a later-turn
 * `a(i).x` reads an empty array. Mirrors matlab_ws_set_struct (kind=12);
 * the read side returns the pointer verbatim via matlab_struct_get_mat's
 * kind=14 pass-through. */
void matlab_ws_set_struct_arr(const char *name, int64_t len, void *arr) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/14);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 14;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = arr;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/14, arr);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Class-instance assignment to the script-level workspace. Stores
 * the obj pointer with kind=2 so matlab_dbg_ws_kind reports it as
 * an object — the DAP formatter then routes through the obj path
 * (`1x1 ClassName`, expandable into properties) instead of treating
 * the pointer as a matlab_mat * and reading garbage. */
void matlab_ws_set_obj(const char *name, int64_t len, matlab_obj *o) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 2;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = o;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/2, o);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* String assignment to the script-level workspace. Stores a
 * matlab_string* with kind=3 so the DAP formatter can render it as
 * `string` / "abc" instead of pointer-casting the descriptor to
 * matlab_mat* (which previously aliased matlab_string::data into
 * matlab_mat::data and matlab_string::len into matlab_mat::rows,
 * making `text = "Test"` show up as a 4 x <heap-garbage> double).
 *
 * The undo machinery's prev_kind / new_kind enum gains a new value
 * (3 = string); apply_ws_undo at the rewind/redo sites restores it
 * via the same struct_reserve + ptr-store shape used for kind=2.
 * The meta record's `kind` field stays in the kind==2 bucket since
 * the rewind dispatch keys off prev_kind, not the meta kind. */
void matlab_ws_set_string(const char *name, int64_t len, void *s) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 3;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = s;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/3, s);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Symbolic Math Toolbox — matlab_sym* assignment to the script-level
 * workspace. Stores with kind=7 so matlab_dbg_ws_kind / the Resolver
 * workspace-kind hook / the DAP variable formatter all recognise the
 * descriptor as a symbolic expression. The DAP renders the value via
 * matlab_dbg_sym_str (defined below), the Resolver stamps the binding
 * for cross-input REPL persistence. */
void matlab_ws_set_sym(const char *name, int64_t len, void *s) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 7;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = s;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/7, s);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* matlab_ws_get_sym(name) — reverse direction. Returns the stored
 * pointer (typed as void* so the C ABI in runtime_sym.h doesn't have
 * to leak SymBox-knowledge here). The caller (lowering's workspace
 * load path) re-types as matlab_sym* via the C ABI. */
void *matlab_ws_get_sym(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    /* Linear scan — workspace is small (REPL-scale). Mirrors the
     * matlab_dbg_ws_* iteration used by the DAP server. */
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *gn = matlab_ws->names[i];
        if (!gn) continue;
        size_t glen = strlen(gn);
        if (glen != (size_t)len) continue;
        if (memcmp(gn, name, (size_t)len) != 0) continue;
        if (matlab_ws->kinds[i] != 7) return nullptr;
        return matlab_ws->ptr_vals[i];
    }
    return nullptr;
}

/* Symbolic Math Toolbox — symbolic matrix workspace setter / getter
 * (kind=8). Mirrors matlab_ws_set_sym (kind=7) shape. The DAP variable
 * formatter renders symmat values via matlab_dbg_symmat_str. */
void matlab_ws_set_symmat(const char *name, int64_t len, void *m) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 8;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = m;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/8, m);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

void *matlab_ws_get_symmat(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *gn = matlab_ws->names[i];
        if (!gn) continue;
        size_t glen = strlen(gn);
        if (glen != (size_t)len) continue;
        if (memcmp(gn, name, (size_t)len) != 0) continue;
        if (matlab_ws->kinds[i] != 8) return nullptr;
        return matlab_ws->ptr_vals[i];
    }
    return nullptr;
}

/* Function-handle workspace ABI (kind=13).  A REPL/DAP variable holding
 * a function handle (`f = @sin`, `g = @myFn`, capture-free `@(x) x+1`)
 * stores a raw function pointer — the address the make_handle / anon
 * outliner resolved at the assignment site.  Storing it under a distinct
 * kind (rather than kind=1 matrix) is what lets the next REPL turn /
 * the whole-program DAP launch recover that the variable is *callable*:
 * Resolver::applyWorkspaceKind stamps Binding::IsHandle on a kind=13
 * lookup, and the MLIR lowering then routes `f(x)` through the
 * matlab_call_handle_s* trampolines below instead of mis-lowering the
 * pointer into a matrix subscript (which read the code pointer as a
 * matlab_mat* and crashed).
 *
 * Only the function pointer survives the round-trip — anonymous closures
 * with captured values can't be reconstructed from the pointer alone, so
 * the lowering only routes capture-free handles here.  The undo record
 * uses the same meta-kind=2 (ptr-in-slot) shape as obj/sym/string. */
void matlab_ws_set_handle(const char *name, int64_t len, void *fn) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 13;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = fn;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/13, fn);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* #119: function-handle call SIGNATURE side-channel.  The bare kind=13
 * pointer can't tell a scalar-returning anon (`@(x) x(1)+x(2)`,
 * double(*)(matlab_mat*)) from a matrix-returning one (`@(x) x*2`,
 * matlab_mat*(*)(matlab_mat*)) — needed to pick the right matrix-argument
 * trampoline / result type on a later turn.  We stash the return-kind in
 * the kind=13 field's otherwise-unused f64 slot: -1 unknown, 0 scalar,
 * 1 matrix.  set is a no-op if the name isn't a live kind=13 field (it is,
 * because the matlab_ws_set_handle store runs immediately before). */
/* VideoWriter variable-name registry (#236).  A REPL `v = VideoWriter(...)`
 * marks `v` here; a later submission's resolver kind-hook reports kind 15 for
 * a marked name so the binding is re-stamped IsVideoWriter and `v.FrameRate =
 * ...` routes to the dedicated setter instead of the struct-field path (which
 * reinterpreted the opaque handle as a struct and crashed the encoder).  The
 * mark is name-keyed and independent of how the handle value is stored, so it
 * works regardless of the value's workspace kind.  Reassigning the name to a
 * non-VideoWriter leaves a stale mark (a documented v1 limitation — the common
 * pattern keeps `v` a VideoWriter for its lifetime). */
static std::set<std::string> matlab_videowriter_names;

void matlab_ws_mark_videowriter(const char *name, int64_t len, int32_t on) {
    if (!name || len <= 0) return;
    matlab_ws_lock();
    std::string key(name, (size_t)len);
    if (on) matlab_videowriter_names.insert(key);
    else    matlab_videowriter_names.erase(key);
    matlab_ws_unlock();
}

int32_t matlab_ws_is_videowriter(const char *name, int64_t len) {
    if (!name || len <= 0) return 0;
    matlab_ws_lock();
    int32_t r =
        matlab_videowriter_names.count(std::string(name, (size_t)len)) ? 1 : 0;
    matlab_ws_unlock();
    return r;
}

void matlab_ws_set_handle_sig(const char *name, int64_t len, int32_t retkind) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *gn = matlab_ws->names[i];
        if (!gn || strlen(gn) != (size_t)len) continue;
        if (memcmp(gn, name, (size_t)len) != 0) continue;
        if (matlab_ws->kinds[i] == 13) matlab_ws->f64_vals[i] = (double)retkind;
        break;
    }
    matlab_ws_unlock();
}

int32_t matlab_ws_get_handle_sig(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    int32_t out = -1;
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *gn = matlab_ws->names[i];
        if (!gn || strlen(gn) != (size_t)len) continue;
        if (memcmp(gn, name, (size_t)len) != 0) continue;
        if (matlab_ws->kinds[i] == 13) out = (int32_t)matlab_ws->f64_vals[i];
        break;
    }
    matlab_ws_unlock();
    return out;
}

/* Matrix-argument handle trampolines (#119).  Mirror the scalar s* family
 * but pass matlab_mat* arguments.  `m*` = scalar (double) return (vector
 * objective ABI, fminunc/fmincon-style); `mm*` = matrix (matlab_mat*)
 * return (vector->vector ABI, residual/model-style). */
double matlab_call_handle_m1(void *fn, void *a) {
    if (!fn) return 0.0;
    return ((double (*)(void *))fn)(a);
}
double matlab_call_handle_m2(void *fn, void *a, void *b) {
    if (!fn) return 0.0;
    return ((double (*)(void *, void *))fn)(a, b);
}
void *matlab_call_handle_mm1(void *fn, void *a) {
    if (!fn) return nullptr;
    return ((void *(*)(void *))fn)(a);
}
void *matlab_call_handle_mm2(void *fn, void *a, void *b) {
    if (!fn) return nullptr;
    return ((void *(*)(void *, void *))fn)(a, b);
}

void *matlab_ws_get_handle(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    void *out = nullptr;
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *gn = matlab_ws->names[i];
        if (!gn) continue;
        size_t glen = strlen(gn);
        if (glen != (size_t)len) continue;
        if (memcmp(gn, name, (size_t)len) != 0) continue;
        if (matlab_ws->kinds[i] == 13) out = matlab_ws->ptr_vals[i];
        break;
    }
    matlab_ws_unlock();
    matlab_ws_check_read_watch(name, len);
    return out;
}

/* Scalar function-handle trampolines.  The compiler emits one of these
 * at a `f(args)` call site where `f` is a workspace-backed handle: the
 * first argument is the stored function pointer, the rest are the
 * f64 call arguments.  We cast the pointer to the matching scalar
 * signature and invoke it.  Covers the common `double -> double` math
 * builtins (matlab_sin_s, matlab_sqrt_s, ...) and capture-free user /
 * anonymous functions whose monomorphised signature is all-f64. */
double matlab_call_handle_s0(void *fn) {
    if (!fn) return 0.0;
    return ((double (*)())fn)();
}
double matlab_call_handle_s1(void *fn, double a) {
    if (!fn) return 0.0;
    return ((double (*)(double))fn)(a);
}
double matlab_call_handle_s2(void *fn, double a, double b) {
    if (!fn) return 0.0;
    return ((double (*)(double, double))fn)(a, b);
}
double matlab_call_handle_s3(void *fn, double a, double b, double c) {
    if (!fn) return 0.0;
    return ((double (*)(double, double, double))fn)(a, b, c);
}

/* Forward-declare the heterogeneous types stored as opaque pointers
 * in the workspace. The real layouts live in matlab_runtime.cpp /
 * the runtime_sym module — runtime_debug.cpp only ever round-trips
 * them through `ptr_vals[idx]` so the type-name is purely for
 * function-signature documentation. */
struct matlab_table_s;        typedef struct matlab_table_s        matlab_table;
struct matlab_categorical_s;  typedef struct matlab_categorical_s  matlab_categorical;
struct matlab_datetime_s;     typedef struct matlab_datetime_s     matlab_datetime;
struct matlab_duration_s;     typedef struct matlab_duration_s     matlab_duration;

/* Phase 5.3 — table workspace setter (kind=6). Stores a matlab_table*
 * so the DAP formatter and Workspace pane render the variable as
 * `NxM table` and the drill-in walks columns via matlab_table_column_*
 * instead of casting the pointer to matlab_mat* (which made the row
 * report "16x52357604992 double" — the table's internal pointer
 * reinterpreted as a column count — and segfaulted on click as
 * matlab_dbg_mat_get walked off the allocation). Picks the lowest
 * free kind slot above the existing block (0–5). */
void matlab_ws_set_table(const char *name, int64_t len, matlab_table *t) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 6;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = t;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/6, t);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Phase 5.2 — categorical workspace setter (kind=9). Mirrors the
 * table setter shape. Kind 9 sits above the sym slots (7/8) so the
 * decode tables stay contiguous. */
void matlab_ws_set_categorical(const char *name, int64_t len,
                                matlab_categorical *c) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 9;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = c;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/9, c);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Phase 5.1 — datetime workspace setter (kind=10). */
void matlab_ws_set_datetime(const char *name, int64_t len,
                             matlab_datetime *d) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 10;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = d;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/10, d);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* Phase 5.1 — duration workspace setter (kind=11). */
void matlab_ws_set_duration(const char *name, int64_t len,
                             matlab_duration *d) {
    matlab_ws_init_if_needed();
    matlab_ws_lock();
    struct matlab_dbg_undo_rec *r =
        matlab_ws_push_undo_locked(name, len, /*kind=*/2);
    int32_t idx = struct_reserve(matlab_ws, name, (int32_t)len);
    matlab_ws->kinds[idx] = 11;
    matlab_ws->f64_vals[idx] = 0.0;
    matlab_ws->ptr_vals[idx] = d;
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/11, d);
    matlab_ws_unlock();
    matlab_ws_check_watch(name, len);
}

/* DAP variable formatter — pretty-prints a matlab_sym* into a stable
 * static buffer and returns it. The DAP server reads the result via
 * the value column of `variables` requests. Returns NULL on miss.
 * Only available when the build was configured with MATLAB_LLVM_WITH_SYM
 * (otherwise the runtime can't link matlab_sym_str from runtime_sym.cpp). */
#ifdef MATLAB_LLVM_WITH_SYM
extern "C" char *matlab_sym_str(const void *e, int64_t *len_out);
extern "C" char *matlab_symmat_str(const void *m, int64_t *len_out);
const char *matlab_dbg_sym_str(void *s, int64_t *len_out) {
    static thread_local char *Cached = nullptr;
    if (Cached) { free(Cached); Cached = nullptr; }
    Cached = matlab_sym_str(s, len_out);
    return Cached;
}
const char *matlab_dbg_symmat_str(void *m, int64_t *len_out) {
    static thread_local char *Cached = nullptr;
    if (Cached) { free(Cached); Cached = nullptr; }
    Cached = matlab_symmat_str(m, len_out);
    return Cached;
}
#else
const char *matlab_dbg_sym_str(void *s, int64_t *len_out) {
    (void)s;
    if (len_out) *len_out = 0;
    return nullptr;
}
const char *matlab_dbg_symmat_str(void *m, int64_t *len_out) {
    (void)m;
    if (len_out) *len_out = 0;
    return nullptr;
}
#endif

double matlab_ws_has(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    return matlab_struct_has_field(matlab_ws, name, len);
}

/* For the REPL's `whos` / `clear` style commands. */
void matlab_ws_clear(void) {
    /* Cheapest correct clear: allocate a fresh struct and let the old
     * one leak. Leak is bounded by the number of clear() calls in a
     * session, which is negligible for human-paced use. */
    matlab_ws = matlab_struct_new();
}

/* Forward declaration — the definition is later in the file, but
 * matlab_ws_clear_one needs the symbol. */
matlab_struct *matlab_struct_rmfield(matlab_struct *s, const char *name,
                                      int64_t len);

/* Remove a single variable from the workspace. Silent no-op if the
 * name isn't present. Matches MATLAB's `clear name` form. */
void matlab_ws_clear_one(const char *name, int64_t len) {
    matlab_ws_init_if_needed();
    matlab_struct_rmfield(matlab_ws, name, len);
}

/* `who` prints just the variable names, one per line. `whos` adds
 * shape/class columns. Both read the workspace struct. */
void matlab_ws_who(void) {
    matlab_ws_init_if_needed();
    pthread_mutex_lock(&matlab_io_mutex);
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        printf("%s\n", matlab_ws->names[i]);
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

void matlab_ws_whos(void) {
    matlab_ws_init_if_needed();
    pthread_mutex_lock(&matlab_io_mutex);
    printf("  %-16s %-16s %-8s\n", "Name", "Size", "Class");
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        const char *name = matlab_ws->names[i];
        if (matlab_ws->kinds[i] == 0) {
            printf("  %-16s %-16s %-8s\n", name, "1x1", "double");
        } else if (matlab_ws->kinds[i] == 1) {
            matlab_mat *m = (matlab_mat *)matlab_ws->ptr_vals[i];
            char shape[64];
            if (!m) {
                snprintf(shape, sizeof shape, "-");
            } else if (mat_is_nd(m)) {
                /* Tier C matN: write the dims tuple as "AxBxCx..." */
                matlab_matN *mn = (matlab_matN *)m;
                int n = 0;
                for (uint32_t k = 0; k < mn->ndims && n < (int)sizeof(shape) - 8; ++k) {
                    n += snprintf(shape + n, sizeof(shape) - (size_t)n,
                                  k == 0 ? "%lld" : "x%lld",
                                  (long long)mn->dims[k]);
                }
            } else if (mat_is_3d(m)) {
                matlab_mat3 *m3 = (matlab_mat3 *)m;
                snprintf(shape, sizeof shape, "%lldx%lldx%lld",
                         (long long)m3->rows, (long long)m3->cols,
                         (long long)m3->depth);
            } else {
                snprintf(shape, sizeof shape, "%lldx%lld",
                         (long long)m->rows, (long long)m->cols);
            }
            printf("  %-16s %-16s %-8s\n", name, shape, "double");
        } else if (matlab_ws->kinds[i] == 3) {
            printf("  %-16s %-16s %-8s\n", name, "1x1", "string");
        } else if (matlab_ws->kinds[i] == 13) {
            printf("  %-16s %-16s %-8s\n", name, "1x1", "function_handle");
        } else {
            printf("  %-16s %-16s %-8s\n", name, "?", "?");
        }
    }
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* dbg(x) / dbg(x, "label") — source-located debug print to stderr.
 * The frontend passes the source file + line (derived from the call
 * site's Location) and the variable name (empty if the argument
 * isn't a bare NameExpr). Value is either an f64 scalar or a
 * matlab_mat* depending on which overload the lowerer selected. */
void matlab_dbg_f64(const char *file, int64_t file_len,
                    int32_t line,
                    const char *label, int64_t label_len,
                    double v) {
    /* The file / label strings come from LLVM globals that are NOT
     * null-terminated, so use the explicit length in the format. */
    int fl = (int)(file_len > 0 ? file_len : 0);
    int ll = (int)(label_len > 0 ? label_len : 0);
    const char *flt = file ? file : "<repl>";
    if (!file) fl = (int)strlen(flt);
    pthread_mutex_lock(&matlab_io_mutex);
    fprintf(stderr, "%.*s:%d: %.*s = %g\n",
            fl, flt, line,
            ll > 0 ? ll : (int)strlen("<expr>"),
            ll > 0 ? label : "<expr>", v);
    pthread_mutex_unlock(&matlab_io_mutex);
}

/* -------- Full DAP hook infrastructure ------------------------------------
 *
 * Injected into the JIT'd code by matlabc -g / matlabc -dap. The hook is
 * called at each top-level statement boundary with (file_id, line) where
 * file_id is the SourceManager's FileID cast to i32. The DAP server
 * (in matlabc's -dap mode) shares this state via locks + condvar and
 * drives the debuggee through setBreakpoints / continue / next commands.
 *
 * Breakpoints are stored as a linear array keyed by (file_id, line). A
 * small capped array is fine since human-set breakpoints don't scale
 * past a few dozen.
 *
 * Frames are tracked by matlab_dbg_enter_frame / _leave_frame so the
 * DAP server can return a multi-entry stackTrace. When -g is on, every
 * emitted user-function body calls enter on entry and leave before
 * each return.
 */
#define MATLAB_DBG_MAX_BREAKPOINTS 256
#define MATLAB_DBG_MAX_FRAMES 128
/* Per-frame Locals: bounded at lowering time by how many distinct
 * named slots a single user function can carry. 64 is well above what
 * any of our examples currently produce; bump if needed. */
#define MATLAB_DBG_MAX_LOCALS 64

/* Per-frame mini-workspace entry. Mirrors the shape of the script-
 * level matlab_ws struct but is keyed by frame index so the DAP
 * server can pick the right slice for the user's selected frame.
 * `kind` follows the same convention as matlab_dbg_ws_kind:
 *   0 = f64 scalar
 *   1 = matlab_mat * (numeric matrix descriptor)
 *   2 = matlab_obj * (user classdef instance — `ptr` is the obj,
 *       its class_id field doubles as the registry key for class
 *       names). The matrix / object pointers are borrowed from the
 *       JIT's slot — the slot is alive for the lifetime of the
 *       frame, which is exactly when the DAP server reads from us. */
struct matlab_dbg_local {
    char *name;       /* heap-copied, null-terminated */
    int64_t name_len;
    int kind;         /* 0 = f64, 1 = matrix ptr, 2 = obj ptr */
    double f64;
    void *ptr;
};

struct matlab_dbg_frame_locals {
    int n;
    struct matlab_dbg_local entries[MATLAB_DBG_MAX_LOCALS];
};

enum matlab_dbg_action {
    MATLAB_DBG_RUN       = 0,   /* no pause (no breakpoints hit) */
    MATLAB_DBG_CONTINUE  = 1,   /* resume from a pause */
    MATLAB_DBG_STEP_OVER = 2,   /* break at next statement at <= target depth */
    MATLAB_DBG_STEP_IN   = 3,   /* break at the very next statement */
    MATLAB_DBG_STEP_OUT  = 4,   /* break at next statement at <  target depth */
    MATLAB_DBG_STOP      = 5,   /* terminate the program */
};

struct matlab_dbg_frame {
    int32_t file_id;
    int32_t line;
    const char *fn_name;
};

/* Record kinds for the reverse-stepping undo log. matlab_dbg_state
 * holds a fixed-size ring buffer of these; matlab_dbg_step_back
 * walks them in reverse to revert variable writes.
 *
 *   0 = statement boundary {file_id, line, thread_slot}
 *   1 = ws_set_f64 {name, prev_kind, prev_f64, prev_existed}
 *   2 = ws_set_mat / ws_set_obj {name, prev_kind, prev_ptr,
 *       prev_existed}
 *   3 = frame_set_* {thread_slot, frame_idx, name, prev_kind,
 *       prev_f64, prev_ptr, prev_existed}
 *   4 = irreversible-op marker (disp / fprintf etc.) — stepBack
 *       refuses to walk past one of these. */
struct matlab_dbg_undo_rec {
    int8_t kind;
    int8_t prev_kind;
    int8_t prev_existed;
    int8_t new_kind;       /* post-write: 0=f64, 1=mat, 2=obj. Used by
                            * the redo path to replay forward through the
                            * log after stepBack — without this we'd only
                            * have the prior values and forward-stepping
                            * after stepBack would have to resume the JIT
                            * from its actual parked PC (one statement
                            * ahead of the rewound caret), confusing the
                            * user into thinking a line was skipped. */
    int32_t file_id;
    int32_t line;
    int32_t frame_idx;
    int32_t thread_slot;
    char *name;        /* heap-owned for kinds 1/2/3 */
    int64_t name_len;
    double prev_f64;
    void *prev_ptr;
    double new_f64;
    void *new_ptr;
};

struct matlab_dbg_state {
    int enabled;
    int stop_on_entry;
    pthread_mutex_t mu;
    pthread_cond_t cv_client;   /* debugger thread waits on this when paused */
    pthread_cond_t cv_server;   /* server waits on this when requesting pause */

    /* Last-hit pause point, published after the hook blocks. */
    int paused;
    int32_t cur_file_id;
    int32_t cur_line;
    /* Index into bp_* of the breakpoint that triggered the current
     * pause. -1 when the pause came from stepping rather than a bp.
     * The DAP server reads cond_text[cur_bp_idx] / log_text[...] to
     * decide whether to evaluate before notifying the IDE. */
    int cur_bp_idx;

    /* What to do when resumed. */
    enum matlab_dbg_action action;
    int32_t step_target_depth;

    /* Exception-breakpoint filter: when set, the hook pauses on the
     * first statement after matlab_set_error fires. Toggled by the
     * DAP server's `setExceptionBreakpoints` handler in response to
     * the IDE's "Pause on Errors" UI. */
    int pause_on_error;

    /* Set non-zero when the current pause was triggered by a
     * `keyboard` builtin call (not a step / bp / error). The DAP
     * server reads this in monitorMain to surface a stop reason of
     * "entry" so the IDE renders the keyboard glyph rather than a
     * generic step/pause. Cleared by the next resume. */
    int paused_from_keyboard;

    /* Data breakpoints (write watchpoints). The DAP server adds an
     * entry via matlab_dbg_add_watchpoint; the runtime's set_*
     * functions check the table after every workspace / frame-local
     * write and trip a pause if the name matches.
     *
     * Scope encoding:
     *   0 = "any" (matches script ws *or* any frame)
     *   1 = script workspace only (matlab_ws_set_*)
     *   2 = innermost frame only (matlab_dbg_frame_set_*)
     * v1 ships scope=0 since the DAP IDE picks the watch from the
     * Variables panel and the user expects "stop when this name
     * gets reassigned anywhere"; tighter scoping can layer on later
     * via the dataBreakpointInfo `accessType` argument.
     *
     * `last_writer_idx` is set by the set_* sites when a watchpoint
     * trips, mirroring how cur_bp_idx works for line breakpoints —
     * the DAP server reads it to surface the originating watch's
     * id in the stopped event's hitBreakpointIds. */
    int n_wp;
    char *wp_name[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t wp_name_len[MATLAB_DBG_MAX_BREAKPOINTS];
    int32_t wp_scope[MATLAB_DBG_MAX_BREAKPOINTS];
    int32_t wp_id[MATLAB_DBG_MAX_BREAKPOINTS];   /* DAP-assigned id */
    /* Access kind: 0 = write only (default; matches the original
     * watch-on-set behaviour), 1 = read only, 2 = read+write.
     * The check helpers below filter by this so a read-only watch
     * doesn't trip on a regular `matlab_ws_set_*`. */
    int8_t  wp_access[MATLAB_DBG_MAX_BREAKPOINTS];
    int last_wp_idx;   /* index of the watchpoint that tripped, or -1 */
    int paused_from_watch;

    /* Breakpoints (file_id, line) — linear scan. cond_text and
     * log_text are heap-owned (NULL when absent). cond_disabled flips
     * to 1 once the DAP server reports a condition syntax error so
     * subsequent hits don't keep retrying it.
     *
     * Hit-count gating: hit_count counts every time the hook reaches
     * this bp's line (incremented unconditionally on a match);
     * hit_op + hit_target encode the user's `hitCondition` (e.g.
     * `>= 100` is op=GE, target=100). The hook compares count vs.
     * target with op — only triggers a pause when the test passes.
     * op=0 means no hit-count gate (default; the bp pauses every
     * time the line runs). */
    int n_bp;
    int32_t bp_file[MATLAB_DBG_MAX_BREAKPOINTS];
    int32_t bp_line[MATLAB_DBG_MAX_BREAKPOINTS];
    char *cond_text[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t cond_len[MATLAB_DBG_MAX_BREAKPOINTS];
    char *log_text[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t log_len[MATLAB_DBG_MAX_BREAKPOINTS];
    int cond_disabled[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t hit_count[MATLAB_DBG_MAX_BREAKPOINTS];
    int64_t hit_target[MATLAB_DBG_MAX_BREAKPOINTS];
    /* hit_op encoding (0 = none, no gate):
     *   1 = ==   (stop on the Nth hit only)
     *   2 = >=   (stop on hit N and every hit after — most common)
     *   3 = >    (stop after N hits)
     *   4 = %    (stop every Nth hit, e.g. `%5` for every 5th iter)
     * Anything else is treated as no gate. */
    int hit_op[MATLAB_DBG_MAX_BREAKPOINTS];

    /* Frame stack. The shared (cross-thread) frames[] array stays here
     * for the legacy single-threaded DAP path, but per-thread frame
     * chains (see thread_frames[] below) now own the source of
     * truth in multi-threaded sessions.
     *
     * On first hook fire from a given thread, n_frames is copied into
     * the thread's per-thread slot from this template. Subsequent
     * enter_frame / leave_frame / frame_set_* mutate the thread's
     * own slot. The shared frames[] is updated in lockstep with the
     * paused-thread's chain so DAP inspectors that read frames[]
     * directly (the legacy code paths) keep working — the
     * paused-thread's view is what's exposed. */
    int n_frames;
    struct matlab_dbg_frame frames[MATLAB_DBG_MAX_FRAMES];
    /* Per-frame Locals. Index aligns with `frames[]`: frame 0 is the
     * script's mini-ws (parallel to matlab_ws but populated by the
     * lowering's mirror calls — covers loop induction variables and
     * other slot-stored vars that don't go through matlab_ws_set_*).
     * Frames 1..n-1 are user-function frames. Cleared on enter, freed
     * on leave. */
    struct matlab_dbg_frame_locals frame_locals[MATLAB_DBG_MAX_FRAMES];

    /* Per-thread frame chain. `thread_keys[i]` is the pthread_t;
     * `thread_n_frames[i]` / `thread_frames[i][]` /
     * `thread_frame_locals[i][]` is that thread's own call-stack
     * state. The thread registry (thread_keys / thread_ids /
     * n_threads) above is the index. The hook reads/writes the
     * calling thread's slot; the DAP inspector functions
     * (frame_count / frame_at / frame_local_*) take an implicit
     * thread idx via paused_thread_idx so the pause is reported
     * against the right call stack.
     *
     * Capacity-32 matches the registry; per-thread MAX_FRAMES is
     * the same as the shared frames[]. The memory cost is bounded
     * (~32 * 32 * sizeof(matlab_dbg_frame)) so we just inline. */
    int                          thread_n_frames[32];
    struct matlab_dbg_frame      thread_frames[32][MATLAB_DBG_MAX_FRAMES];
    struct matlab_dbg_frame_locals thread_frame_locals[32][MATLAB_DBG_MAX_FRAMES];
    int32_t                      thread_step_target_depth[32];

    /* Reverse-stepping undo log. Members declared inline; the
     * matlab_dbg_undo_rec struct itself is at file scope (above
     * matlab_dbg_state). */
    int n_undo;
    int undo_head;     /* next slot to write (ring buffer head) */
    int undo_full;     /* set once we've wrapped — gates how far we can rewind */
    struct matlab_dbg_undo_rec undo_log[4096];
    /* Recording flag — clear during the rewind itself so
     * apply-undo's reverse-set doesn't push a meta-record. */
    int recording_undo;
    /* Rewind <-> redo bookkeeping. After a stepBack the JIT
     * thread is still parked one statement past the rewound
     * caret; a forward step that simply resumed the JIT would
     * confuse the user (they see line 17 in the IDE but resume
     * lands at line 20 with line 19's writes applied). Instead,
     * the DAP `next`/`stepIn`/`continue` handlers consult
     * `rewound` and route through matlab_dbg_step_forward_redo —
     * walking the undo log forward (past the current
     * undo_head, up to redo_cap) and re-applying each record's
     * post-write state. When undo_head catches up to redo_cap,
     * `rewound` clears and the next forward step resumes the
     * JIT normally. */
    int rewound;
    int redo_cap;      /* ring index one past the last "future" slot —
                        * mirrors how undo_head normally points one
                        * past the live tail. While rewound, slots in
                        * [undo_head, redo_cap) are the future records
                        * the user can walk forward through. */

    /* File-id <-> name table. Populated by matlab_dbg_register_file. */
    int n_files;
    const char *file_names[256];
    int64_t file_name_lens[256];

    /* Thread registry. Populated lazily on first hook entry from
     * each pthread that runs JIT'd code (the main worker plus any
     * parfor-spawned workers). The DAP server's `threads` request
     * enumerates this list; `stopped` events name the originating
     * thread by id.
     *
     * Identity is the pthread_t value; the `id` we hand to the DAP
     * client is a sequential integer (1 = main worker; 2..N =
     * parfor workers in spawn order). The mapping is one-shot per
     * thread and persists for the rest of the session — even if a
     * thread is joined, we keep its slot so any earlier `stopped`
     * event id stays valid in the IDE's UI history.
     *
     * Limitation (v1): the frame stack itself (`frames[]` /
     * `frame_locals[]` / `n_frames` above) is shared across all
     * threads. A parfor body that hits a bp will surface the right
     * thread id in the stopped event, but `stackTrace(threadId)`
     * returns whatever the last-modifying thread put on the global
     * stack — which can be the queried thread or a sibling. Per-
     * thread frame stacks are the follow-up; documented in
     * docs/debug.md. */
    int n_threads;
    pthread_t thread_keys[32];
    int32_t  thread_ids[32];   /* sequential, 1-based; matches DAP threadId */
    /* Index into thread_keys/_ids of the thread that hit the
     * current pause, or -1 when no pause is active. Set by the
     * hook when should_pause flips on; cleared on resume. The
     * DAP server reads it via matlab_dbg_paused_thread_id() to
     * surface the originating thread on `stopped` events. */
    int paused_thread_idx;

    /* Class-id -> class-name table. Populated by
     * matlab_dbg_register_class at the top of the script body when -g
     * is on (one entry per classdef in the translation unit). The DAP
     * server uses this to surface a class instance as
     * `1x1 ClassName` in the LOCALS panel and in the watch box.
     * 64 is far above what any realistic program touches; a linear
     * scan is cheap given how rarely these are read.
     *
     * `class_names[i]` is heap-copied on register and never freed —
     * the registration is once-per-program and the strings are tiny. */
    int n_classes;
    int32_t class_ids[64];
    char *class_names[64];
    int64_t class_name_lens[64];
};

static struct matlab_dbg_state matlab_dbg = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv_client = PTHREAD_COND_INITIALIZER,
    .cv_server = PTHREAD_COND_INITIALIZER,
    .action = MATLAB_DBG_RUN,
};

/* Forward decls for the per-thread chain helpers — definitions
 * live further down alongside enter_frame. Multiple call sites
 * up here (err_snapshot_frames, watch_trip, keyboard_hook) need
 * to consult the per-thread chain before its definition appears,
 * so they're declared at the top of the matlab_dbg section.
 * Same shape for the undo-log helpers used by enable() and the
 * matlab_ws_set_* / matlab_dbg_frame_set_* call sites further
 * up the file. */
static int matlab_dbg_thread_slot_locked(void);
static int matlab_dbg_thread_init_chain_locked(void);
static void matlab_dbg_undo_clear_locked(void);

/* Forward decl: defined alongside matlab_dbg_enter_frame below but
 * called from matlab_dbg_enable to clear any frame-locals state left
 * over from a prior launch. */
static void matlab_dbg_free_frame_locals(int frame_idx);

/* --- error() backtrace snapshot --------------------------------------
 *
 * matlab_set_error / matlab_set_error_msg snapshot the current frame
 * stack here BEFORE any unwind pops the runtime frames. Without the
 * snapshot, by the time the script returns to the DAP server (or a
 * `disp(ME.message)` runs in a catch body) the leave_frame calls
 * fired on each function return have erased the call site that
 * threw, leaving us with nothing useful to print.
 *
 * The snapshot is intentionally a value-copy (file_id, line, name
 * pointer). The fn_name pointers stored in matlab_dbg.frames[].fn_name
 * are runtime-owned (either string literals from the JIT'd const
 * globals, or "<script>" itself), so copying the pointer is safe —
 * they outlive the snapshot.
 *
 * `matlab_err_emit_traceback_to_stderr` prints the snapshot to stderr
 * with the format:
 *
 *   error: <msg>
 *     at <fn> (<file>:<line>)
 *     at <fn> (<file>:<line>)
 *
 * Gated on matlab_dbg.enabled so that non-debug binaries (the
 * production -emit-c / -emit-cpp / -emit-llvm path with no -dap)
 * keep their existing silent semantics — only DAP / `-g` runs see
 * the diagnostic. */
static int matlab_err_n_frames = 0;
static struct matlab_dbg_frame matlab_err_frames[MATLAB_DBG_MAX_FRAMES];

/* Forward decl — defined later in the file. The error snapshot
 * path and the hook both need it, but its body sits next to the
 * other thread-chain helpers far below. */
static int matlab_dbg_thread_init_chain_locked(void);

/* Non-static: called from matlab_runtime.cpp's matlab_set_error path. */
void matlab_err_snapshot_frames(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Free any names retained from a previous error snapshot before
     * stamping new ones in — otherwise repeated error() calls leak. */
    for (int i = 0; i < matlab_err_n_frames; ++i) {
        free((char *)matlab_err_frames[i].fn_name);
        matlab_err_frames[i].fn_name = NULL;
    }
    /* Snapshot the calling thread's per-thread chain (post-refactor
     * source of truth for frame state). The shared frames[] is now
     * a paused-thread snapshot, refreshed only on hook pause; an
     * error fired between pauses sees stale data there. The
     * per-thread chain is always up-to-date because every
     * enter_frame / leave_frame / hook fire from this thread
     * touched it directly. */
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
    for (int i = 0; i < n; ++i) {
        matlab_err_frames[i].file_id =
            matlab_dbg.thread_frames[slot][i].file_id;
        matlab_err_frames[i].line =
            matlab_dbg.thread_frames[slot][i].line;
        const char *src = matlab_dbg.thread_frames[slot][i].fn_name;
        if (src) {
            size_t L = strlen(src);
            char *copy = (char *)malloc(L + 1);
            if (copy) { memcpy(copy, src, L); copy[L] = '\0'; }
            matlab_err_frames[i].fn_name = copy;
        } else {
            matlab_err_frames[i].fn_name = NULL;
        }
    }
    matlab_err_n_frames = n;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Resolve a file_id back to its registered name. Mirrors
 * matlab_dbg_file_name but is callable without holding the dbg mutex
 * (the caller already takes care of synchronization). Returns
 * "<unknown>" when the file_id is out of range. */
static const char *matlab_err_file_name_locked(int32_t file_id, int64_t *len_out) {
    int max = (int)(sizeof matlab_dbg.file_names /
                    sizeof matlab_dbg.file_names[0]);
    if (file_id >= 1 && file_id <= max) {
        const char *name = matlab_dbg.file_names[file_id - 1];
        if (name) {
            if (len_out) *len_out = matlab_dbg.file_name_lens[file_id - 1];
            return name;
        }
    }
    static const char unknown[] = "<unknown>";
    if (len_out) *len_out = (int64_t)(sizeof unknown - 1);
    return unknown;
}

/* Non-static: called from matlab_runtime.cpp's matlab_set_error path. */
void matlab_err_emit_traceback_to_stderr(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int debug_on = matlab_dbg.enabled;
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (!debug_on) return;

    /* Build the whole traceback into a fixed-size buffer and emit it
     * via a single write(2) call. We can't use fprintf here because
     * libc's <stdio.h> file lock can deadlock if the worker thread
     * happens to hold a recursive_mutex inside LLVM's ExecutionEngine
     * at the point error() fires — observed during DAP shutdown when
     * stderr-bound fprintf races with the engine's own diagnostic
     * stream. write(2) bypasses all of that. */
    char buf[2048];
    size_t off = 0;
    #define APP_LIT(s) do { \
        size_t l = sizeof(s) - 1; \
        if (off + l > sizeof buf) l = sizeof buf - off; \
        memcpy(buf + off, s, l); \
        off += l; \
    } while (0)
    #define APP(fmt, ...) do { \
        if (off < sizeof buf) { \
            int n = snprintf(buf + off, sizeof buf - off, fmt, __VA_ARGS__); \
            if (n > 0) off += (size_t)n > sizeof buf - off \
                              ? sizeof buf - off : (size_t)n; \
        } \
    } while (0)

    APP_LIT("error: ");
    if (matlab_error_msg_len > 0) {
        size_t mlen = (size_t)matlab_error_msg_len;
        if (off + mlen > sizeof buf) mlen = sizeof buf - off;
        memcpy(buf + off, matlab_error_msg, mlen);
        off += mlen;
    }
    APP_LIT("\n");

    pthread_mutex_lock(&matlab_dbg.mu);
    for (int idx = matlab_err_n_frames - 1; idx >= 0; --idx) {
        const struct matlab_dbg_frame *f = &matlab_err_frames[idx];
        const char *fn = f->fn_name ? f->fn_name : "<frame>";
        int64_t fnLen = 0;
        const char *file = matlab_err_file_name_locked(f->file_id, &fnLen);
        APP("  at %s (%.*s:%d)\n", fn, (int)fnLen, file, f->line);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    #undef APP

    (void)!write(STDERR_FILENO, buf, off);
}

/* Public read-only accessors so the DAP server (or a future REPL UI)
 * can render the same backtrace as a structured response. */
int matlab_err_traceback_count(void) {
    int n;
    pthread_mutex_lock(&matlab_dbg.mu);
    n = matlab_err_n_frames;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

int matlab_err_traceback_at(int i, int32_t *file_id, int32_t *line,
                             const char **fn_name) {
    int ok = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    /* i = 0 = innermost; mirrors matlab_dbg_frame_at's API shape. */
    int idx = matlab_err_n_frames - 1 - i;
    if (idx >= 0 && idx < matlab_err_n_frames) {
        if (file_id) *file_id = matlab_err_frames[idx].file_id;
        if (line)    *line    = matlab_err_frames[idx].line;
        if (fn_name) *fn_name = matlab_err_frames[idx].fn_name;
        ok = 1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* DAP `setExceptionBreakpoints` plumbing — toggle the pause-on-error
 * filter the hook checks above. Held under matlab_dbg.mu so a flip
 * mid-eval doesn't race the hook's read of the same field. */
void matlab_dbg_set_pause_on_error(int on) {
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.pause_on_error = on ? 1 : 0;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP `exceptionInfo` reader — surfaces the message captured by
 * matlab_set_error_msg before the unwind. Returns NULL/0 when no
 * error has fired this session. The buffer is owned by the runtime
 * (static char[1024], null-terminated); the caller must not free it. */
const char *matlab_dbg_last_error_msg(int64_t *len_out) {
    if (len_out) *len_out = matlab_error_msg_len;
    return matlab_error_msg_len > 0 ? matlab_error_msg : NULL;
}

/* Lowered call site for a `keyboard` builtin in user code. Sets
 * paused=1 and blocks on the same condvar a real breakpoint uses,
 * so the DAP server's monitor thread wakes and emits a `stopped`
 * event. The `paused_from_keyboard` flag tells monitorMain to
 * surface stop reason="entry" instead of "step" — the IDE then
 * renders the keyboard / pause-on-source glyph rather than a
 * generic step icon.
 *
 * No-op when matlab_dbg.enabled == 0 (release builds without -g):
 * a `keyboard` call simply returns immediately. The latest source
 * location is left as whatever the most recent matlab_dbg_hook
 * recorded — already what the user wants for the call site
 * because the hook fires at the same statement. */
/* Forward decl — defined alongside the thread enumeration helpers
 * further down. The keyboard / watch trip code below needs it. */

void matlab_dbg_keyboard_hook(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (!matlab_dbg.enabled) {
        pthread_mutex_unlock(&matlab_dbg.mu);
        return;
    }
    int thr_idx = matlab_dbg_thread_init_chain_locked();
    /* Copy the innermost frame's (file_id, line) into the cur_*
     * fields so the DAP `stopped` event reports the keyboard call
     * site, then snapshot the calling thread's per-thread chain
     * into the shared frames[] view so DAP inspectors that read
     * frames[]/frame_locals[] directly see the caller's stack
     * (not whatever the last paused thread left). */
    int n_thr = matlab_dbg.thread_n_frames[thr_idx];
    if (n_thr > 0) {
        matlab_dbg.cur_file_id =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].file_id;
        matlab_dbg.cur_line =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].line;
    }
    int snap_n = n_thr > MATLAB_DBG_MAX_FRAMES ? MATLAB_DBG_MAX_FRAMES : n_thr;
    matlab_dbg.n_frames = snap_n;
    for (int i = 0; i < snap_n; ++i) {
        matlab_dbg.frames[i] = matlab_dbg.thread_frames[thr_idx][i];
        matlab_dbg.frame_locals[i] =
            matlab_dbg.thread_frame_locals[thr_idx][i];
    }
    matlab_dbg.cur_bp_idx = -1;
    matlab_dbg.paused = 1;
    matlab_dbg.paused_from_keyboard = 1;
    matlab_dbg.paused_thread_idx = thr_idx;
    pthread_cond_broadcast(&matlab_dbg.cv_server);
    while (matlab_dbg.paused) {
        pthread_cond_wait(&matlab_dbg.cv_client, &matlab_dbg.mu);
    }
    matlab_dbg.paused_from_keyboard = 0;
    matlab_dbg.paused_thread_idx = -1;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP-side reader: was the most recent pause triggered by a
 * keyboard() call? monitorMain checks this before mapping
 * (BpIdx == -1) to reason="step", switching to "entry" instead
 * when this flag is set. */
int matlab_dbg_was_paused_from_keyboard(void) {
    int v;
    pthread_mutex_lock(&matlab_dbg.mu);
    v = matlab_dbg.paused_from_keyboard;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return v;
}

/* --- Data breakpoints (write watchpoints) --------------------------- */

/* Add a watchpoint by name with caller-assigned id (the DAP server
 * encodes its dataId from the name's hash so subsequent setBreakpoints
 * round-trips reuse the same id). scope is 0 (any) / 1 (script-ws) /
 * 2 (innermost-frame). Returns 1 on success, 0 on table-full or
 * duplicate. The runtime owns the heap-copy of `name`. */
/* Forward decl — _ex body follows immediately, but the back-compat
 * shim above forwards into it. */
int matlab_dbg_add_watchpoint_ex(const char *name, int64_t name_len,
                                  int32_t scope, int32_t id,
                                  int32_t access);

int matlab_dbg_add_watchpoint(const char *name, int64_t name_len,
                               int32_t scope, int32_t id) {
    /* Backward-compat shim — defaults to write-only (the original
     * accessType v1 supported). New callers should use the _ex
     * variant below. */
    return matlab_dbg_add_watchpoint_ex(name, name_len, scope, id,
                                         /*access=*/0);
}

int matlab_dbg_add_watchpoint_ex(const char *name, int64_t name_len,
                                  int32_t scope, int32_t id,
                                  int32_t access) {
    if (!name || name_len <= 0) return 0;
    if (access < 0 || access > 2) access = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    /* De-dup: if a watch with the same id already exists, refresh
     * its scope+access rather than appending a duplicate row. */
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        if (matlab_dbg.wp_id[i] == id) {
            matlab_dbg.wp_scope[i] = scope;
            matlab_dbg.wp_access[i] = (int8_t)access;
            pthread_mutex_unlock(&matlab_dbg.mu);
            return 1;
        }
    }
    int ok = matlab_dbg.n_wp < MATLAB_DBG_MAX_BREAKPOINTS;
    if (ok) {
        int i = matlab_dbg.n_wp;
        matlab_dbg.wp_name[i] = (char *)malloc((size_t)name_len + 1);
        memcpy(matlab_dbg.wp_name[i], name, (size_t)name_len);
        matlab_dbg.wp_name[i][name_len] = '\0';
        matlab_dbg.wp_name_len[i] = name_len;
        matlab_dbg.wp_scope[i] = scope;
        matlab_dbg.wp_id[i] = id;
        matlab_dbg.wp_access[i] = (int8_t)access;
        matlab_dbg.n_wp++;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Wipe the entire watchpoint table. The DAP `setDataBreakpoints`
 * request passes a fresh full list each time (same semantics as
 * setBreakpoints), so the cleanest implementation is clear-then-add.
 * Keeps the per-call code in the DAP handler simple. */
void matlab_dbg_clear_watchpoints(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        free(matlab_dbg.wp_name[i]);
        matlab_dbg.wp_name[i] = NULL;
        matlab_dbg.wp_name_len[i] = 0;
        matlab_dbg.wp_scope[i] = 0;
        matlab_dbg.wp_id[i] = 0;
        matlab_dbg.wp_access[i] = 0;
    }
    matlab_dbg.n_wp = 0;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP-side reader for the stopped-event handler: returns the id of
 * the most recent tripped watchpoint, or 0 if no watch has tripped
 * since the last resume. Cleared on resume by the worker's hook. */
int32_t matlab_dbg_last_watchpoint_id(void) {
    int32_t id = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.last_wp_idx >= 0 &&
        matlab_dbg.last_wp_idx < matlab_dbg.n_wp)
        id = matlab_dbg.wp_id[matlab_dbg.last_wp_idx];
    pthread_mutex_unlock(&matlab_dbg.mu);
    return id;
}

/* "Was the most recent pause caused by a tripped watchpoint?" — same
 * shape as matlab_dbg_was_paused_from_keyboard. The monitor checks
 * this when mapping BpIdx==-1 to a stop reason; "data breakpoint" is
 * the DAP standard reason for watchpoint hits. */
int matlab_dbg_was_paused_from_watch(void) {
    int v;
    pthread_mutex_lock(&matlab_dbg.mu);
    v = matlab_dbg.paused_from_watch;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return v;
}

/* Internal: scan the watchpoint table for a name match. scope_hint
 * is the call site's scope (1 = script-ws, 2 = frame-set); a watch
 * with scope=0 (any) matches both. Returns the index of the matching
 * watch, or -1 on miss. CALLER MUST HOLD matlab_dbg.mu — this is
 * called from inside the set_* lock-region. */
static int matlab_dbg_watch_check(const char *name, int64_t name_len,
                                   int32_t scope_hint) {
    /* Write-path: skip read-only watches. access==0 (write) and
     * access==2 (readWrite) qualify; access==1 (read-only) does
     * not. */
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        if (matlab_dbg.wp_name_len[i] != name_len) continue;
        int32_t s = matlab_dbg.wp_scope[i];
        if (s != 0 && s != scope_hint) continue;
        if (matlab_dbg.wp_access[i] == 1) continue;  /* read-only */
        if (memcmp(matlab_dbg.wp_name[i], name, (size_t)name_len) == 0)
            return i;
    }
    return -1;
}

/* Read-path counterpart. Called from matlab_ws_get_*; only matches
 * watches whose access kind includes "read" (1 or 2). Same scope-
 * filter shape as the write check. CALLER MUST HOLD matlab_dbg.mu. */
static int matlab_dbg_watch_check_read(const char *name, int64_t name_len,
                                        int32_t scope_hint) {
    for (int i = 0; i < matlab_dbg.n_wp; ++i) {
        if (matlab_dbg.wp_name_len[i] != name_len) continue;
        int32_t s = matlab_dbg.wp_scope[i];
        if (s != 0 && s != scope_hint) continue;
        int8_t a = matlab_dbg.wp_access[i];
        if (a != 1 && a != 2) continue;  /* write-only — skip */
        if (memcmp(matlab_dbg.wp_name[i], name, (size_t)name_len) == 0)
            return i;
    }
    return -1;
}

/* Internal: trip a watchpoint. Sets the paused-from-watch flag plus
 * cur_* fields and blocks on the same condvar a real bp uses, so
 * the DAP monitor wakes and emits a `stopped` event. Same pattern
 * as matlab_dbg_keyboard_hook. CALLER MUST HOLD matlab_dbg.mu. */
/* --- Reverse stepping (undo log) ---------------------------------- */

/* Append a record to the ring buffer, evicting the oldest if full.
 * For kinds 1/2/3 the heap-owned `name` of the evicted record is
 * freed. CALLER MUST HOLD matlab_dbg.mu. */
static struct matlab_dbg_undo_rec *matlab_dbg_undo_alloc_locked(void) {
    int slot = matlab_dbg.undo_head;
    struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[slot];
    /* Evict the previous tenant's heap allocation. */
    if (matlab_dbg.undo_full && r->name) {
        free(r->name);
        r->name = NULL;
    }
    matlab_dbg.undo_head = (slot + 1) % 4096;
    if (matlab_dbg.undo_head == 0) matlab_dbg.undo_full = 1;
    if (!matlab_dbg.undo_full) matlab_dbg.n_undo = matlab_dbg.undo_head;
    else matlab_dbg.n_undo = 4096;
    /* Reset to a clean record. */
    memset(r, 0, sizeof *r);
    return r;
}

/* Clear the entire undo log — called on enable() so re-launches
 * start fresh, and after a successful rewind so a subsequent
 * forward-step's writes don't conflate with stale undo records. */
static void matlab_dbg_undo_clear_locked(void) {
    int n = matlab_dbg.undo_full ? 4096 : matlab_dbg.undo_head;
    for (int i = 0; i < n; ++i) {
        free(matlab_dbg.undo_log[i].name);
        matlab_dbg.undo_log[i].name = NULL;
    }
    matlab_dbg.undo_head = 0;
    matlab_dbg.undo_full = 0;
    matlab_dbg.n_undo = 0;
    matlab_dbg.rewound = 0;
    matlab_dbg.redo_cap = 0;
}

/* Stamp a statement-boundary record. The hook calls this on every
 * fire so stepBack knows where each statement began. The frame
 * depth (n_frames at the time of the stamp) is stored in
 * `frame_idx` so stepBack can refuse to walk past a boundary that
 * crossed a function call — the user expects "step back" to stay
 * within the current frame (the language-level debugger contract),
 * not silently teleport into the caller. */
static void matlab_dbg_undo_record_stmt_locked(int32_t file_id,
                                                int32_t line,
                                                int32_t thread_slot) {
    if (!matlab_dbg.recording_undo) return;
    struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
    r->kind = 0;
    r->file_id = file_id;
    r->line = line;
    r->thread_slot = thread_slot;
    r->frame_idx = matlab_dbg.thread_n_frames[thread_slot];
}

/* Stamp an irreversible-op marker. The set_error path and the
 * disp/fprintf JIT entries call this so a stepBack that reaches
 * the marker stops with a clear message instead of silently
 * walking past a printed line. */
void matlab_dbg_undo_record_irreversible(const char *reason) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.recording_undo) {
        struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
        r->kind = 4;
        if (reason) {
            int64_t L = (int64_t)strlen(reason);
            r->name = (char *)malloc((size_t)L + 1);
            if (r->name) {
                memcpy(r->name, reason, (size_t)L + 1);
                r->name_len = L;
            }
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Capture current value of `name` from matlab_ws before it gets
 * overwritten. Returns prev_kind (-1 if missing), prev_f64,
 * prev_ptr. Used by the undo path on every ws_set_*. CALLER MUST
 * HOLD matlab_dbg.mu — matlab_struct accesses are themselves
 * lock-free, but we want the snapshot atomic w.r.t. the upcoming
 * write. */
static void matlab_ws_capture_prior(const char *name, int64_t len,
                                     int8_t *out_kind, int8_t *out_existed,
                                     double *out_f64, void **out_ptr) {
    *out_kind = -1; *out_existed = 0; *out_f64 = 0.0; *out_ptr = NULL;
    if (!matlab_ws) return;
    /* Walk the matlab_struct in-place — no public accessor returns
     * the kind alongside the value efficiently, so reach into the
     * struct directly. */
    for (int32_t i = 0; i < matlab_ws->nfields; ++i) {
        int nl = (int)strlen(matlab_ws->names[i]);
        if (nl == (int)len &&
            memcmp(matlab_ws->names[i], name, (size_t)len) == 0) {
            *out_existed = 1;
            *out_kind = (int8_t)matlab_ws->kinds[i];
            *out_f64 = matlab_ws->f64_vals[i];
            *out_ptr = matlab_ws->ptr_vals[i];
            return;
        }
    }
}

/* Workspace serialization. matlab_ws_set_* / matlab_ws_get_*
 * bracket every matlab_struct read/mutation on matlab_ws with
 * this pair so concurrent parfor workers can't race on names[],
 * nfields, or struct_grow_if_needed's realloc. Reusing
 * matlab_dbg.mu (rather than a separate ws mutex) keeps the
 * lock-order graph trivial: dbg.mu is the only ws lock, no
 * nesting needed. The undo-rewind path in matlab_dbg_step_back
 * already mutates ws while holding matlab_dbg.mu, so it gets the
 * same protection for free without changes. */
static void matlab_ws_lock(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
}
static void matlab_ws_unlock(void) {
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Helpers for the matlab_ws_set_* / matlab_dbg_frame_set_* sites
 * to backfill the post-write state into the record returned by
 * push_undo_locked. NULL-safe because push_undo_locked returns
 * NULL when recording is off — the call sites stay branch-free. */
static void matlab_dbg_undo_record_set_new_f64(
    struct matlab_dbg_undo_rec *r, double v) {
    if (!r) return;
    r->new_kind = 0;
    r->new_f64 = v;
    r->new_ptr = NULL;
}

static void matlab_dbg_undo_record_set_new_ptr(
    struct matlab_dbg_undo_rec *r, int new_kind, void *p) {
    if (!r) return;
    r->new_kind = (int8_t)new_kind;
    r->new_f64 = 0.0;
    r->new_ptr = p;
}

/* Push a ws_set undo record. CALLER MUST HOLD matlab_dbg.mu — the
 * matlab_ws_set_* sites take the workspace lock (which is the
 * same mutex) around the full push-undo + struct mutation pair
 * so concurrent parfor workers can't race on matlab_ws while one
 * is in struct_grow_if_needed. The fast path is a single
 * recording_undo check; if recording is off (no DAP session), the
 * function returns immediately.
 *
 * Returns the freshly allocated record so the caller can fill in
 * `new_kind`/`new_f64`/`new_ptr` after the actual write — the
 * redo path uses those to replay forward through the log without
 * needing to resume the JIT thread (which is parked one statement
 * ahead of the rewound caret). Returns NULL when recording is off
 * so callers don't have to nil-guard themselves; the field-fill
 * path checks for NULL. */
static struct matlab_dbg_undo_rec *
matlab_ws_push_undo_locked(const char *name, int64_t len,
                            int kind_being_written) {
    if (!matlab_dbg.recording_undo) return NULL;
    int8_t prev_kind, prev_existed;
    double prev_f64;
    void *prev_ptr;
    matlab_ws_capture_prior(name, len, &prev_kind, &prev_existed,
                             &prev_f64, &prev_ptr);
    struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
    /* Kind 1 for f64 writes, 2 for mat/obj writes — the rewind
     * path uses this to pick the right matlab_ws_set_* on undo. */
    r->kind = (kind_being_written == 0) ? 1 : 2;
    r->prev_kind = prev_kind;
    r->prev_existed = prev_existed;
    r->prev_f64 = prev_f64;
    r->prev_ptr = prev_ptr;
    r->name = (char *)malloc((size_t)len + 1);
    if (r->name) {
        memcpy(r->name, name, (size_t)len);
        r->name[len] = '\0';
        r->name_len = len;
    }
    return r;
}

/* Frame-local undo: capture prior entry from the named frame
 * (innermost of the calling thread). Same shape as the ws helper
 * but operates on thread_frame_locals. CALLER HOLDS matlab_dbg.mu.
 * Returns the freshly allocated record so the caller can fill in
 * the new_* fields after the write (NULL when recording is off). */
static struct matlab_dbg_undo_rec *
matlab_dbg_frame_push_undo_locked(int thread_slot,
                                   int frame_idx,
                                   const char *name,
                                   int64_t len) {
    if (!matlab_dbg.recording_undo) return NULL;
    struct matlab_dbg_frame_locals *fl =
        &matlab_dbg.thread_frame_locals[thread_slot][frame_idx];
    int8_t prev_kind = -1, prev_existed = 0;
    double prev_f64 = 0.0;
    void *prev_ptr = NULL;
    for (int i = 0; i < fl->n; ++i) {
        if (fl->entries[i].name_len == len &&
            memcmp(fl->entries[i].name, name, (size_t)len) == 0) {
            prev_existed = 1;
            prev_kind = (int8_t)fl->entries[i].kind;
            prev_f64 = fl->entries[i].f64;
            prev_ptr = fl->entries[i].ptr;
            break;
        }
    }
    struct matlab_dbg_undo_rec *r = matlab_dbg_undo_alloc_locked();
    r->kind = 3;
    r->thread_slot = thread_slot;
    r->frame_idx = frame_idx;
    r->prev_kind = prev_kind;
    r->prev_existed = prev_existed;
    r->prev_f64 = prev_f64;
    r->prev_ptr = prev_ptr;
    r->name = (char *)malloc((size_t)len + 1);
    if (r->name) {
        memcpy(r->name, name, (size_t)len);
        r->name[len] = '\0';
        r->name_len = len;
    }
    return r;
}

/* Rewind one statement: pop undo records starting from undo_head
 * back until a kind=0 (statement boundary) is found, applying each
 * write in reverse order. Stops at an irreversible-op marker
 * (kind=4) without rewinding past it. Returns the line number to
 * resume at, or 0 if the log is exhausted / the next record is an
 * irreversible op.
 *
 * After rewinding, the next forward-step will re-execute the
 * statement we just rolled back. The undo log itself is NOT
 * cleared — the records we just popped are gone (head moved
 * back), but anything older stays in case the user wants to
 * stepBack again. */
int matlab_dbg_step_back(int32_t *out_file_id, int32_t *out_line,
                         char *out_msg, int64_t msg_cap) {
    if (out_file_id) *out_file_id = 0;
    if (out_line) *out_line = 0;
    if (out_msg && msg_cap > 0) out_msg[0] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.n_undo == 0) {
        if (out_msg && msg_cap > 0)
            snprintf(out_msg, (size_t)msg_cap, "undo log is empty");
        pthread_mutex_unlock(&matlab_dbg.mu);
        return 0;
    }
    /* Snapshot the future-cap on the first stepBack of a sequence.
     * Subsequent rewind-then-rewind preserves it so a later
     * forward-step can redo all the way back to the JIT's parked
     * position. The cap is cleared on the next JIT resume (when
     * the user runs forward past the recorded future), via
     * matlab_dbg_undo_clear_locked or the redo-caught-up branch. */
    if (!matlab_dbg.rewound) {
        matlab_dbg.redo_cap = matlab_dbg.undo_head;
    }
    /* Algorithm:
     *   1. Drop the head boundary record (the current "now" marker
     *      from the most recent hook fire — represents the line the
     *      worker is paused at).
     *   2. Walk back, applying each non-boundary record's revert.
     *   3. Stop at the next boundary; that's the new "now". The
     *      boundary itself stays in the log (head = boundary_idx +
     *      1) so subsequent stepBacks see it as their head and drop
     *      it on entry.
     *
     * `head` points one PAST the last written entry, so the actual
     * top is at head-1. Wrap negative-mod the same way the ring
     * buffer wraps positive. */
    int idx = matlab_dbg.undo_head;
    int wall = matlab_dbg.undo_full ? 4096 : matlab_dbg.undo_head;
    int popped = 0;
    /* Disable recording so any matlab_struct_* writes we issue here
     * to roll values back don't push fresh undo records. */
    matlab_dbg.recording_undo = 0;
    int hit_boundary = 0;
    int hit_irreversible = 0;
    int32_t boundary_file_id = 0, boundary_line = 0;
    int boundary_idx = -1;
    /* Capture the paused thread's current frame depth — stepBack
     * is a same-frame operation. Boundary records carry the depth
     * at stamp time in `frame_idx`; we accept only matches.
     * Boundaries from inside callee frames (deeper) were stamped
     * during the recursive descent and shouldn't pull us into
     * those frames; boundaries from caller frames (shallower)
     * mean we walked off the front of the current function — we
     * stop with hit_boundary=0 so the IDE renders the rewind as
     * "exhausted within this frame" instead of teleporting up. */
    int paused_thread = matlab_dbg.paused_thread_idx;
    int target_depth = (paused_thread >= 0 && paused_thread < 32)
                       ? matlab_dbg.thread_n_frames[paused_thread] : 0;

    /* Step 1: drop the current "now" boundary. The hook stamps a
     * boundary on every fire, so head-1 IS that stamp when paused
     * at a fresh statement (the typical case). If the top isn't a
     * boundary — e.g. the bp fired mid-statement after some
     * writes — we keep walking; the writes get reverted normally
     * and the boundary we hit is the previous statement's. */
    {
        int peek = (idx == 0) ? 4096 - 1 : idx - 1;
        struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[peek];
        if (r->kind == 0) {
            idx = peek;
            ++popped;
        }
    }

    while (popped < wall) {
        idx = (idx == 0) ? 4096 - 1 : idx - 1;
        struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[idx];
        ++popped;
        if (r->kind == 4) {
            /* Irreversible op marker — stop here. The user has
             * to live with the prior printed output. Don't pop
             * the marker; leave it so a second stepBack also
             * stops here, not behind it. */
            hit_irreversible = 1;
            if (out_msg && msg_cap > 0) {
                if (r->name)
                    snprintf(out_msg, (size_t)msg_cap,
                             "can't reverse past: %s", r->name);
                else
                    snprintf(out_msg, (size_t)msg_cap,
                             "can't reverse past an irreversible operation");
            }
            ++idx;
            if (idx >= 4096) idx = 0;
            --popped;
            break;
        }
        if (r->kind == 0) {
            /* Statement boundary candidate. Match against the
             * paused thread's current frame depth so stepBack
             * doesn't cross function-call frames. Cases:
             *   - depth == target: same frame, this is our stop.
             *   - depth >  target: deeper frame (a callee that
             *     ran during the previous statement, like
             *     `disp(fact(4))` calling fact). Skip past the
             *     whole nested call by continuing to walk back
             *     until we find a same-depth boundary again.
             *   - depth <  target: caller frame. We walked off
             *     the front of the current function. Stop with
             *     hit_boundary = 0 (treated as "log exhausted
             *     within this frame") rather than teleport up.
             * Records with frame_idx == 0 (legacy from before
             * the depth tracking) are accepted as same-frame —
             * back-compat for any pre-existing boundary stamps. */
            int rec_depth = r->frame_idx;
            if (rec_depth != 0 && rec_depth > target_depth) {
                /* Deeper frame — keep walking back. */
                continue;
            }
            if (rec_depth != 0 && rec_depth < target_depth) {
                /* Shallower (caller) frame — refuse to cross. */
                ++idx;
                if (idx >= 4096) idx = 0;
                --popped;
                break;
            }
            /* Same frame: this is the new "now". Keep it in the
             * log so the next stepBack drops it (per step 1). */
            hit_boundary = 1;
            boundary_idx = idx;
            boundary_file_id = r->file_id;
            boundary_line = r->line;
            /* Advance idx past the boundary so head ends up just
             * after it; the boundary record stays in place. */
            ++idx;
            if (idx >= 4096) idx = 0;
            --popped;  /* don't count the kept boundary as popped */
            break;
        }
        /* Apply the undo: revert the write described by this record. */
        if (r->kind == 1 || r->kind == 2) {
            /* matlab_ws revert. If the variable existed before,
             * restore the previous (kind, value/ptr); if it
             * didn't, remove the binding entirely via
             * matlab_struct_rmfield so the rewound state matches
             * the pre-write workspace exactly (no stale "x = 0"
             * shadow). */
            if (r->prev_existed) {
                if (r->prev_kind == 0) {
                    matlab_struct_set_f64(matlab_ws, r->name, r->name_len,
                                           r->prev_f64);
                } else if (r->prev_kind == 1) {
                    matlab_struct_set_mat(matlab_ws, r->name, r->name_len,
                                           (matlab_mat *)r->prev_ptr);
                } else if (r->prev_kind == 2 || r->prev_kind == 3) {
                    int32_t i = struct_reserve(matlab_ws, r->name,
                                                (int32_t)r->name_len);
                    matlab_ws->kinds[i] = r->prev_kind;
                    matlab_ws->f64_vals[i] = 0.0;
                    matlab_ws->ptr_vals[i] = r->prev_ptr;
                }
            } else {
                /* Variable didn't exist before — remove the
                 * binding so `who` / `whos` / DAP variable
                 * inspection see the pre-write state. */
                matlab_struct_rmfield(matlab_ws, r->name, r->name_len);
            }
        } else if (r->kind == 3) {
            /* frame_local revert. Walk the entries[] of the
             * stamped frame and reset the named entry. If the
             * variable didn't exist pre-write, drop it from the
             * table (last-entry-swap) so subsequent reads miss. */
            int t = r->thread_slot;
            int f = r->frame_idx;
            if (t >= 0 && t < 32 && f >= 0 && f < MATLAB_DBG_MAX_FRAMES) {
                struct matlab_dbg_frame_locals *fl =
                    &matlab_dbg.thread_frame_locals[t][f];
                int found = -1;
                for (int i = 0; i < fl->n; ++i) {
                    if (fl->entries[i].name_len == r->name_len &&
                        memcmp(fl->entries[i].name, r->name,
                                (size_t)r->name_len) == 0) {
                        found = i; break;
                    }
                }
                if (r->prev_existed) {
                    if (found >= 0) {
                        fl->entries[found].kind = r->prev_kind;
                        fl->entries[found].f64 = r->prev_f64;
                        fl->entries[found].ptr = r->prev_ptr;
                    }
                } else if (found >= 0) {
                    /* Variable didn't exist before — drop it. */
                    free(fl->entries[found].name);
                    fl->entries[found] = fl->entries[fl->n - 1];
                    fl->n--;
                }
            }
        }
        /* The popped record's name copy is freed when its slot is
         * later overwritten by undo_alloc; no free here. */
    }
    /* Move head back by `popped` (the boundary we hit stays in the
     * log because we decremented popped before breaking). */
    (void)boundary_idx;  /* used only for the assertion in builds with -DDBG */
    matlab_dbg.undo_head = idx;
    matlab_dbg.n_undo -= popped;
    if (matlab_dbg.n_undo < 0) matlab_dbg.n_undo = 0;
    /* If we walked the whole buffer and never hit a boundary or
     * irreversible op, the rewind is best-effort — the IDE got
     * its values rolled back but no line to resume at. Treat as
     * "nothing more to rewind". */
    matlab_dbg.recording_undo = 1;
    /* Mark rewound state on success so the next forward step
     * (next/stepIn/continue) goes through matlab_dbg_step_forward_redo
     * instead of resuming the JIT — see redo_cap above. We tag
     * this even when hit_boundary==0 (rewind exhausted within the
     * frame) so a forward step still walks the redo log instead
     * of jumping the JIT ahead. The flag clears the moment the
     * user redoes back to the JIT's parked position. */
    if (hit_boundary || popped > 0) {
        matlab_dbg.rewound = 1;
    }
    /* Update the innermost frame's (file_id, line) to the rewound
     * boundary so DAP `stackTrace` reflects the new position.
     * Without this, the per-thread chain still has the old line
     * from the original hook fire, the inspector renders the
     * caret there, and the user sees no visible movement even
     * though `cur_line` and the `stopped` event line are correct.
     *
     * Done BEFORE the shared-frames snapshot below so the
     * snapshot picks up the rewound line. */
    int p = matlab_dbg.paused_thread_idx;
    if (hit_boundary && p >= 0 && p < 32) {
        int n_thr = matlab_dbg.thread_n_frames[p];
        if (n_thr > 0) {
            matlab_dbg.thread_frames[p][n_thr - 1].file_id = boundary_file_id;
            matlab_dbg.thread_frames[p][n_thr - 1].line = boundary_line;
        }
    }
    /* Refresh shared frames[] from the paused thread so DAP
     * inspectors see the rewound view. */
    if (p >= 0 && p < 32) {
        int n = matlab_dbg.thread_n_frames[p];
        if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
        matlab_dbg.n_frames = n;
        for (int i = 0; i < n; ++i) {
            matlab_dbg.frames[i] = matlab_dbg.thread_frames[p][i];
            matlab_dbg.frame_locals[i] =
                matlab_dbg.thread_frame_locals[p][i];
        }
    }
    if (hit_boundary) {
        if (out_file_id) *out_file_id = boundary_file_id;
        if (out_line) *out_line = boundary_line;
        matlab_dbg.cur_file_id = boundary_file_id;
        matlab_dbg.cur_line = boundary_line;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return hit_irreversible ? -1 : (hit_boundary ? 1 : 0);
}

/* Forward decl for the helper that lives further down — the
 * redo path below needs it but it's defined alongside the rest
 * of the frame-local machinery. */
static int matlab_dbg_frame_local_find_or_alloc_in(
    struct matlab_dbg_frame_locals *fl, const char *name, int64_t name_len);

/* DAP-server query: is the runtime currently in a rewound state
 * (caret behind the JIT's parked position)? While true, the
 * server routes forward steps through matlab_dbg_step_forward_redo
 * instead of waking the JIT — otherwise the resumed JIT would
 * execute one statement past the rewound caret and the user
 * sees a "skipped" line. Cleared automatically when redo catches
 * up to the recorded future_cap. */
int matlab_dbg_is_rewound(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int r = matlab_dbg.rewound;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return r;
}

/* Walk the undo log forward from undo_head, applying each
 * record's post-write state, until we reach a same-frame
 * statement boundary or catch up to redo_cap.
 *
 * Return values mirror matlab_dbg_step_back:
 *    1  = landed on a boundary; out_file_id / out_line carry it.
 *    0  = caught up to redo_cap (rewound cleared, JIT should
 *         be resumed normally for the next forward step).
 *   -1  = hit an irreversible-op marker (kind=4) — the redo
 *         path can't replay past disp/fprintf side effects, so
 *         we stop with the message in out_msg.
 *
 * The runtime side-effects mirror step_back's tail: cur_file_id
 * / cur_line, the paused-thread innermost frame's line, and the
 * shared frames[] snapshot all get updated so DAP inspectors see
 * the redo'd view consistently. */
int matlab_dbg_step_forward_redo(int32_t *out_file_id, int32_t *out_line,
                                  char *out_msg, int64_t msg_cap) {
    if (out_file_id) *out_file_id = 0;
    if (out_line) *out_line = 0;
    if (out_msg && msg_cap > 0) out_msg[0] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    if (!matlab_dbg.rewound) {
        pthread_mutex_unlock(&matlab_dbg.mu);
        return 0; /* nothing to redo; caller falls through to JIT resume */
    }
    int idx = matlab_dbg.undo_head;
    int cap = matlab_dbg.redo_cap;
    /* Don't let the apply-new writes push fresh undo records;
     * we're walking *over* existing records, not creating new
     * ones. Same trick step_back uses on the rewind path. */
    matlab_dbg.recording_undo = 0;
    int hit_boundary = 0;
    int hit_irreversible = 0;
    int32_t boundary_file_id = 0, boundary_line = 0;
    int paused_thread = matlab_dbg.paused_thread_idx;
    int target_depth = (paused_thread >= 0 && paused_thread < 32)
                       ? matlab_dbg.thread_n_frames[paused_thread] : 0;
    /* Walk forward up to redo_cap. step_back leaves the prior
     * caret's boundary record IMMEDIATELY BEHIND undo_head (it's
     * kept in the log so the next stepBack can drop it as its
     * own "now" marker on entry). So undo_head itself is the
     * first slot of the recorded *future* — typically a write
     * record — and walking forward, the first same-frame
     * boundary we hit IS the new caret. After matching, we
     * advance idx past that boundary so the same step_back
     * invariant holds: undo_head sits just after the boundary
     * representing the current caret. */
    while (idx != cap) {
        struct matlab_dbg_undo_rec *r = &matlab_dbg.undo_log[idx];
        int next_idx = (idx + 1) % 4096;
        if (r->kind == 0) {
            int rec_depth = r->frame_idx;
            if (rec_depth != 0 && rec_depth > target_depth) {
                /* Deeper frame: a callee-side boundary recorded
                 * during a function call from the previous
                 * statement. Skip past the whole nested call
                 * until we find a same-depth boundary. */
                idx = next_idx;
                continue;
            }
            if (rec_depth != 0 && rec_depth < target_depth) {
                /* Shallower frame: the recorded future left the
                 * current function. Refuse to cross — same
                 * contract as step_back going the other way.
                 * Treat as exhausted within this frame. */
                break;
            }
            /* Same-frame boundary: this is the new caret. Move
             * undo_head past it (same as step_back's keep-the-
             * boundary contract). */
            hit_boundary = 1;
            boundary_file_id = r->file_id;
            boundary_line = r->line;
            idx = next_idx;
            break;
        }
        if (r->kind == 4) {
            hit_irreversible = 1;
            if (out_msg && msg_cap > 0) {
                if (r->name)
                    snprintf(out_msg, (size_t)msg_cap,
                             "can't redo past: %s", r->name);
                else
                    snprintf(out_msg, (size_t)msg_cap,
                             "can't redo past an irreversible operation");
            }
            break;
        }
        /* Re-apply the new state captured at the original write. */
        if (r->kind == 1 || r->kind == 2) {
            if (r->new_kind == 0) {
                matlab_struct_set_f64(matlab_ws, r->name, r->name_len,
                                       r->new_f64);
            } else if (r->new_kind == 1) {
                matlab_struct_set_mat(matlab_ws, r->name, r->name_len,
                                       (matlab_mat *)r->new_ptr);
            } else if (r->new_kind == 2 || r->new_kind == 3) {
                int32_t i = struct_reserve(matlab_ws, r->name,
                                             (int32_t)r->name_len);
                matlab_ws->kinds[i] = r->new_kind;
                matlab_ws->f64_vals[i] = 0.0;
                matlab_ws->ptr_vals[i] = r->new_ptr;
            }
        } else if (r->kind == 3) {
            int t = r->thread_slot;
            int f = r->frame_idx;
            if (t >= 0 && t < 32 && f >= 0 && f < MATLAB_DBG_MAX_FRAMES) {
                struct matlab_dbg_frame_locals *fl =
                    &matlab_dbg.thread_frame_locals[t][f];
                int found = matlab_dbg_frame_local_find_or_alloc_in(
                    fl, r->name, r->name_len);
                if (found >= 0) {
                    fl->entries[found].kind = (int32_t)r->new_kind;
                    fl->entries[found].f64 = r->new_f64;
                    fl->entries[found].ptr = r->new_ptr;
                }
            }
        }
        idx = next_idx;
    }
    matlab_dbg.undo_head = idx;
    matlab_dbg.recording_undo = 1;
    /* Recount n_undo from the ring state (head + full flag) so
     * the next stepBack's `n_undo == 0` early-out reads the
     * post-redo size, not the rewind-time size. */
    if (matlab_dbg.undo_full) matlab_dbg.n_undo = 4096;
    else matlab_dbg.n_undo = matlab_dbg.undo_head;
    /* If we walked all the way to redo_cap, we're caught up
     * with the JIT's parked position — clear rewound so the
     * caller falls through to a normal JIT resume. */
    int caught_up = (idx == cap);
    if (caught_up) {
        matlab_dbg.rewound = 0;
        matlab_dbg.redo_cap = 0;
    }
    /* Mirror step_back's caret update: write the new line into
     * the paused thread's innermost frame so DAP stackTrace
     * reflects it. */
    int p = matlab_dbg.paused_thread_idx;
    if (hit_boundary && p >= 0 && p < 32) {
        int n_thr = matlab_dbg.thread_n_frames[p];
        if (n_thr > 0) {
            matlab_dbg.thread_frames[p][n_thr - 1].file_id = boundary_file_id;
            matlab_dbg.thread_frames[p][n_thr - 1].line = boundary_line;
        }
    }
    if (p >= 0 && p < 32) {
        int n = matlab_dbg.thread_n_frames[p];
        if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
        matlab_dbg.n_frames = n;
        for (int i = 0; i < n; ++i) {
            matlab_dbg.frames[i] = matlab_dbg.thread_frames[p][i];
            matlab_dbg.frame_locals[i] =
                matlab_dbg.thread_frame_locals[p][i];
        }
    }
    if (hit_boundary) {
        if (out_file_id) *out_file_id = boundary_file_id;
        if (out_line) *out_line = boundary_line;
        matlab_dbg.cur_file_id = boundary_file_id;
        matlab_dbg.cur_line = boundary_line;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (hit_irreversible) return -1;
    if (hit_boundary) return 1;
    return 0; /* caught up — caller resumes the JIT */
}

static void matlab_dbg_watch_trip(int wp_idx) {
    int thr_idx = matlab_dbg_thread_init_chain_locked();
    int n_thr = matlab_dbg.thread_n_frames[thr_idx];
    if (n_thr > 0) {
        matlab_dbg.cur_file_id =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].file_id;
        matlab_dbg.cur_line =
            matlab_dbg.thread_frames[thr_idx][n_thr - 1].line;
    }
    /* Snapshot calling thread's chain into shared frames[] for
     * DAP inspectors. Same trick as matlab_dbg_hook on pause. */
    int snap_n = n_thr > MATLAB_DBG_MAX_FRAMES ? MATLAB_DBG_MAX_FRAMES : n_thr;
    matlab_dbg.n_frames = snap_n;
    for (int i = 0; i < snap_n; ++i) {
        matlab_dbg.frames[i] = matlab_dbg.thread_frames[thr_idx][i];
        matlab_dbg.frame_locals[i] =
            matlab_dbg.thread_frame_locals[thr_idx][i];
    }
    matlab_dbg.cur_bp_idx = -1;
    matlab_dbg.last_wp_idx = wp_idx;
    matlab_dbg.paused = 1;
    matlab_dbg.paused_from_watch = 1;
    matlab_dbg.paused_thread_idx = thr_idx;
    pthread_cond_broadcast(&matlab_dbg.cv_server);
    while (matlab_dbg.paused) {
        pthread_cond_wait(&matlab_dbg.cv_client, &matlab_dbg.mu);
    }
    matlab_dbg.paused_from_watch = 0;
    matlab_dbg.last_wp_idx = -1;
    matlab_dbg.paused_thread_idx = -1;
}

/* Lazy thread registration. Called from the hook on every entry —
 * fast path is a constant-time scan of the (small) thread_keys
 * table. New thread → append + assign sequential id. CALLER MUST
 * HOLD matlab_dbg.mu (the hook already does). Returns the slot
 * index in the threads table; the DAP-facing thread id is
 * thread_ids[idx]. */
static int matlab_dbg_thread_slot_locked(void) {
    pthread_t self = pthread_self();
    for (int i = 0; i < matlab_dbg.n_threads; ++i) {
        if (pthread_equal(matlab_dbg.thread_keys[i], self)) return i;
    }
    if (matlab_dbg.n_threads >= 32) {
        /* Table full — reuse slot 0 (main worker). Means the
         * 33rd parfor worker borrows the main worker's id. Better
         * than refusing to track and breaking the hook entirely. */
        return 0;
    }
    int idx = matlab_dbg.n_threads++;
    matlab_dbg.thread_keys[idx] = self;
    /* Sequential id starting at 1; thread 1 is the main worker
     * registered on its first hook fire. Matches the DAP server's
     * pre-existing assumption that threadId 1 is "main". */
    matlab_dbg.thread_ids[idx] = idx + 1;
    return idx;
}

/* DAP-side enumeration: total registered threads. */
int matlab_dbg_thread_count(void) {
    int n;
    pthread_mutex_lock(&matlab_dbg.mu);
    n = matlab_dbg.n_threads;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

/* DAP-side: thread id at index. Returns 0 on out-of-range. */
int32_t matlab_dbg_thread_id_at(int idx) {
    int32_t id = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_threads)
        id = matlab_dbg.thread_ids[idx];
    pthread_mutex_unlock(&matlab_dbg.mu);
    return id;
}

/* DAP-side: id of the thread that triggered the current pause, or
 * 0 if no pause is active. */
int32_t matlab_dbg_paused_thread_id(void) {
    int32_t id = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    int idx = matlab_dbg.paused_thread_idx;
    if (idx >= 0 && idx < matlab_dbg.n_threads)
        id = matlab_dbg.thread_ids[idx];
    pthread_mutex_unlock(&matlab_dbg.mu);
    return id;
}

/* Body of the matlab_ws_set_* watchpoint helper — forward-declared
 * up by the matlab_ws_set_* sites where matlab_dbg state isn't yet
 * in scope. The write has already landed when this fires, so the
 * IDE inspecting the variable on pause sees the new value (matches
 * gdb's "old/new" model where the new value is visible at the stop). */
static void matlab_ws_check_watch(const char *name, int64_t len) {
    if (!name || len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int idx = matlab_dbg_watch_check(name, len, /*scope_hint=*/1);
        if (idx >= 0) matlab_dbg_watch_trip(idx);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Read-side counterpart. Fast path: when n_wp is 0 (no watches at
 * all) the n_wp check fails outside the lock, so we bail without
 * paying mutex cost. The full check happens only when read-watches
 * are armed.
 *
 * Note: scope_hint is hardcoded to 1 (script-ws) because the only
 * read-path call sites are matlab_ws_get_*. Frame-local reads in
 * user code go through stack slots and never call into this API,
 * so they aren't visible to read-watches. The DAP `setDataBreakpoints`
 * handler advertises this limitation in its accessTypes. */
static void matlab_ws_check_read_watch(const char *name, int64_t len) {
    if (!name || len <= 0) return;
    /* Lock-free fast path. matlab_dbg.n_wp is an `int` and the
     * worst case of a torn read is at most a one-statement delay
     * before the watch fires — preferable to taking the global
     * mutex on every JIT-emitted ws_get_* call. */
    if (matlab_dbg.n_wp == 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int idx = matlab_dbg_watch_check_read(name, len, /*scope_hint=*/1);
        if (idx >= 0) matlab_dbg_watch_trip(idx);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Called from the server thread to enable the hook and set the
 * stop-on-entry mode before the worker starts. */
void matlab_dbg_enable(int stop_on_entry) {
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.enabled = 1;
    matlab_dbg.stop_on_entry = stop_on_entry ? 1 : 0;
    matlab_dbg.action = stop_on_entry ? MATLAB_DBG_STEP_IN : MATLAB_DBG_RUN;
    matlab_dbg.n_frames = 1;
    matlab_dbg.frames[0].file_id = 0;
    matlab_dbg.frames[0].line = 0;
    matlab_dbg.frames[0].fn_name = "<script>";
    matlab_dbg.last_wp_idx = -1;
    matlab_dbg.paused_thread_idx = -1;
    /* Turn on undo recording for reverse-stepping. The hook stamps
     * a stmt-boundary record on every fire; ws_set_* / frame_set_*
     * push prev-value records before each write. Clearing the log
     * here ensures a re-launch starts with an empty undo history. */
    matlab_dbg_undo_clear_locked();
    matlab_dbg.recording_undo = 1;
    /* Reset the thread registry on every launch so DAP threadIds
     * start fresh — a re-launch otherwise carries stale entries
     * from the prior session into the IDE's threads pane. Per-
     * thread frame chains and Locals are cleared in lockstep. */
    matlab_dbg.n_threads = 0;
    /* Clear any stale Locals captured during a previous launch
     * (both the legacy shared frame_locals[] and every per-thread
     * slot). dbg state is process-static and DAP can re-launch. */
    for (int i = 0; i < MATLAB_DBG_MAX_FRAMES; ++i)
        matlab_dbg_free_frame_locals(i);
    for (int t = 0; t < 32; ++t) {
        matlab_dbg.thread_n_frames[t] = 0;
        matlab_dbg.thread_step_target_depth[t] = 0;
        for (int i = 0; i < MATLAB_DBG_MAX_FRAMES; ++i) {
            struct matlab_dbg_frame_locals *fl =
                &matlab_dbg.thread_frame_locals[t][i];
            for (int e = 0; e < fl->n; ++e) free(fl->entries[e].name);
            fl->n = 0;
            /* Also free any heap-owned fn_name on stale frames so a
             * re-launch doesn't leak the prior session's strings. */
            char *owned =
                (char *)matlab_dbg.thread_frames[t][i].fn_name;
            if (owned && i > 0) {
                /* Frame 0 is the literal "<script>" — never freed. */
                free(owned);
            }
            matlab_dbg.thread_frames[t][i].fn_name = NULL;
            matlab_dbg.thread_frames[t][i].file_id = 0;
            matlab_dbg.thread_frames[t][i].line = 0;
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Register (file_id -> filename) so the DAP server can resolve
 * breakpoints by file path. Called once per source file before the
 * debuggee starts. file_id is 1-based; we store 0-based. */
void matlab_dbg_register_file(int32_t file_id,
                               const char *name, int64_t name_len) {
    if (file_id <= 0 || file_id > (int32_t)(sizeof matlab_dbg.file_names /
                                              sizeof matlab_dbg.file_names[0]))
        return;
    /* Copy the name so we own it. */
    char *copy = (char *)malloc((size_t)name_len + 1);
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.file_names[file_id - 1] = copy;
    matlab_dbg.file_name_lens[file_id - 1] = name_len;
    if (file_id > matlab_dbg.n_files) matlab_dbg.n_files = file_id;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Register (class_id -> class-name) so the DAP server can render a
 * class instance as `1x1 ClassName` instead of falling back to the
 * matrix shape (which read garbage off the obj struct). Called once
 * per classdef from the lowered script entry when -g is on. The
 * registration is idempotent — re-registering the same class_id
 * overwrites the existing entry, which keeps the path safe under
 * repeated launches in long-lived DAP sessions. The string is heap-
 * copied here and freed at process exit (i.e. never — small and
 * bounded by the number of distinct classdefs in the program). */
void matlab_dbg_register_class(int32_t class_id,
                                const char *name, int64_t name_len) {
    if (class_id <= 0 || !name || name_len <= 0) return;
    char *copy = (char *)malloc((size_t)name_len + 1);
    if (!copy) return;
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = -1;
    for (int i = 0; i < matlab_dbg.n_classes; ++i) {
        if (matlab_dbg.class_ids[i] == class_id) { slot = i; break; }
    }
    if (slot < 0) {
        int cap = (int)(sizeof matlab_dbg.class_ids /
                        sizeof matlab_dbg.class_ids[0]);
        if (matlab_dbg.n_classes < cap) {
            slot = matlab_dbg.n_classes++;
            matlab_dbg.class_ids[slot] = class_id;
            matlab_dbg.class_names[slot] = NULL;
        }
    }
    if (slot >= 0) {
        free(matlab_dbg.class_names[slot]);
        matlab_dbg.class_names[slot] = copy;
        matlab_dbg.class_name_lens[slot] = name_len;
    } else {
        free(copy);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Look up the class name registered for a given class_id. Returns
 * NULL if the class hasn't been registered (DebugMode off, or a
 * built-in struct slipped through with kind=2 — defensive). */
const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out) {
    const char *name = NULL;
    int64_t len = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    for (int i = 0; i < matlab_dbg.n_classes; ++i) {
        if (matlab_dbg.class_ids[i] == class_id) {
            name = matlab_dbg.class_names[i];
            len  = matlab_dbg.class_name_lens[i];
            break;
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (len_out) *len_out = name ? len : 0;
    return name;
}

/* Property introspection on a matlab_obj. Used by the DAP server to
 * expand a class-instance row into one child per property. The obj
 * pointer is borrowed from the per-frame Locals table; reading
 * fields is lock-free (the mutating paths run on the debuggee
 * thread, which is paused while the server is reading). */
int matlab_dbg_obj_field_count(void *obj) {
    if (!obj) return 0;
    return ((matlab_obj *)obj)->nfields;
}

const char *matlab_dbg_obj_field_name(void *obj, int i, int64_t *len_out) {
    if (!obj || i < 0) { if (len_out) *len_out = 0; return NULL; }
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) { if (len_out) *len_out = 0; return NULL; }
    const char *n = o->names[i];
    if (len_out) *len_out = n ? (int64_t)strlen(n) : 0;
    return n;
}

int matlab_dbg_obj_field_kind(void *obj, int i) {
    if (!obj || i < 0) return -1;
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) return -1;
    return o->kinds[i];
}

double matlab_dbg_obj_field_f64(void *obj, int i) {
    if (!obj || i < 0) return 0.0;
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) return 0.0;
    return o->f64_vals[i];
}

void *matlab_dbg_obj_field_ptr(void *obj, int i) {
    if (!obj || i < 0) return NULL;
    matlab_obj *o = (matlab_obj *)obj;
    if (i >= o->nfields) return NULL;
    return o->ptr_vals[i];
}

int32_t matlab_dbg_obj_class_id_of(void *obj) {
    return obj ? ((matlab_obj *)obj)->class_id : 0;
}

/* Look up a registered filename by file_id. Returns NULL if unknown.
 * The returned pointer is valid for the lifetime of the process —
 * we own the heap copy made by matlab_dbg_register_file. Used by the
 * DAP server to resolve a paused frame's file_id back to a path so
 * stackTrace responses can reference the correct source. */
const char *matlab_dbg_file_name(int32_t file_id, int64_t *len_out) {
    const char *name = NULL;
    int64_t len = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (file_id >= 1 &&
        file_id <= (int32_t)(sizeof matlab_dbg.file_names /
                              sizeof matlab_dbg.file_names[0])) {
        name = matlab_dbg.file_names[file_id - 1];
        len = matlab_dbg.file_name_lens[file_id - 1];
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (len_out) *len_out = name ? len : 0;
    return name;
}

/* Called from the server thread. Returns the previous breakpoint
 * count for that file so the server can clear-and-reset atomically.
 * Simple: we wipe every breakpoint for that file then re-add. The
 * cond_text / log_text heap copies are freed before compaction so a
 * setBreakpoints replay doesn't leak. */
void matlab_dbg_clear_breakpoints_in_file(int32_t file_id) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int w = 0;
    for (int i = 0; i < matlab_dbg.n_bp; ++i) {
        if (matlab_dbg.bp_file[i] == file_id) {
            free(matlab_dbg.cond_text[i]);
            free(matlab_dbg.log_text[i]);
            continue;
        }
        matlab_dbg.bp_file[w] = matlab_dbg.bp_file[i];
        matlab_dbg.bp_line[w] = matlab_dbg.bp_line[i];
        matlab_dbg.cond_text[w] = matlab_dbg.cond_text[i];
        matlab_dbg.cond_len[w]  = matlab_dbg.cond_len[i];
        matlab_dbg.log_text[w]  = matlab_dbg.log_text[i];
        matlab_dbg.log_len[w]   = matlab_dbg.log_len[i];
        matlab_dbg.cond_disabled[w] = matlab_dbg.cond_disabled[i];
        matlab_dbg.hit_count[w]  = matlab_dbg.hit_count[i];
        matlab_dbg.hit_target[w] = matlab_dbg.hit_target[i];
        matlab_dbg.hit_op[w]     = matlab_dbg.hit_op[i];
        ++w;
    }
    /* Zero out the slots we evicted so subsequent _ex inserts don't
     * inherit a stale pointer the compaction loop just moved away.
     * Hit-count fields reset to 0 so a re-set bp counts from
     * scratch; otherwise repeated `setBreakpoints` round-trips
     * during a debug session would silently inherit prior counts. */
    for (int i = w; i < matlab_dbg.n_bp; ++i) {
        matlab_dbg.cond_text[i] = NULL; matlab_dbg.cond_len[i] = 0;
        matlab_dbg.log_text[i]  = NULL; matlab_dbg.log_len[i]  = 0;
        matlab_dbg.cond_disabled[i] = 0;
        matlab_dbg.hit_count[i] = 0;
        matlab_dbg.hit_target[i] = 0;
        matlab_dbg.hit_op[i] = 0;
    }
    matlab_dbg.n_bp = w;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

int matlab_dbg_add_breakpoint(int32_t file_id, int32_t line) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int ok = matlab_dbg.n_bp < MATLAB_DBG_MAX_BREAKPOINTS;
    if (ok) {
        int i = matlab_dbg.n_bp;
        matlab_dbg.bp_file[i] = file_id;
        matlab_dbg.bp_line[i] = line;
        matlab_dbg.cond_text[i] = NULL; matlab_dbg.cond_len[i] = 0;
        matlab_dbg.log_text[i]  = NULL; matlab_dbg.log_len[i]  = 0;
        matlab_dbg.cond_disabled[i] = 0;
        matlab_dbg.n_bp++;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Conditional / log-point-aware insert with optional hit-count
 * gate. Either text pointer may be NULL (with matching len = 0)
 * to mean "no condition" / "no log". hit_op == 0 disables the
 * hit-count gate. The runtime owns the heap copy so the server
 * can release its own buffers immediately after returning. */
int matlab_dbg_add_breakpoint_ex2(int32_t file_id, int32_t line,
                                   const char *cond, int64_t cond_len,
                                   const char *log,  int64_t log_len,
                                   int hit_op, int64_t hit_target) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int ok = matlab_dbg.n_bp < MATLAB_DBG_MAX_BREAKPOINTS;
    if (ok) {
        int i = matlab_dbg.n_bp;
        matlab_dbg.bp_file[i] = file_id;
        matlab_dbg.bp_line[i] = line;
        matlab_dbg.cond_text[i] = NULL; matlab_dbg.cond_len[i] = 0;
        matlab_dbg.log_text[i]  = NULL; matlab_dbg.log_len[i]  = 0;
        matlab_dbg.cond_disabled[i] = 0;
        matlab_dbg.hit_count[i] = 0;
        matlab_dbg.hit_target[i] = hit_target;
        matlab_dbg.hit_op[i] = hit_op;
        if (cond && cond_len > 0) {
            matlab_dbg.cond_text[i] = (char *)malloc((size_t)cond_len + 1);
            memcpy(matlab_dbg.cond_text[i], cond, (size_t)cond_len);
            matlab_dbg.cond_text[i][cond_len] = '\0';
            matlab_dbg.cond_len[i] = cond_len;
        }
        if (log && log_len > 0) {
            matlab_dbg.log_text[i] = (char *)malloc((size_t)log_len + 1);
            memcpy(matlab_dbg.log_text[i], log, (size_t)log_len);
            matlab_dbg.log_text[i][log_len] = '\0';
            matlab_dbg.log_len[i] = log_len;
        }
        matlab_dbg.n_bp++;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Backward-compat wrapper for the v1 _ex API (no hit-count gate). */
int matlab_dbg_add_breakpoint_ex(int32_t file_id, int32_t line,
                                  const char *cond, int64_t cond_len,
                                  const char *log,  int64_t log_len) {
    return matlab_dbg_add_breakpoint_ex2(file_id, line, cond, cond_len,
                                          log, log_len, 0, 0);
}

/* Snapshot the cond / log text for a given bp index. Caller-supplied
 * pointers receive runtime-owned strings that stay valid until the
 * next clear_breakpoints_in_file call. The disabled out-param is
 * non-zero when the condition was previously rejected (eval failed)
 * — callers should treat the bp as condition-less but still suppress
 * the pause to match VS Code's "broken condition is silent" UX.
 * Returns 0 on out-of-range. */
int matlab_dbg_breakpoint_meta(int idx, const char **cond, int64_t *cond_len,
                                const char **log, int64_t *log_len,
                                int *disabled) {
    int ok = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_bp) {
        if (cond)     *cond     = matlab_dbg.cond_text[idx];
        if (cond_len) *cond_len = matlab_dbg.cond_len[idx];
        if (log)      *log      = matlab_dbg.log_text[idx];
        if (log_len)  *log_len  = matlab_dbg.log_len[idx];
        if (disabled) *disabled = matlab_dbg.cond_disabled[idx];
        ok = 1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Iterate the runtime's active breakpoint table: returns (file_id,
 * line) for breakpoint `idx`, or 0 on out-of-range. The DAP
 * server's `reverseContinue` handler uses this to check whether a
 * rewound line matches any active bp — `breakpoint_meta` above
 * exposes the cond/log strings but not the source location, which
 * is the bit reverseContinue actually needs. */
int matlab_dbg_breakpoint_at(int idx, int32_t *file_id, int32_t *line) {
    int ok = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_bp) {
        if (file_id) *file_id = matlab_dbg.bp_file[idx];
        if (line)    *line    = matlab_dbg.bp_line[idx];
        ok = 1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

void matlab_dbg_disable_condition(int idx) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (idx >= 0 && idx < matlab_dbg.n_bp)
        matlab_dbg.cond_disabled[idx] = 1;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

int matlab_dbg_get_pause_bp(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int idx = matlab_dbg.cur_bp_idx;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return idx;
}

/* Called from the server thread after handling a stopped event.
 * Sets the next action and wakes the worker. */
void matlab_dbg_resume(int action) {
    pthread_mutex_lock(&matlab_dbg.mu);
    matlab_dbg.action = (enum matlab_dbg_action)action;
    /* Once the JIT actually runs forward, the recorded future is
     * about to be overwritten (or already was, via the redo
     * walk). Clear the rewound flag and reset redo_cap so the
     * next stepBack snapshots a fresh future. The redo path
     * already clears these on its own caught-up branch; this is
     * the belt-and-braces side for the cases where the DAP
     * server resumes the JIT directly without going through
     * redo (e.g., reverseContinue → next semantics). */
    matlab_dbg.rewound = 0;
    matlab_dbg.redo_cap = 0;
    /* Step targets are per-thread: a step in worker A must use
     * worker A's depth, not whatever the legacy shared
     * n_frames last got snapshotted to. We seed every thread's
     * target depth from the currently-paused thread's depth so
     * the resume kicks the right thread to the right place. */
    int paused = matlab_dbg.paused_thread_idx;
    if (paused >= 0 && paused < 32) {
        int n = matlab_dbg.thread_n_frames[paused];
        if (action == MATLAB_DBG_STEP_OVER)
            matlab_dbg.thread_step_target_depth[paused] = n;
        else if (action == MATLAB_DBG_STEP_OUT)
            matlab_dbg.thread_step_target_depth[paused] = n - 1;
    }
    /* Legacy single-thread fallback: keep updating the shared
     * step_target_depth so any unconverted single-threaded
     * stepping path still reads a sane value. */
    if (action == MATLAB_DBG_STEP_OVER)
        matlab_dbg.step_target_depth = matlab_dbg.n_frames;
    else if (action == MATLAB_DBG_STEP_OUT)
        matlab_dbg.step_target_depth = matlab_dbg.n_frames - 1;
    matlab_dbg.paused = 0;
    pthread_cond_broadcast(&matlab_dbg.cv_client);
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Called from the server thread to read the current pause point. */
void matlab_dbg_get_pause(int32_t *file_id, int32_t *line) {
    pthread_mutex_lock(&matlab_dbg.mu);
    *file_id = matlab_dbg.cur_file_id;
    *line = matlab_dbg.cur_line;
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Frame counts are published so the server can draw a stackTrace. */
int matlab_dbg_frame_count(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int n = matlab_dbg.n_frames;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

/* Snapshot frame i (0-based, 0 = innermost) into caller-supplied outs.
 * Returns 1 on success. fn_name's storage is runtime-owned. */
int matlab_dbg_frame_at(int i, int32_t *file_id, int32_t *line,
                         const char **fn_name) {
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Frames are stored with index 0 = outermost. Convert. */
    int idx = matlab_dbg.n_frames - 1 - i;
    int ok = idx >= 0 && idx < matlab_dbg.n_frames;
    if (ok) {
        *file_id = matlab_dbg.frames[idx].file_id;
        *line    = matlab_dbg.frames[idx].line;
        *fn_name = matlab_dbg.frames[idx].fn_name;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return ok;
}

/* Workspace snapshot — the server asks for these on every `variables`
 * request. Output uses a small array populated by the caller; the
 * server copies fields out while holding no runtime lock. The struct
 * for a variable's value is returned as its stored f64 or matrix
 * pointer; the server formats for display. */
int matlab_dbg_ws_count(void) {
    matlab_ws_init_if_needed();
    return matlab_ws ? matlab_ws->nfields : 0;
}

const char *matlab_dbg_ws_name(int i, int64_t *len_out) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) {
        *len_out = 0;
        return "";
    }
    const char *n = matlab_ws->names[i];
    *len_out = (int64_t)strlen(n);
    return n;
}

int matlab_dbg_ws_kind(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return -1;
    return matlab_ws->kinds[i];
}

double matlab_dbg_ws_f64(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return 0.0;
    return matlab_ws->f64_vals[i];
}

void *matlab_dbg_ws_ptr(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return NULL;
    return matlab_ws->ptr_vals[i];
}

/* #116: is the workspace slot `i` a 3-D array (matlab_mat3)?  A 3-D value is
 * stored under the generic mat kind=1, so the workspace-kind hook can't tell
 * it apart from a 2-D matrix without inspecting the pointer.  The REPL
 * workspace-kind hook calls this for kind=1 slots and reports a distinct kind
 * (16) when true, letting the Resolver re-stamp the binding 3-D so the next
 * turn's N-D subscript store/read detectors fire (instead of backing off to
 * "unsupported call shape"). */
int matlab_dbg_ws_is_mat3(int i) {
    matlab_ws_init_if_needed();
    if (!matlab_ws || i < 0 || i >= matlab_ws->nfields) return 0;
    if (matlab_ws->kinds[i] != 1) return 0;
    void *p = matlab_ws->ptr_vals[i];
    return (p && mat_is_3d(p)) ? 1 : 0;
}

/* Shape accessors used by the DAP `variables` formatter. Thin wrappers
 * around the opaque matlab_mat struct — the DAP server doesn't have
 * access to the internal layout. */
int64_t matlab_dbg_mat_rows(matlab_mat *m) { return m ? m->rows : 0; }
int64_t matlab_dbg_mat_cols(matlab_mat *m) { return m ? m->cols : 0; }

/* Element accessor for the DAP matrix-expansion path. Out-of-range
 * indices return 0.0 so a malformed children request can't read past
 * the data buffer. Indices are 1-based to match how the DAP server
 * presents cells (`(1,1)`, `(1,2)`, ...) — we subtract one before
 * indexing the row-major buffer. Complex / 3-D / typed-int matrices
 * have their own accessors below; this one returns 0.0 if asked
 * about a tagged descriptor. */
double matlab_dbg_mat_get(matlab_mat *m, int64_t i, int64_t j) {
    if (!m || !m->data) return 0.0;
    if (i < 1 || j < 1) return 0.0;
    if (mat_is_complex(m) || mat_is_3d(m)) return 0.0;
    if (i > m->rows || j > m->cols) return 0.0;
    /* Row-major: data[(i-1) * cols + (j-1)]. */
    return m->data[(i - 1) * m->cols + (j - 1)];
}

/* Discriminators + per-kind accessors used by the DAP `variables`
 * expander to drill into complex and 3-D matrices.
 *
 * The DAP server stores any kind=1 ws/frame value as a `void *`
 * because matlab_mat / matlab_mat_c / matlab_mat3 share that LLVM
 * type but have different layouts. Each helper below begins by
 * confirming the magic before accessing layout-specific fields, so
 * passing a plain matlab_mat into `matlab_dbg_mat_c_re()` is a
 * defensive zero rather than a wild read. */
int32_t matlab_dbg_mat_kind(const void *p) {
    if (!p) return 0;
    if (mat_is_complex(p)) return 2;   /* matlab_mat_c */
    if (mat_is_3d(p))      return 3;   /* matlab_mat3   */
    if (mat_is_nd(p))      return 4;   /* matlab_matN   */
    return 1;                          /* plain matlab_mat */
}

/* matN accessors — ndims / per-axis dim / flat-element read.  Caller-side
 * (matlabc DAP handlers, the workspace mirror) treat a matN like a
 * sequence: total = prod(dims), iterate by flat linear index.  The dims
 * tuple itself is exposed via matlab_dbg_matN_dim(M, k) (1-based k). */
int32_t matlab_dbg_matN_ndims(const void *p) {
    if (!p || !mat_is_nd(p)) return 0;
    return (int32_t)((const matlab_matN *)p)->ndims;
}
int64_t matlab_dbg_matN_dim(const void *p, int32_t k_1based) {
    if (!p || !mat_is_nd(p)) return 0;
    const matlab_matN *m = (const matlab_matN *)p;
    if (k_1based < 1 || (uint32_t)k_1based > m->ndims) return 1;
    return m->dims[k_1based - 1];
}
int64_t matlab_dbg_matN_numel(const void *p) {
    if (!p || !mat_is_nd(p)) return 0;
    const matlab_matN *m = (const matlab_matN *)p;
    int64_t t = 1; for (uint32_t k = 0; k < m->ndims; ++k) t *= m->dims[k];
    return t;
}
/* Flat-linear read (0-based); cheap for the debugger's per-cell drill. */
double matlab_dbg_matN_get_lin(const void *p, int64_t lin_zero_based) {
    if (!p || !mat_is_nd(p)) return 0.0;
    const matlab_matN *m = (const matlab_matN *)p;
    int64_t n = matlab_dbg_matN_numel(p);
    if (lin_zero_based < 0 || lin_zero_based >= n) return 0.0;
    return m->data[lin_zero_based];
}
/* matlab_mat_c accessors are defined alongside its struct body
 * further down in the complex section — that section needs to be
 * in scope to access ->re / ->im / ->rows / ->cols. The discriminator
 * above is layout-agnostic (reads only the magic at offset 0) so
 * it lives here. */
int64_t matlab_dbg_mat_c_rows(const matlab_mat_c *m);
int64_t matlab_dbg_mat_c_cols(const matlab_mat_c *m);
double matlab_dbg_mat_c_re(const matlab_mat_c *m, int64_t i, int64_t j);
double matlab_dbg_mat_c_im(const matlab_mat_c *m, int64_t i, int64_t j);
int64_t matlab_dbg_mat3_rows(const matlab_mat3 *m) {
    if (!m || !mat_is_3d(m)) return 0;
    return m->rows;
}
int64_t matlab_dbg_mat3_cols(const matlab_mat3 *m) {
    if (!m || !mat_is_3d(m)) return 0;
    return m->cols;
}
int64_t matlab_dbg_mat3_depth(const matlab_mat3 *m) {
    if (!m || !mat_is_3d(m)) return 0;
    return m->depth;
}
/* Memory-inspection accessors. The DAP `readMemory` / `writeMemory`
 * requests use a `memoryReference` (per spec, a hex string) plus an
 * offset to identify what to read. We hand out memory refs only for
 * matrix data buffers — everything else is opaque or scalar — and
 * the readMemory handler decodes the hex back to a pointer to walk
 * the cells as raw bytes. Returning the buffer pointer + total byte
 * size lets the DAP server bound the read so a 100MB readMemory
 * request can't walk past the buffer. */
void *matlab_dbg_mat_data_ptr(void *Mraw) {
    if (!Mraw) return NULL;
    int32_t kind = matlab_dbg_mat_kind(Mraw);
    if (kind == 1) return ((matlab_mat *)Mraw)->data;
    if (kind == 3) return ((matlab_mat3 *)Mraw)->data;
    if (kind == 4) return ((matlab_matN *)Mraw)->data;
    /* Complex matrices have two parallel buffers (re/im); a single
     * pointer can't cover both. Refuse for now — the IDE's memory
     * view would only see the real component, which would be
     * misleading. */
    return NULL;
}
int64_t matlab_dbg_mat_data_bytes(void *Mraw) {
    if (!Mraw) return 0;
    int32_t kind = matlab_dbg_mat_kind(Mraw);
    if (kind == 1) {
        matlab_mat *m = (matlab_mat *)Mraw;
        return m->rows * m->cols * (int64_t)sizeof(double);
    }
    if (kind == 3) {
        matlab_mat3 *m = (matlab_mat3 *)Mraw;
        return m->rows * m->cols * m->depth * (int64_t)sizeof(double);
    }
    if (kind == 4) {
        return matlab_dbg_matN_numel(Mraw) * (int64_t)sizeof(double);
    }
    return 0;
}

double matlab_dbg_mat3_get(const matlab_mat3 *m,
                           int64_t i, int64_t j, int64_t k) {
    if (!m || !mat_is_3d(m) || !m->data) return 0.0;
    if (i < 1 || j < 1 || k < 1) return 0.0;
    if (i > m->rows || j > m->cols || k > m->depth) return 0.0;
    /* Slice-major: matches mat3_offset above. */
    return m->data[(k - 1) * m->rows * m->cols + (i - 1) * m->cols + (j - 1)];
}

/* The injected hook. Called from JIT'd code at each statement entry
 * when compiled with -g. Takes (file_id, line) as raw ints so the
 * emitted call is cheap — just two arith.constant ops feeding a
 * known runtime symbol. */
void matlab_dbg_hook(int32_t file_id, int32_t line) {
    pthread_mutex_lock(&matlab_dbg.mu);
    if (!matlab_dbg.enabled) {
        pthread_mutex_unlock(&matlab_dbg.mu);
        return;
    }
    /* Lazy-register the calling thread on first hook entry so the
     * DAP server can enumerate it via `threads`. Also seeds the
     * thread's per-thread frame chain with a `<script>` entry on
     * first touch so frame[0].fn_name reads correctly. */
    int thr_idx = matlab_dbg_thread_init_chain_locked();
    int *thr_n = &matlab_dbg.thread_n_frames[thr_idx];
    /* Update the innermost frame's line in the calling thread's
     * own chain. Concurrent parfor workers each touch their own
     * slot, so no cross-thread corruption. */
    if (*thr_n > 0) {
        matlab_dbg.thread_frames[thr_idx][*thr_n - 1].file_id = file_id;
        matlab_dbg.thread_frames[thr_idx][*thr_n - 1].line = line;
    }
    /* Statement-boundary record for reverse stepping. The undo
     * log gets one of these per hook fire; stepBack walks back
     * until it finds the previous boundary. Cheap (no allocation
     * — kind=0 records just stamp ints). */
    matlab_dbg_undo_record_stmt_locked(file_id, line, thr_idx);

    int should_pause = 0;
    int matched_bp = -1;
    /* Stepping: decide based on action + the calling thread's own
     * depth (step targets are per-thread; a step in worker A
     * shouldn't fire when worker B reaches its target depth). */
    switch (matlab_dbg.action) {
    case MATLAB_DBG_STEP_IN:
        should_pause = 1;
        break;
    case MATLAB_DBG_STEP_OVER:
        if (*thr_n <= matlab_dbg.thread_step_target_depth[thr_idx])
            should_pause = 1;
        break;
    case MATLAB_DBG_STEP_OUT:
        if (*thr_n <= matlab_dbg.thread_step_target_depth[thr_idx])
            should_pause = 1;
        break;
    case MATLAB_DBG_STOP:
        pthread_mutex_unlock(&matlab_dbg.mu);
        pthread_exit(NULL);
        return;
    default:
        break;
    }
    /* Breakpoint check (regardless of step action). Records the
     * matched index so the DAP server can read the breakpoint's
     * condition / log strings without re-walking the table.
     *
     * Hit-count gate: when hit_op is set, increment hit_count and
     * compare to hit_target with the encoded operator. A hit_op
     * of 0 (no gate) goes straight to should_pause = 1, matching
     * the prior behaviour. The gate runs BEFORE the conditional /
     * log eval so a `hitCondition: ">= 100"` skips the JIT cost
     * for the first 99 hits — important for tight loops. */
    for (int i = 0; i < matlab_dbg.n_bp; ++i) {
        if (matlab_dbg.bp_file[i] == file_id &&
            matlab_dbg.bp_line[i] == line) {
            matched_bp = i;
            int op = matlab_dbg.hit_op[i];
            if (op != 0) {
                int64_t c = ++matlab_dbg.hit_count[i];
                int64_t t = matlab_dbg.hit_target[i];
                int gate = 0;
                switch (op) {
                case 1: gate = (c == t); break;
                case 2: gate = (c >= t); break;
                case 3: gate = (c >  t); break;
                case 4: gate = (t > 0 && c % t == 0); break;
                default: gate = 1; break;
                }
                if (!gate) break;
            }
            should_pause = 1;
            break;
        }
    }
    /* Exception-breakpoint filter: pause if the error flag is set
     * AND the DAP client has enabled the `error` filter. Reads the
     * error flag directly to avoid recursing through the public API
     * while we already hold matlab_dbg.mu. */
    if (matlab_dbg.pause_on_error && matlab_error_flag) {
        should_pause = 1;
    }
    if (should_pause) {
        matlab_dbg.cur_file_id = file_id;
        matlab_dbg.cur_line = line;
        matlab_dbg.cur_bp_idx = matched_bp;
        matlab_dbg.paused = 1;
        matlab_dbg.paused_thread_idx = thr_idx;
        /* Snapshot the calling thread's frame chain into the
         * shared frames[] / frame_locals[] arrays so DAP
         * inspectors that still read those directly see the
         * paused thread's stack. The legacy single-threaded
         * accessors (matlab_dbg_frame_count / _frame_at /
         * _frame_local_*) are unmodified — they read from
         * frames[]/frame_locals[] which is now a snapshot view.
         *
         * Names are kept as-is (the per-thread chain owns them),
         * so the snapshot is a shallow copy. The shared array's
         * matlab_dbg_free_frame_locals path is no longer called
         * during normal lifecycle — ownership stays with the
         * per-thread arrays. */
        int n = matlab_dbg.thread_n_frames[thr_idx];
        if (n > MATLAB_DBG_MAX_FRAMES) n = MATLAB_DBG_MAX_FRAMES;
        matlab_dbg.n_frames = n;
        for (int i = 0; i < n; ++i) {
            matlab_dbg.frames[i] = matlab_dbg.thread_frames[thr_idx][i];
            matlab_dbg.frame_locals[i] =
                matlab_dbg.thread_frame_locals[thr_idx][i];
        }
        /* Signal the server that we're paused; wait for resume. */
        pthread_cond_broadcast(&matlab_dbg.cv_server);
        while (matlab_dbg.paused) {
            pthread_cond_wait(&matlab_dbg.cv_client, &matlab_dbg.mu);
        }
        matlab_dbg.paused_thread_idx = -1;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Frame-tracking hooks (used when -g is on and we instrument user
 * function entry/exit). The name pointer the JIT hands us is into a
 * read-only global that is NOT null-terminated — the global is sized
 * exactly to the bytes of the function name, with no trailing 0. Any
 * caller that subsequently uses %s on `fn_name` would read past the
 * global into whatever happens to be next in the constant pool, which
 * is exactly what tripped up the DAP `stackTrace` response and the
 * error()-backtrace printer.
 *
 * Heap-copy the name on enter and free it on leave. This keeps every
 * downstream consumer (DAP server, traceback printer, future eval)
 * able to treat fn_name as a plain C string. The cost is a tiny
 * malloc/free per call when -g is on, which is the path that's
 * already paying the per-statement hook overhead. */
/* Free the locals stored at frame_idx — used both on leave_frame
 * and as a defensive reset on enter_frame in case a previous run
 * left stale entries (shouldn't happen with balanced enter/leave but
 * cheap insurance for recursive functions reusing the same depth). */
static void matlab_dbg_free_frame_locals(int frame_idx) {
    /* Caller must hold matlab_dbg.mu. */
    if (frame_idx < 0 || frame_idx >= MATLAB_DBG_MAX_FRAMES) return;
    struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
    for (int i = 0; i < fl->n; ++i) {
        free(fl->entries[i].name);
        fl->entries[i].name = NULL;
    }
    fl->n = 0;
}

/* Resolve the calling thread's per-thread frame chain, lazily
 * seeding it with a `<script>` entry on first touch so DAP
 * inspectors always have a frame[0] to read. CALLER MUST HOLD
 * matlab_dbg.mu. */
static int matlab_dbg_thread_init_chain_locked(void) {
    int slot = matlab_dbg_thread_slot_locked();
    if (matlab_dbg.thread_n_frames[slot] == 0) {
        matlab_dbg.thread_n_frames[slot] = 1;
        matlab_dbg.thread_frames[slot][0].file_id = 0;
        matlab_dbg.thread_frames[slot][0].line = 0;
        matlab_dbg.thread_frames[slot][0].fn_name = "<script>";
    }
    return slot;
}

void matlab_dbg_enter_frame(const char *fn_name, int64_t name_len) {
    if (name_len < 0) name_len = 0;
    char *owned = (char *)malloc((size_t)name_len + 1);
    if (owned) {
        if (name_len > 0) memcpy(owned, fn_name, (size_t)name_len);
        owned[name_len] = '\0';
    }
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Per-thread chain push. Each pthread (main worker, parfor
     * workers) maintains its own call-stack so concurrent
     * parfor-body enters don't trample each other. The shared
     * frames[] is refreshed only when this thread pauses (in
     * the hook) so DAP inspectors that still read frames[]
     * directly see the paused thread's stack. */
    int slot = matlab_dbg_thread_init_chain_locked();
    int *pn = &matlab_dbg.thread_n_frames[slot];
    if (*pn < MATLAB_DBG_MAX_FRAMES) {
        struct matlab_dbg_frame_locals *fl =
            &matlab_dbg.thread_frame_locals[slot][*pn];
        for (int i = 0; i < fl->n; ++i) free(fl->entries[i].name);
        fl->n = 0;
        matlab_dbg.thread_frames[slot][*pn].fn_name = owned;
        matlab_dbg.thread_frames[slot][*pn].file_id = 0;
        matlab_dbg.thread_frames[slot][*pn].line = 0;
        (*pn)++;
    } else {
        free(owned);  /* table full; drop the name we copied */
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

void matlab_dbg_leave_frame(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = matlab_dbg_thread_init_chain_locked();
    int *pn = &matlab_dbg.thread_n_frames[slot];
    if (*pn > 1) {
        (*pn)--;
        char *owned = (char *)matlab_dbg.thread_frames[slot][*pn].fn_name;
        matlab_dbg.thread_frames[slot][*pn].fn_name = NULL;
        free(owned);
        struct matlab_dbg_frame_locals *fl =
            &matlab_dbg.thread_frame_locals[slot][*pn];
        for (int i = 0; i < fl->n; ++i) free(fl->entries[i].name);
        fl->n = 0;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Mirror entry points called from the lowering after every store to
 * a named slot when DebugMode is on. Records the variable's current
 * value into the innermost frame's mini-workspace so the DAP server
 * can render Locals for any frame in the stack — not just the
 * script-level workspace.
 *
 * The implementation is deliberately the simple linear-scan one:
 * MATLAB programs' per-function variable counts are tiny (a handful)
 * and stores are cheap; a hash table would be heavier for no gain.
 * Names are heap-copied on first set (subsequent updates reuse the
 * existing entry). The matrix pointer is stored as borrowed — the
 * matrix struct itself is owned by the JIT's slot or workspace and
 * survives at least until matlab_dbg_leave_frame fires. */
/* Generic find-or-alloc operating on a caller-supplied
 * frame_locals slot. Lets the per-thread frame_set_* path target
 * its own thread's slot without going through the shared
 * matlab_dbg.frame_locals[]. */
static int matlab_dbg_frame_local_find_or_alloc_in(
    struct matlab_dbg_frame_locals *fl,
    const char *name, int64_t name_len) {
    if (!fl) return -1;
    for (int i = 0; i < fl->n; ++i) {
        if (fl->entries[i].name_len == name_len &&
            memcmp(fl->entries[i].name, name, (size_t)name_len) == 0)
            return i;
    }
    if (fl->n >= MATLAB_DBG_MAX_LOCALS) return -1;
    char *copy = (char *)malloc((size_t)name_len + 1);
    if (!copy) return -1;
    memcpy(copy, name, (size_t)name_len);
    copy[name_len] = '\0';
    int idx = fl->n++;
    fl->entries[idx].name = copy;
    fl->entries[idx].name_len = name_len;
    fl->entries[idx].kind = 0;
    fl->entries[idx].f64 = 0.0;
    fl->entries[idx].ptr = NULL;
    return idx;
}

/* Resolve the calling thread's innermost-frame frame_locals slot,
 * lazily seeding the chain if this is the thread's first touch.
 * Returns NULL if the chain is empty (n == 0) — caller drops the
 * write silently in that case. */
static struct matlab_dbg_frame_locals *
matlab_dbg_thread_innermost_locals_locked(void) {
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    if (n <= 0 || n > MATLAB_DBG_MAX_FRAMES) return NULL;
    return &matlab_dbg.thread_frame_locals[slot][n - 1];
}

void matlab_dbg_frame_set_f64(const char *name, int64_t name_len, double v) {
    if (!name || name_len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    /* Write into the calling thread's innermost-frame slot. Per-
     * thread storage means concurrent parfor workers' Locals don't
     * trample each other's frames. The shared frame_locals[] is
     * refreshed by the hook on pause via the snapshot-to-shared
     * copy, so DAP inspectors see the paused thread's view. */
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    struct matlab_dbg_undo_rec *r = NULL;
    if (n > 0)
        r = matlab_dbg_frame_push_undo_locked(slot, n - 1, name, name_len);
    struct matlab_dbg_frame_locals *fl =
        matlab_dbg_thread_innermost_locals_locked();
    if (fl) {
        int idx = matlab_dbg_frame_local_find_or_alloc_in(fl, name, name_len);
        if (idx >= 0) {
            fl->entries[idx].kind = 0;
            fl->entries[idx].f64 = v;
            fl->entries[idx].ptr = NULL;
        }
    }
    matlab_dbg_undo_record_set_new_f64(r, v);
    /* Watchpoint check on frame-local writes. scope_hint=2 (frame).
     * Already inside the dbg mutex, so we call _watch_check /
     * _trip directly without re-locking. */
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int wp = matlab_dbg_watch_check(name, name_len, /*scope_hint=*/2);
        if (wp >= 0) matlab_dbg_watch_trip(wp);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

void matlab_dbg_frame_set_mat(const char *name, int64_t name_len, void *mat) {
    if (!name || name_len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    struct matlab_dbg_undo_rec *r = NULL;
    if (n > 0)
        r = matlab_dbg_frame_push_undo_locked(slot, n - 1, name, name_len);
    struct matlab_dbg_frame_locals *fl =
        matlab_dbg_thread_innermost_locals_locked();
    if (fl) {
        int idx = matlab_dbg_frame_local_find_or_alloc_in(fl, name, name_len);
        if (idx >= 0) {
            fl->entries[idx].kind = 1;
            fl->entries[idx].ptr = mat;
            fl->entries[idx].f64 = 0.0;
        }
    }
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/1, mat);
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int wp = matlab_dbg_watch_check(name, name_len, /*scope_hint=*/2);
        if (wp >= 0) matlab_dbg_watch_trip(wp);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* Class-instance variant. `obj` is a matlab_obj* whose class_id tag
 * is the registry key for the class name (see
 * matlab_dbg_register_class). Same lifetime contract as set_mat —
 * the obj is borrowed from the JIT's slot. */
void matlab_dbg_frame_set_obj(const char *name, int64_t name_len, void *obj) {
    if (!name || name_len <= 0) return;
    pthread_mutex_lock(&matlab_dbg.mu);
    int slot = matlab_dbg_thread_init_chain_locked();
    int n = matlab_dbg.thread_n_frames[slot];
    struct matlab_dbg_undo_rec *r = NULL;
    if (n > 0)
        r = matlab_dbg_frame_push_undo_locked(slot, n - 1, name, name_len);
    struct matlab_dbg_frame_locals *fl =
        matlab_dbg_thread_innermost_locals_locked();
    if (fl) {
        int idx = matlab_dbg_frame_local_find_or_alloc_in(fl, name, name_len);
        if (idx >= 0) {
            fl->entries[idx].kind = 2;
            fl->entries[idx].ptr = obj;
            fl->entries[idx].f64 = 0.0;
        }
    }
    matlab_dbg_undo_record_set_new_ptr(r, /*new_kind=*/2, obj);
    if (matlab_dbg.enabled && matlab_dbg.n_wp > 0) {
        int wp = matlab_dbg_watch_check(name, name_len, /*scope_hint=*/2);
        if (wp >= 0) matlab_dbg_watch_trip(wp);
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
}

/* DAP read-side: enumerate Locals for a given frame index. Frame
 * indexing here matches matlab_dbg.frames[] (0 = outermost / script,
 * n_frames-1 = innermost). The DAP server adapts this to its own
 * top-of-stack-first frame ordering. */
int matlab_dbg_frame_locals_count(int frame_idx) {
    int n = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames)
        n = matlab_dbg.frame_locals[frame_idx].n;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return n;
}

const char *matlab_dbg_frame_local_name(int frame_idx, int i,
                                         int64_t *len_out) {
    const char *p = NULL;
    int64_t L = 0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) {
            p = fl->entries[i].name;
            L = fl->entries[i].name_len;
        }
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    if (len_out) *len_out = L;
    return p;
}

int matlab_dbg_frame_local_kind(int frame_idx, int i) {
    int k = -1;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) k = fl->entries[i].kind;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return k;
}

double matlab_dbg_frame_local_f64(int frame_idx, int i) {
    double v = 0.0;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) v = fl->entries[i].f64;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return v;
}

void *matlab_dbg_frame_local_ptr(int frame_idx, int i) {
    void *p = NULL;
    pthread_mutex_lock(&matlab_dbg.mu);
    if (frame_idx >= 0 && frame_idx < matlab_dbg.n_frames) {
        struct matlab_dbg_frame_locals *fl = &matlab_dbg.frame_locals[frame_idx];
        if (i >= 0 && i < fl->n) p = fl->entries[i].ptr;
    }
    pthread_mutex_unlock(&matlab_dbg.mu);
    return p;
}

/* Blocks until the worker is paused or has exited. Used by the
 * server to know when it can handle client requests safely. */
void matlab_dbg_wait_for_pause(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    while (!matlab_dbg.paused)
        pthread_cond_wait(&matlab_dbg.cv_server, &matlab_dbg.mu);
    pthread_mutex_unlock(&matlab_dbg.mu);
}

int matlab_dbg_is_paused(void) {
    pthread_mutex_lock(&matlab_dbg.mu);
    int p = matlab_dbg.paused;
    pthread_mutex_unlock(&matlab_dbg.mu);
    return p;
}

void matlab_dbg_mat(const char *file, int64_t file_len,
                    int32_t line,
                    const char *label, int64_t label_len,
                    matlab_mat *m) {
    int fl = (int)(file_len > 0 ? file_len : 0);
    int ll = (int)(label_len > 0 ? label_len : 0);
    const char *flt = file ? file : "<repl>";
    if (!file) fl = (int)strlen(flt);
    pthread_mutex_lock(&matlab_io_mutex);
    if (!m) {
        fprintf(stderr, "%.*s:%d: %.*s = <null>\n",
                fl, flt, line,
                ll > 0 ? ll : (int)strlen("<expr>"),
                ll > 0 ? label : "<expr>");
        pthread_mutex_unlock(&matlab_io_mutex);
        return;
    }
    fprintf(stderr, "%.*s:%d: %.*s = [%lldx%lld]\n",
            fl, flt, line,
            ll > 0 ? ll : (int)strlen("<expr>"),
            ll > 0 ? label : "<expr>",
            (long long)m->rows, (long long)m->cols);
    /* Also print the matrix content (up to 8 rows / 8 cols) so
     * small matrices are readable inline. */
    int64_t maxr = m->rows > 8 ? 8 : m->rows;
    int64_t maxc = m->cols > 8 ? 8 : m->cols;
    for (int64_t i = 0; i < maxr; ++i) {
        fprintf(stderr, "  ");
        for (int64_t j = 0; j < maxc; ++j) {
            fprintf(stderr, " %10g", m->data[i * m->cols + j]);
        }
        if (m->cols > 8) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }
    if (m->rows > 8) fprintf(stderr, "  ...\n");
    pthread_mutex_unlock(&matlab_io_mutex);
}
} /* extern "C" */
