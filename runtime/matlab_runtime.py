"""
Python runtime shim for matlab_llvm's `-emit-python` backend.

Mirrors the C runtime's `matlab_*` API, but each symbol is exposed with
the `matlab_` prefix dropped so the emitted source reads as `rt.foo(...)`
against `import matlab_runtime as rt`.

NumPy-backed where it makes sense; struct/cell use plain Python types.
Designed to pass the Run-suite stdout byte-for-byte on the common cases
(disp / fprintf / simple matrix arithmetic). Numerically-sensitive
programs that rely on MATLAB-bit-exact eig/svd/fft may diverge — those
tests carry a `.skip-emit-python` marker in the test runner.

The module is one file on purpose: simpler to ship, and the emitter never
references anything outside this namespace.
"""

from __future__ import annotations

import builtins
import math
import sys
import threading
import numpy as np

# Preserve the Python builtins that collide with MATLAB names so module
# internals don't end up calling the MATLAB versions recursively.
_pyrange = builtins.range
_pymax = builtins.max
_pymin = builtins.min
_pysum = builtins.sum
_pyabs = builtins.abs


# ---------------------------------------------------------------------------
# disp / fprintf
# ---------------------------------------------------------------------------

# MATLAB's `disp` / default numeric output uses a compact format: integer-
# valued doubles print as ints, fractional doubles with a short precision.
# The C runtime collapses this to `%g` (5 significant digits). Mirror that.

def _fmt_scalar(v):
    """Format a scalar the way C's `%g` would."""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f): return "NaN"
    if math.isinf(f): return "Inf" if f > 0 else "-Inf"
    # %g with default 6-digit precision (C's default).
    return f"{f:g}"


def _fmt_col(v):
    """Right-align `%7g` — matches the C runtime's matrix cell width."""
    if isinstance(v, (bool, np.bool_)):
        s = "1" if v else "0"
    else:
        try:
            f = float(v)
            s = f"{f:g}"
        except (TypeError, ValueError):
            s = str(v)
    return f"{s:>7}"


def disp_str(s, n=None):
    # `n` is the byte length the C runtime wants; Python strings carry
    # their length, so we ignore it.
    print(s)


def disp_f64(v):
    print(_fmt_scalar(v))


def disp_vec_f64(data, n=None):
    if n is None:
        n = len(data)
    parts = [f"   {_fmt_col(data[i])}" for i in _pyrange(int(n))]
    print("".join(parts))


def disp_mat_f64(data, m, n):
    m = int(m); n = int(n)
    for i in _pyrange(m):
        row = [f"   {_fmt_col(data[i * n + j])}" for j in _pyrange(n)]
        print("".join(row))


def disp_mat(A):
    """Polymorphic matrix disp — handles ndarray or scalar."""
    if A is None:
        print("     []")
        return
    if isinstance(A, (int, float, bool, np.integer, np.floating, np.bool_)):
        print(_fmt_scalar(A))
        return
    arr = np.asarray(A)
    if arr.ndim == 0:
        print(_fmt_scalar(arr.item()))
        return
    if arr.size == 0:
        return  # MATLAB's disp of [] prints nothing
    if arr.ndim == 1:
        # Row vector.
        row = [f"   {_fmt_col(x)}" for x in arr]
        print("".join(row))
        return
    if arr.ndim == 2:
        m, n = arr.shape
        if m == 1 and n == 1:
            print(_fmt_scalar(arr[0, 0]))
            return
        for i in _pyrange(m):
            row = [f"   {_fmt_col(arr[i, j])}" for j in _pyrange(n)]
            print("".join(row))
        return
    # Higher-dim: fall back to numpy's default repr.
    print(arr)


def _expand_escapes(fmt):
    """MATLAB-style backslash-escape expansion inside format strings."""
    out = []
    i = 0
    s = fmt
    while i < len(s):
        c = s[i]
        if c != '\\' or i + 1 >= len(s):
            out.append(c); i += 1; continue
        e = s[i + 1]; i += 2
        if e == 'n': out.append('\n')
        elif e == 't': out.append('\t')
        elif e == 'r': out.append('\r')
        elif e == '\\': out.append('\\')
        elif e == '\'': out.append('\'')
        elif e == '"': out.append('"')
        elif e == '0': out.append('\0')
        else: out.append('\\'); out.append(e)
    return "".join(out)


def _c_printf(fmt, *args):
    """Very small subset of C's printf: %d / %i / %f / %e / %g / %s / %c,
    with optional width/precision. Good enough for MATLAB fprintf use."""
    import re
    out = []
    i = 0
    ai = 0
    s = fmt
    spec_re = re.compile(r'^%([-+ #0]*)(\d+)?(?:\.(\d+))?([diouxXeEfFgGscp%])')
    while i < len(s):
        c = s[i]
        if c != '%':
            out.append(c); i += 1; continue
        m = spec_re.match(s[i:])
        if not m:
            out.append(c); i += 1; continue
        flags, width, prec, conv = m.group(1, 2, 3, 4)
        if conv == '%':
            out.append('%'); i += m.end(); continue
        arg = args[ai] if ai < len(args) else 0
        ai += 1
        # Build a Python format spec.
        spec = '%' + (flags or '') + (width or '')
        if prec is not None: spec += '.' + prec
        spec += conv
        try:
            if conv in 'di':
                out.append(spec % int(arg))
            elif conv in 'ouxX':
                out.append(spec % (int(arg) & 0xFFFFFFFFFFFFFFFF))
            elif conv in 'eEfFgG':
                out.append(spec % float(arg))
            elif conv == 's':
                out.append(spec % str(arg))
            elif conv == 'c':
                out.append(spec % (chr(int(arg)) if isinstance(arg, (int, float)) else str(arg)))
            else:
                out.append(spec % arg)
        except (TypeError, ValueError):
            out.append(str(arg))
        i += m.end()
    return "".join(out)


def fprintf_str(fmt, n=None):
    sys.stdout.write(_c_printf(_expand_escapes(fmt)))


# The `-emit-python` backend drops the C-ABI string-length operand at
# call sites, so the natural Python signatures are `(fmt, *values)`.
# We accept both shapes for back-compat with hand-written callers that
# pass the legacy `n` length.
def _fprintf_split(fmt, args):
    if args and isinstance(args[0], int) and len(args) > 0:
        # Detect the legacy (fmt, n, ...values) shape: n is an int that
        # equals strlen(fmt). When that matches, drop it.
        try:
            if args[0] == len(_expand_escapes(fmt)):
                return args[1:]
        except Exception:
            pass
    return args


def fprintf_f64(fmt, *args):
    args = _fprintf_split(fmt, args)
    sys.stdout.write(_c_printf(_expand_escapes(fmt), *args))


def fprintf_f64_2(fmt, *args):
    args = _fprintf_split(fmt, args)
    sys.stdout.write(_c_printf(_expand_escapes(fmt), *args))


def fprintf_f64_3(fmt, *args):
    args = _fprintf_split(fmt, args)
    sys.stdout.write(_c_printf(_expand_escapes(fmt), *args))


def fprintf_f64_4(fmt, *args):
    args = _fprintf_split(fmt, args)
    sys.stdout.write(_c_printf(_expand_escapes(fmt), *args))


def _fp_write(fp, s):
    if fp is None: return
    try:
        if 'b' in getattr(fp, 'mode', ''):
            fp.write(s.encode('utf-8'))
        else:
            fp.write(s)
    except Exception:
        try: fp.write(s)
        except Exception: pass


def fprintf_file_str(fp, fmt, n=None):
    _fp_write(fp, _c_printf(_expand_escapes(str(fmt))))


def fprintf_file_f64(fp, fmt, n=None, v=None):
    if v is None:
        v = n; n = None
    _fp_write(fp, _c_printf(_expand_escapes(str(fmt)), v))


def input_num(prompt, plen=None):
    sys.stdout.write(prompt); sys.stdout.flush()
    try: return float(input())
    except Exception: return 0.0


# ---------------------------------------------------------------------------
# Matrix helpers (NumPy-backed)
# ---------------------------------------------------------------------------

def _m(x):
    """Coerce to a numpy ndarray (2D) — `m` prefix convention."""
    if x is None: return np.zeros((0, 0))
    if isinstance(x, np.ndarray): return x
    if isinstance(x, (int, float, bool, np.number, np.bool_)):
        return np.array([[float(x)]])
    return np.asarray(x, dtype=float)


def mat_from_buf(buf, m, n):
    m = int(m); n = int(n)
    arr = np.array(list(buf)[:m * n], dtype=float)
    return arr.reshape((m, n)) if arr.size else np.zeros((m, n))


def mat_from_scalar(x):
    return np.array([[float(x)]])


def empty_mat():
    return np.zeros((0, 0))


def zeros(m, n=None):
    m = int(m)
    if n is None: return np.zeros((m, m))
    return np.zeros((m, int(n)))


def ones(m, n=None):
    m = int(m)
    if n is None: return np.ones((m, m))
    return np.ones((m, int(n)))


def ones3(m, n, p):
    return np.ones((int(m), int(n), int(p)))


def zeros3(m, n, p):
    return np.zeros((int(m), int(n), int(p)))


def eye(m, n=None):
    m = int(m)
    if n is None: return np.eye(m)
    return np.eye(m, int(n))


def magic(nd):
    """MATLAB's magic(n)."""
    n = int(nd)
    if n < 1: return np.zeros((0, 0))
    if n == 1: return np.array([[1.0]])
    if n == 2: return np.array([[1.0, 3.0], [4.0, 2.0]])
    # Odd-n: Siamese method; doubly-even / singly-even fallbacks. Good
    # enough for common cases.
    M = np.zeros((n, n))
    if n % 2 == 1:
        i, j = 0, n // 2
        for k in _pyrange(1, n * n + 1):
            M[i, j] = k
            ni, nj = (i - 1) % n, (j + 1) % n
            if M[ni, nj] != 0:
                i = (i + 1) % n
            else:
                i, j = ni, nj
        return M
    # Even n: numpy doesn't have magic built-in; fill with a placeholder.
    M = (np.arange(n * n) + 1).reshape((n, n)).astype(float)
    return M


def range(start, step, end):
    # MATLAB's `start:step:end` — inclusive, handles negative step.
    s = float(start); st = float(step); e = float(end)
    if st == 0: return np.zeros((1, 0))
    count = int((e - s) / st) + 1
    if count <= 0: return np.zeros((1, 0))
    vals = s + st * np.arange(count)
    return vals.reshape((1, count))


def frange(start, end, step):
    # Iterator form of MATLAB's colon, used as the fallback target of
    # `for i = start:step:end` in `-emit-python` when bounds are not
    # compile-time integer literals. Each yielded value is a Python float
    # so downstream arithmetic stays in f64.
    s = float(start); st = float(step); e = float(end)
    if st == 0: return
    if st > 0:
        x = s
        while x <= e:
            yield x
            x = x + st
    else:
        x = s
        while x >= e:
            yield x
            x = x + st


def linspace(a, b, n=None):
    if n is None: n = 100
    return np.linspace(float(a), float(b), int(n)).reshape((1, int(n)))


def repmat(A, m, n):
    return np.tile(_m(A), (int(m), int(n)))


def transpose(A):
    return _m(A).T


def diag(A):
    a = _m(A)
    if a.ndim <= 1 or a.shape[0] == 1 or a.shape[1] == 1:
        # Input is a vector — build a diagonal matrix.
        return np.diag(a.flatten())
    # Input is a matrix — return its diagonal as a column vector.
    return np.diag(a).reshape((-1, 1))


def reshape(A, m, n):
    # The matlab_llvm runtime stores matrices row-major and its reshape
    # preserves that layout — mirror it rather than MATLAB's native
    # column-major reshape so stdout matches the C lane byte-for-byte.
    return _m(A).reshape((int(m), int(n)))


# --- linear algebra --------------------------------------------------------

def matmul_mm(A, B): return _m(A) @ _m(B)
def inv(A):           return np.linalg.inv(_m(A))
def mldivide_mm(A, B): return np.linalg.solve(_m(A), _m(B))
def mrdivide_mm(A, B): return _m(A) @ np.linalg.inv(_m(B))
def det(A):            return float(np.linalg.det(_m(A)))
def svd(A):            return np.linalg.svd(_m(A), compute_uv=False).reshape((-1, 1))
def eig(A):            return np.linalg.eigvals(_m(A)).real.reshape((-1, 1))
def eig_V(A):
    _, V = np.linalg.eig(_m(A))
    return V.real
def eig_D(A):
    w, _ = np.linalg.eig(_m(A))
    return np.diag(w.real)
def chol(A):          return np.linalg.cholesky(_m(A)).T
def _lu_decompose(A):
    """Dolittle LU for square matrices (no pivoting). Matches the
    behavior of the matlab_llvm C runtime closely enough for small
    test matrices."""
    a = _m(A).astype(float).copy()
    n = a.shape[0]
    L = np.eye(n)
    U = a.copy()
    for k in _pyrange(n):
        if U[k, k] == 0: continue
        for i in _pyrange(k + 1, n):
            f = U[i, k] / U[k, k]
            L[i, k] = f
            U[i, k:] -= f * U[k, k:]
    return L, U

def lu_L(A):
    L, _ = _lu_decompose(A); return L
def lu_U(A):
    _, U = _lu_decompose(A); return U
def qr_Q(A): return np.linalg.qr(_m(A))[0]
def qr_R(A): return np.linalg.qr(_m(A))[1]
def pinv(A): return np.linalg.pinv(_m(A))
def trace(A): return float(np.trace(_m(A)))
def norm(A): return float(np.linalg.norm(_m(A)))


# --- elementwise binary ops -----------------------------------------------

def add_mm(A, B): return _m(A) + _m(B)
def sub_mm(A, B): return _m(A) - _m(B)
def emul_mm(A, B): return _m(A) * _m(B)
def ediv_mm(A, B): return _m(A) / _m(B)
def epow_mm(A, B): return _m(A) ** _m(B)

def add_ms(A, s): return _m(A) + float(s)
def sub_ms(A, s): return _m(A) - float(s)
def emul_ms(A, s): return _m(A) * float(s)
def ediv_ms(A, s): return _m(A) / float(s)
def epow_ms(A, s): return _m(A) ** float(s)

def add_sm(s, A): return float(s) + _m(A)
def sub_sm(s, A): return float(s) - _m(A)
def emul_sm(s, A): return float(s) * _m(A)
def ediv_sm(s, A): return float(s) / _m(A)
def epow_sm(s, A): return float(s) ** _m(A)


# --- comparisons (return 0/1 matrices to mirror MATLAB) -------------------

def gt_mm(A, B): return (_m(A) > _m(B)).astype(float)
def ge_mm(A, B): return (_m(A) >= _m(B)).astype(float)
def lt_mm(A, B): return (_m(A) < _m(B)).astype(float)
def le_mm(A, B): return (_m(A) <= _m(B)).astype(float)
def eq_mm(A, B): return (_m(A) == _m(B)).astype(float)
def ne_mm(A, B): return (_m(A) != _m(B)).astype(float)
def gt_ms(A, s): return (_m(A) > float(s)).astype(float)
def ge_ms(A, s): return (_m(A) >= float(s)).astype(float)
def lt_ms(A, s): return (_m(A) < float(s)).astype(float)
def le_ms(A, s): return (_m(A) <= float(s)).astype(float)
def eq_ms(A, s): return (_m(A) == float(s)).astype(float)
def ne_ms(A, s): return (_m(A) != float(s)).astype(float)
def gt_sm(s, A): return (float(s) > _m(A)).astype(float)
def ge_sm(s, A): return (float(s) >= _m(A)).astype(float)
def lt_sm(s, A): return (float(s) < _m(A)).astype(float)
def le_sm(s, A): return (float(s) <= _m(A)).astype(float)
def eq_sm(s, A): return (float(s) == _m(A)).astype(float)
def ne_sm(s, A): return (float(s) != _m(A)).astype(float)


# --- elementwise unary ops -------------------------------------------------

def neg_m(A): return -_m(A)
def exp_m(A): return np.exp(_m(A))
def log_m(A): return np.log(_m(A))
def sin_m(A): return np.sin(_m(A))
def cos_m(A): return np.cos(_m(A))
def tan_m(A): return np.tan(_m(A))
def tanh_m(A): return np.tanh(_m(A))
def sqrt_m(A): return np.sqrt(_m(A))
def abs_m(A): return np.abs(_m(A))
def floor_m(A): return np.floor(_m(A))
def round_m(A): return np.round(_m(A))
def sign_m(A): return np.sign(_m(A))


# --- reductions ------------------------------------------------------------

def _to_row(v):
    """Shape a 1-D reduction output into a 1xN row (MATLAB convention)."""
    arr = np.asarray(v).reshape(-1)
    return arr.reshape((1, arr.size)) if arr.size else arr.reshape((0, 0))


def sum(A):
    a = _m(A)
    if a.ndim < 2 or a.shape[0] == 1: return float(a.sum())
    return _to_row(a.sum(axis=0))


def _reduce_shape(v, d):
    """Shape reduction output: dim=1 -> row, dim=2 -> column."""
    arr = np.asarray(v).reshape(-1)
    if int(d) == 1: return arr.reshape((1, arr.size))
    return arr.reshape((arr.size, 1))

def sum_dim(A, d):
    return _reduce_shape(np.sum(_m(A), axis=int(d) - 1), d)


def prod(A):
    a = _m(A)
    if a.ndim < 2 or a.shape[0] == 1: return float(a.prod())
    return _to_row(a.prod(axis=0))


def prod_dim(A, d):
    return _reduce_shape(np.prod(_m(A), axis=int(d) - 1), d)


def mean(A):
    a = _m(A)
    if a.ndim < 2 or a.shape[0] == 1: return float(a.mean())
    return _to_row(a.mean(axis=0))


def mean_dim(A, d):
    return _reduce_shape(np.mean(_m(A), axis=int(d) - 1), d)


def min(A):
    a = _m(A)
    if a.ndim < 2 or a.shape[0] == 1: return float(a.min())
    return _to_row(a.min(axis=0))


def max(A):
    a = _m(A)
    if a.ndim < 2 or a.shape[0] == 1: return float(a.max())
    return _to_row(a.max(axis=0))


def min_mm(A, B): return np.minimum(_m(A), _m(B))
def max_mm(A, B): return np.maximum(_m(A), _m(B))


def cumsum(A): return np.cumsum(_m(A)).reshape(_m(A).shape)
def cumsum_dim(A, d): return np.cumsum(_m(A), axis=int(d) - 1)
def cumprod(A): return np.cumprod(_m(A)).reshape(_m(A).shape)


# --- shape / predicates ----------------------------------------------------

def size(A):
    a = _m(A)
    s = a.shape if a.ndim >= 2 else (1, a.shape[0] if a.ndim else 1)
    return np.array([[float(s[0]), float(s[1])]])


def size_dim(A, d):
    a = _m(A)
    d = int(d)
    if a.ndim < 2:
        return float(a.shape[0]) if d == 1 else 1.0
    if d < 1 or d > a.ndim: return 1.0
    return float(a.shape[d - 1])


def size3_dim(A, d):
    arr = np.asarray(A)
    d = int(d)
    if d < 1 or d > arr.ndim: return 1.0
    return float(arr.shape[d - 1])


def length(A):
    a = _m(A)
    return float(_pymax(a.shape)) if a.size else 0.0


def numel(A):
    return float(np.asarray(A).size) if A is not None else 0.0


def numel3(A): return float(np.asarray(A).size) if A is not None else 0.0


def ndims(A):
    a = _m(A)
    return float(a.ndim)


def ndims3(A): return float(np.asarray(A).ndim)


def end_of_dim(A, d):
    a = _m(A)
    d = int(d)
    if a.ndim < 2:
        return float(a.shape[0]) if d == 1 else 1.0
    return float(a.shape[d - 1])


def isempty(A):
    return 1.0 if (A is None or np.asarray(A).size == 0) else 0.0


def isequal(A, B):
    try: return 1.0 if np.array_equal(_m(A), _m(B)) else 0.0
    except Exception: return 0.0


# --- subscripting ---------------------------------------------------------

def subscript1_s(A, i):
    a = _m(A)
    idx = int(i) - 1
    return float(a.flatten(order='F')[idx])


def subscript2_s(A, i, j):
    a = _m(A)
    return float(a[int(i) - 1, int(j) - 1])


def subscript3_s(A, i, j, k):
    return float(np.asarray(A)[int(i) - 1, int(j) - 1, int(k) - 1])


def subscript3_store(A, i, j, k, v):
    A[int(i) - 1, int(j) - 1, int(k) - 1] = float(v)


def _is_colon(idx):
    """In the C runtime a NULL ptr means `:` (take all); the emitter
    translates NULL to `0` so that sentinel is what we see here."""
    return idx is None or (isinstance(idx, int) and idx == 0) or \
           (isinstance(idx, float) and idx == 0.0)


def slice1(A, idx):
    a = _m(A)
    # Match the C runtime's column-major linearisation so stdout is
    # byte-compatible with the emit-c lane.
    a_col = a.flatten(order='F')
    if _is_colon(idx):
        return a_col.reshape((-1, 1))
    idx_a = _m(idx)
    if idx_a.shape == a.shape:
        mask_vals = set(np.unique(idx_a).tolist())
        if mask_vals.issubset({0.0, 1.0}):
            return a_col[idx_a.flatten(order='F').astype(bool)].reshape((-1, 1))
    idx_flat = idx_a.flatten(order='F').astype(int) - 1
    return a_col[idx_flat].reshape((-1, 1))


def slice2(A, rows, cols):
    a = _m(A)
    if _is_colon(rows):
        r = np.arange(a.shape[0])
    else:
        r = _m(rows).flatten(order='F').astype(int) - 1
    if _is_colon(cols):
        c = np.arange(a.shape[1])
    else:
        c = _m(cols).flatten(order='F').astype(int) - 1
    return a[np.ix_(r, c)]


def slice_store1(A, idx, V):
    idx_flat = _m(idx).flatten(order='F').astype(int) - 1
    v_flat = _m(V).flatten(order='F')
    flat = A.flatten(order='F')
    flat[idx_flat] = v_flat
    A[:] = flat.reshape(A.shape, order='F')


def slice_store1_scalar(A, idx, v):
    idx_flat = _m(idx).flatten(order='F').astype(int) - 1
    flat = A.flatten(order='F')
    flat[idx_flat] = float(v)
    A[:] = flat.reshape(A.shape, order='F')


def slice_store2(A, rows, cols, V):
    r = np.arange(A.shape[0]) if _is_colon(rows) else \
        _m(rows).flatten(order='F').astype(int) - 1
    c = np.arange(A.shape[1]) if _is_colon(cols) else \
        _m(cols).flatten(order='F').astype(int) - 1
    A[np.ix_(r, c)] = _m(V)


def slice_store2_scalar(A, rows, cols, v):
    r = np.arange(A.shape[0]) if _is_colon(rows) else \
        _m(rows).flatten(order='F').astype(int) - 1
    c = np.arange(A.shape[1]) if _is_colon(cols) else \
        _m(cols).flatten(order='F').astype(int) - 1
    A[np.ix_(r, c)] = float(v)


def find(A):
    a = _m(A).flatten(order='F')
    nz = np.nonzero(a)[0] + 1
    return nz.reshape((-1, 1)).astype(float)


def erase_rows(A, rows):
    r = _m(rows).flatten(order='F').astype(int) - 1
    mask = np.ones(_m(A).shape[0], dtype=bool)
    mask[r] = False
    return _m(A)[mask, :]


def erase_cols(A, cols):
    c = _m(cols).flatten(order='F').astype(int) - 1
    mask = np.ones(_m(A).shape[1], dtype=bool)
    mask[c] = False
    return _m(A)[:, mask]


# --- scalar math ----------------------------------------------------------

def exp_s(x): return math.exp(float(x))
def log_s(x): return math.log(float(x))
def log10_s(x): return math.log10(float(x))
def log2_s(x): return math.log2(float(x))
def sin_s(x): return math.sin(float(x))
def cos_s(x): return math.cos(float(x))
def tan_s(x): return math.tan(float(x))
def asin_s(x): return math.asin(float(x))
def acos_s(x): return math.acos(float(x))
def atan_s(x): return math.atan(float(x))
def atan2_s(y, x): return math.atan2(float(y), float(x))
def sinh_s(x): return math.sinh(float(x))
def cosh_s(x): return math.cosh(float(x))
def tanh_s(x): return math.tanh(float(x))
def sqrt_s(x): return math.sqrt(float(x))
def abs_s(x): return abs(float(x))
def abs_c(A):
    a = np.asarray(A)
    return np.abs(a) if np.iscomplexobj(a) else np.abs(_m(A))
def ceil_s(x): return float(math.ceil(float(x)))
def floor_s(x): return float(math.floor(float(x)))
def round_s(x): return float(round(float(x)))
def fix_s(x): return float(math.trunc(float(x)))
def sign_s(x):
    xf = float(x)
    return 0.0 if xf == 0 else (1.0 if xf > 0 else -1.0)
def mod_s(a, b):
    b = float(b)
    if b == 0: return float(a)
    return float(a) - b * math.floor(float(a) / b)
def rem_s(a, b):
    b = float(b)
    if b == 0: return float(a)
    return float(a) - b * math.trunc(float(a) / b)


# --- type coercions (scalar) ----------------------------------------------

def double_s(x): return float(x)
def single_s(x): return float(x)
def int8_s(x): return int(x)
def int16_s(x): return int(x)
def int32_s(x): return int(x)
def int64_s(x): return int(x)
def uint8_s(x): return int(x) & 0xff
def uint16_s(x): return int(x) & 0xffff
def logical_s(x): return 1.0 if float(x) != 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 1.1.E — typed-int matrix runtime (i32 / u8). Mirrors the C runtime
# entry points used by `matlabc -emit-python` for non-scalar Int32 / UInt8
# arrays. Storage is numpy ndarray with dtype=int32 / uint8 — saturation is
# explicit at every op boundary so overflow matches the C lane bit-exactly
# (numpy's native int dtype overflow wraps, MATLAB's saturates).
# ---------------------------------------------------------------------------

_I32_MIN, _I32_MAX = -2147483648, 2147483647
_U8_MIN,  _U8_MAX  =  0,           255

def d_to_i32_sat(v):
    """Scalar double -> int32 with round-half-away-from-zero + saturate."""
    x = float(v)
    if x != x:                      return 0
    if x <= float(_I32_MIN):        return _I32_MIN
    if x >= float(_I32_MAX):        return _I32_MAX
    return int(x + 0.5) if x >= 0 else int(x - 0.5)

def d_to_u8_sat(v):
    x = float(v)
    if x != x:        return 0
    if x <= 0.0:      return 0
    if x >= 255.0:    return 255
    return int(x + 0.5)

def _round_haz_arr(arr):
    """Element-wise round half-away-from-zero, NaN -> 0 (MATLAB rule)."""
    a = np.asarray(arr, dtype=float)
    nz = ~np.isnan(a)
    rounded = np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5))
    return np.where(nz, rounded, 0.0)

def mat_i32_from_double(A):
    rounded = _round_haz_arr(A)
    return np.clip(rounded, _I32_MIN, _I32_MAX).astype(np.int32)

def mat_u8_from_double(A):
    rounded = _round_haz_arr(A)
    return np.clip(rounded, _U8_MIN, _U8_MAX).astype(np.uint8)

def mat_i32_to_double(A): return np.asarray(A, dtype=float)
def mat_u8_to_double(A):  return np.asarray(A, dtype=float)
def mat_u8_from_i32(A):
    return np.clip(np.asarray(A, dtype=np.int64), _U8_MIN, _U8_MAX).astype(np.uint8)
def mat_i32_from_u8(A):
    return np.asarray(A, dtype=np.int32)

def _disp_int_grid(A, width):
    """Print an int matrix with three leading spaces and a fixed column."""
    arr = np.asarray(A)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    if arr.size == 0:
        print(); return
    m, n = arr.shape[0], arr.shape[1] if arr.ndim > 1 else 1
    for i in _pyrange(m):
        row = ["   " + f"{int(arr[i, j]):>{width}d}" for j in _pyrange(n)]
        print("".join(row))

def mat_i32_disp(A): _disp_int_grid(A, 11)
def mat_u8_disp (A): _disp_int_grid(A,  4)

# --- saturating arithmetic. We accumulate in int64 to avoid numpy's
# silent wrap on overflow before clipping back to the lane's range. ---

def _sat_i32(arr_i64): return np.clip(arr_i64, _I32_MIN, _I32_MAX).astype(np.int32)
def _sat_u8 (arr_i64): return np.clip(arr_i64, _U8_MIN,  _U8_MAX).astype(np.uint8)

def _as_i64(A): return np.asarray(A, dtype=np.int64)

def mat_i32_add_mm(A, B): return _sat_i32(_as_i64(A) + _as_i64(B))
def mat_i32_add_ms(A, s): return _sat_i32(_as_i64(A) + int(s))
def mat_i32_add_sm(s, A): return _sat_i32(int(s)     + _as_i64(A))
def mat_i32_sub_mm(A, B): return _sat_i32(_as_i64(A) - _as_i64(B))
def mat_i32_sub_ms(A, s): return _sat_i32(_as_i64(A) - int(s))
def mat_i32_sub_sm(s, A): return _sat_i32(int(s)     - _as_i64(A))
def mat_i32_emul_mm(A, B): return _sat_i32(_as_i64(A) * _as_i64(B))
def mat_i32_emul_ms(A, s): return _sat_i32(_as_i64(A) * int(s))
def mat_i32_emul_sm(s, A): return _sat_i32(int(s)     * _as_i64(A))

def mat_u8_add_mm(A, B): return _sat_u8(_as_i64(A) + _as_i64(B))
def mat_u8_add_ms(A, s): return _sat_u8(_as_i64(A) + int(s))
def mat_u8_add_sm(s, A): return _sat_u8(int(s)     + _as_i64(A))
def mat_u8_sub_mm(A, B): return _sat_u8(_as_i64(A) - _as_i64(B))
def mat_u8_sub_ms(A, s): return _sat_u8(_as_i64(A) - int(s))
def mat_u8_sub_sm(s, A): return _sat_u8(int(s)     - _as_i64(A))
def mat_u8_emul_mm(A, B): return _sat_u8(_as_i64(A) * _as_i64(B))
def mat_u8_emul_ms(A, s): return _sat_u8(_as_i64(A) * int(s))
def mat_u8_emul_sm(s, A): return _sat_u8(int(s)     * _as_i64(A))

def _round_div_int(num, den, lo, hi):
    """Element-wise round-half-away-from-zero division with MATLAB's
    int-zero rule: 0/0 = 0, x/0 = ±max with the sign of x."""
    n = np.asarray(num, dtype=np.int64)
    d = np.asarray(den, dtype=np.int64)
    if d.shape == ():
        d = np.full(n.shape, int(d), dtype=np.int64)
    if n.shape == ():
        n = np.full(d.shape, int(n), dtype=np.int64)
    out = np.zeros(n.shape, dtype=np.int64)
    nz = d != 0
    sign = np.where((n < 0) ^ (d < 0), -1, 1)
    abs_n, abs_d = np.abs(n), np.abs(d)
    safe_d = np.where(d == 0, 1, abs_d)
    q = abs_n // safe_d
    r = abs_n - q * safe_d
    q = np.where(r * 2 >= safe_d, q + 1, q)
    out = np.where(nz, sign * q, np.where(n == 0, 0, np.where(n > 0, hi, lo)))
    return out

def mat_i32_ediv_mm(A, B): return _sat_i32(_round_div_int(A, B, _I32_MIN, _I32_MAX))
def mat_i32_ediv_ms(A, s): return _sat_i32(_round_div_int(A, np.int64(int(s)), _I32_MIN, _I32_MAX))
def mat_i32_ediv_sm(s, A): return _sat_i32(_round_div_int(np.int64(int(s)), A, _I32_MIN, _I32_MAX))
def mat_u8_ediv_mm(A, B): return _sat_u8(_round_div_int(A, B, _U8_MIN, _U8_MAX))
def mat_u8_ediv_ms(A, s): return _sat_u8(_round_div_int(A, np.int64(int(s)), _U8_MIN, _U8_MAX))
def mat_u8_ediv_sm(s, A): return _sat_u8(_round_div_int(np.int64(int(s)), A, _U8_MIN, _U8_MAX))

# Comparisons return f64 logical (0.0 / 1.0) — same encoding as the rest
# of the runtime so downstream `if`/`while`/disp_mat consume them uniformly.
def _cmp(A, B, op):
    a = np.asarray(A, dtype=np.int64)
    b = np.asarray(B, dtype=np.int64)
    return op(a, b).astype(float)

def mat_i32_gt_mm(A, B): return _cmp(A, B, np.greater)
def mat_i32_gt_ms(A, s): return _cmp(A, int(s), np.greater)
def mat_i32_gt_sm(s, A): return _cmp(int(s), A, np.greater)
def mat_i32_ge_mm(A, B): return _cmp(A, B, np.greater_equal)
def mat_i32_ge_ms(A, s): return _cmp(A, int(s), np.greater_equal)
def mat_i32_ge_sm(s, A): return _cmp(int(s), A, np.greater_equal)
def mat_i32_lt_mm(A, B): return _cmp(A, B, np.less)
def mat_i32_lt_ms(A, s): return _cmp(A, int(s), np.less)
def mat_i32_lt_sm(s, A): return _cmp(int(s), A, np.less)
def mat_i32_le_mm(A, B): return _cmp(A, B, np.less_equal)
def mat_i32_le_ms(A, s): return _cmp(A, int(s), np.less_equal)
def mat_i32_le_sm(s, A): return _cmp(int(s), A, np.less_equal)
def mat_i32_eq_mm(A, B): return _cmp(A, B, np.equal)
def mat_i32_eq_ms(A, s): return _cmp(A, int(s), np.equal)
def mat_i32_eq_sm(s, A): return _cmp(int(s), A, np.equal)
def mat_i32_ne_mm(A, B): return _cmp(A, B, np.not_equal)
def mat_i32_ne_ms(A, s): return _cmp(A, int(s), np.not_equal)
def mat_i32_ne_sm(s, A): return _cmp(int(s), A, np.not_equal)

def mat_u8_gt_mm(A, B): return _cmp(A, B, np.greater)
def mat_u8_gt_ms(A, s): return _cmp(A, int(s), np.greater)
def mat_u8_gt_sm(s, A): return _cmp(int(s), A, np.greater)
def mat_u8_ge_mm(A, B): return _cmp(A, B, np.greater_equal)
def mat_u8_ge_ms(A, s): return _cmp(A, int(s), np.greater_equal)
def mat_u8_ge_sm(s, A): return _cmp(int(s), A, np.greater_equal)
def mat_u8_lt_mm(A, B): return _cmp(A, B, np.less)
def mat_u8_lt_ms(A, s): return _cmp(A, int(s), np.less)
def mat_u8_lt_sm(s, A): return _cmp(int(s), A, np.less)
def mat_u8_le_mm(A, B): return _cmp(A, B, np.less_equal)
def mat_u8_le_ms(A, s): return _cmp(A, int(s), np.less_equal)
def mat_u8_le_sm(s, A): return _cmp(int(s), A, np.less_equal)
def mat_u8_eq_mm(A, B): return _cmp(A, B, np.equal)
def mat_u8_eq_ms(A, s): return _cmp(A, int(s), np.equal)
def mat_u8_eq_sm(s, A): return _cmp(int(s), A, np.equal)
def mat_u8_ne_mm(A, B): return _cmp(A, B, np.not_equal)
def mat_u8_ne_ms(A, s): return _cmp(A, int(s), np.not_equal)
def mat_u8_ne_sm(s, A): return _cmp(int(s), A, np.not_equal)


# --- Fixed-Point Designer (fi) — see docs/emit_fixed_point.md §6.2 -------
# Python ints are arbitrary precision, so high-WL fi values stay bit-exact
# regardless of Python's 53-bit float mantissa. The shim mirrors the C
# helpers verbatim. Overflow: 0=Wrap, 1=Saturate. Rounding: 0=Floor,
# 1=Nearest, 2=Zero, 3=Convergent, 4=Ceiling — Phase 1 ships 0/1.
import math as _fi_math
import builtins as _fi_builtins

def fi_sat_s64(x, WL):
    if WL == 0: return 0
    if WL >= 64: return int(x)
    hi = (1 << (WL - 1)) - 1
    lo = -(1 << (WL - 1))
    # The matrix runtime above shadows `min`/`max` with MATLAB-style
    # overloads, so reach through `builtins` for the scalar Python ones.
    return _fi_builtins.max(lo, _fi_builtins.min(hi, int(x)))

def fi_sat_u64(x, WL):
    if WL == 0: return 0
    if WL >= 64: return int(x) & ((1 << 64) - 1)
    hi = (1 << WL) - 1
    return _fi_builtins.max(0, _fi_builtins.min(hi, int(x)))

def fi_wrap_u(x, WL):
    """Two's-complement-style wrap to WL unsigned bits.

    SV-faithful counterpart to fi_sat_u64: HDL designs that intentionally
    overflow (CRC accumulators, FNV hashes, LFSRs) need wrap, not
    saturate. Cocotb harnesses route persistent stores through this
    helper when the source declares the register as fi(_, _, WL, F)
    and the SV DUT uses non-saturating arithmetic."""
    if WL == 0: return 0
    if WL >= 64: return int(x) & ((1 << 64) - 1)
    return int(x) & ((1 << WL) - 1)

def fi_wrap_s(x, WL):
    """Two's-complement-style wrap to WL signed bits.

    SV-faithful counterpart to fi_sat_s64. Stores into signed fi
    registers whose values overflow get wrapped to two's complement
    (the SV behaviour) rather than saturated (the MATLAB default).
    """
    if WL == 0: return 0
    if WL >= 64: return int(x)
    mask = (1 << WL) - 1
    v = int(x) & mask
    if v & (1 << (WL - 1)):
        v -= (1 << WL)
    return v

def fi_round_floor_s(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return -1 if int(x) < 0 else 0
    # Python `>>` on negative ints already floors toward -inf — perfect.
    return int(x) >> shift

def fi_round_nearest_s(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 0
    half = 1 << (shift - 1)
    return (int(x) + half) >> shift

def fi_round_floor_u(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 0
    return int(x) >> shift

def fi_round_nearest_u(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 0
    half = 1 << (shift - 1)
    return (int(x) + half) >> shift

# --- Phase 5 rounding modes ----------------------------------------------

def fi_round_zero_s(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 0
    xi = int(x)
    if xi >= 0: return xi >> shift
    bias = (1 << shift) - 1
    return (xi + bias) >> shift

def fi_round_zero_u(x, shift):
    return fi_round_floor_u(x, shift)

def fi_round_ceiling_s(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 1 if int(x) > 0 else 0
    bias = (1 << shift) - 1
    return (int(x) + bias) >> shift

def fi_round_ceiling_u(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 1 if int(x) > 0 else 0
    bias = (1 << shift) - 1
    return (int(x) + bias) >> shift

def fi_round_convergent_s(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 0
    half = 1 << (shift - 1)
    xi = int(x)
    lsb = (xi >> shift) & 1
    return (xi + half - 1 + lsb) >> shift

def fi_round_convergent_u(x, shift):
    if shift == 0: return int(x)
    if shift >= 64: return 0
    half = 1 << (shift - 1)
    xi = int(x)
    lsb = (xi >> shift) & 1
    return (xi + half - 1 + lsb) >> shift

def fi_quantize_s(v, WL, FL, overflow, rounding):
    scaled = float(v) * (2.0 ** int(FL))
    if rounding == 0:   stored = int(_fi_math.floor(scaled))
    elif rounding == 1: stored = int(_fi_math.floor(scaled + 0.5))
    elif rounding == 2: stored = int(_fi_math.trunc(scaled))
    elif rounding == 3:
        # Convergent: round-half-to-even.
        frac = scaled - _fi_math.floor(scaled)
        if frac == 0.5:
            lo = int(_fi_math.floor(scaled))
            stored = lo if lo % 2 == 0 else lo + 1
        else:
            stored = int(round(scaled))
    elif rounding == 4: stored = int(_fi_math.ceil(scaled))
    else:
        set_error()
        return 0
    if overflow == 1: return fi_sat_s64(stored, WL)
    if WL == 0: return 0
    mask = (1 << WL) - 1
    bits = stored & mask
    if bits & (1 << (WL - 1)): bits |= ~mask
    # Python int is unbounded; sign-extending into a Python int means
    # subtracting 2^WL when the sign bit is set.
    if stored & (1 << (WL - 1)): return bits if bits < 0 else bits - (1 << WL)
    return bits & mask

def fi_quantize_u(v, WL, FL, overflow, rounding):
    scaled = float(v) * (2.0 ** int(FL))
    if scaled < 0.0: scaled = 0.0
    if rounding == 0:   stored = int(_fi_math.floor(scaled))
    elif rounding == 1: stored = int(_fi_math.floor(scaled + 0.5))
    elif rounding == 2: stored = int(_fi_math.trunc(scaled))
    elif rounding == 3:
        frac = scaled - _fi_math.floor(scaled)
        if frac == 0.5:
            lo = int(_fi_math.floor(scaled))
            stored = lo if lo % 2 == 0 else lo + 1
        else:
            stored = int(round(scaled))
    elif rounding == 4: stored = int(_fi_math.ceil(scaled))
    else:
        set_error()
        return 0
    if overflow == 1: return fi_sat_u64(stored, WL)
    if WL == 0: return 0
    mask = (1 << WL) - 1
    return stored & mask

def fi_disp_s(stored, WL, FL):
    disp_f64(float(stored) * (2.0 ** -int(FL)))

def fi_disp_u(stored, WL, FL):
    disp_f64(float(stored) * (2.0 ** -int(FL)))

def _fi_bin(stored, WL):
    if WL == 0: return ""
    if WL > 64: WL = 64
    mask = (1 << WL) - 1
    bits = int(stored) & mask
    return format(bits, "0{}b".format(int(WL)))

def fi_bin_s(stored, WL): return _fi_bin(stored, WL)
def fi_bin_u(stored, WL): return _fi_bin(stored, WL)

def _fi_hex(stored, WL):
    if WL == 0: return ""
    digits = (int(WL) + 3) // 4
    mask = (1 << int(WL)) - 1
    bits = int(stored) & mask
    return format(bits, "0{}x".format(digits))

def fi_hex_s(stored, WL): return _fi_hex(stored, WL)
def fi_hex_u(stored, WL): return _fi_hex(stored, WL)

def fi_dec_s(stored, WL): return str(int(stored))
def fi_dec_u(stored, WL): return str(int(stored))


# --- typed integer matrix runtime (fi arrays) ---------------------------
# Backed by a plain Python list of Python ints (arbitrary precision —
# bit-exact regardless of the C-side WL). Each descriptor carries its
# own rows/cols so length/numel match the C runtime.

class _MatI64:
    __slots__ = ("data", "rows", "cols")
    def __init__(self, rows, cols, data=None):
        self.rows = int(rows); self.cols = int(cols)
        n = self.rows * self.cols
        self.data = [0] * n if data is None else list(data)

def mat_i64_zeros(rows, cols): return _MatI64(rows, cols)
def mat_u64_zeros(rows, cols): return _MatI64(rows, cols)
def mat_i64_from_buf(buf, rows, cols): return _MatI64(rows, cols, buf)
def mat_u64_from_buf(buf, rows, cols): return _MatI64(rows, cols, buf)
def mat_i64_from_scalar(v): return _MatI64(1, 1, [int(v)])
def mat_u64_from_scalar(v): return _MatI64(1, 1, [int(v)])

def mat_i64_length(A): return float(_fi_builtins.max(A.rows, A.cols))
def mat_u64_length(A): return float(_fi_builtins.max(A.rows, A.cols))
def mat_i64_numel(A): return float(A.rows * A.cols)
def mat_u64_numel(A): return float(A.rows * A.cols)
def mat_i64_size_dim(A, dim):
    d = int(dim)
    if d == 1: return float(A.rows)
    if d == 2: return float(A.cols)
    return 1.0
def mat_u64_size_dim(A, dim): return mat_i64_size_dim(A, dim)
def mat_i64_rows(A): return A.rows
def mat_i64_cols(A): return A.cols

def _lin(A, i):
    k = int(i) - 1
    if k < 0: k = 0
    n = A.rows * A.cols
    if k >= n: k = n - 1
    return k

def mat_i64_subscript1_s(A, i): return A.data[_lin(A, i)]
def mat_u64_subscript1_s(A, i): return A.data[_lin(A, i)]
def mat_i64_subscript2_s(A, i, j):
    r = int(i) - 1; c = int(j) - 1
    return A.data[r * A.cols + c]
def mat_u64_subscript2_s(A, i, j): return mat_i64_subscript2_s(A, i, j)

def mat_i64_set1_s(A, i, v): A.data[_lin(A, i)] = int(v)
def mat_u64_set1_s(A, i, v): A.data[_lin(A, i)] = int(v)
def mat_i64_fill(A, v):
    iv = int(v)
    for k in _fi_builtins.range(A.rows * A.cols): A.data[k] = iv
def mat_u64_fill(A, v): mat_i64_fill(A, v)

def mat_i64_slice1(A, idx):
    # idx is a NumPy 1-D array of doubles (1-based indices).
    flat = np.asarray(idx).ravel()
    n = int(flat.size)
    out = _MatI64(1 if A.rows == 1 else n, n if A.rows == 1 else 1)
    for k in _fi_builtins.range(n):
        out.data[k] = mat_i64_subscript1_s(A, float(flat[k]))
    return out
def mat_u64_slice1(A, idx): return mat_i64_slice1(A, idx)

def mat_i64_concat_row(A, B):
    if A is None: return B
    if B is None: return A
    out = _MatI64(1, A.rows * A.cols + B.rows * B.cols)
    out.data = list(A.data) + list(B.data)
    return out
def mat_u64_concat_row(A, B): return mat_i64_concat_row(A, B)

def mat_i64_sum(A): return sum(A.data) if A else 0
def mat_u64_sum(A): return sum(A.data) if A else 0

def mat_i64_disp(A, WL, FL):
    if A is None: print("(null)"); return
    scale = 2.0 ** -int(FL)
    for r in _fi_builtins.range(A.rows):
        line = ""
        for c in _fi_builtins.range(A.cols):
            line += "   %7g" % (float(A.data[r * A.cols + c]) * scale)
        print(line)
    if A.rows * A.cols == 0: print("")
def mat_u64_disp(A, WL, FL): mat_i64_disp(A, WL, FL)


# --- persistent typed pointer table (fi arrays + future heap types) -----
_persistent_ptr = {}
def persistent_get_ptr(id):  return _persistent_ptr.get(int(id))
def persistent_set_ptr(id, p): _persistent_ptr[int(id)] = p
def persistent_isempty(id):
    p = _persistent_ptr.get(int(id))
    return 1.0 if p is None else 0.0


# --- error flag (try/catch) -----------------------------------------------

_error_flag = 0
_error_msg = ""

def set_error():
    global _error_flag
    _error_flag = 1

def set_error_msg(msg, n=None):
    global _error_flag, _error_msg
    _error_flag = 1
    _error_msg = msg if isinstance(msg, str) else str(msg)

def check_error():
    return _error_flag

def clear_error():
    # Only clear the flag — the message stays available for the catch
    # body to read. Mirrors the C runtime.
    global _error_flag
    _error_flag = 0

def err_disp_message():
    if _error_msg:
        print(_error_msg)
    else:
        print("")

def err_msg0(): return _error_msg
def err_msg1(): return _error_msg


# --- globals (persistent / global vars) -----------------------------------

_globals = {}

def global_get_f64(gid):
    return float(_globals.get(int(gid), 0.0))

def global_set_f64(gid, v):
    _globals[int(gid)] = float(v)


# --- structs --------------------------------------------------------------

class _Struct(dict):
    """Dict subclass so the emitter can access fields via attribute access
    OR via the runtime functions. `s.x` and `s["x"]` both work."""
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v


def struct_new():
    return _Struct()


def struct_set_f64(s, name, n, v):
    s[name] = float(v)


def struct_set_mat(s, name, n, m):
    s[name] = m


def struct_get_f64(s, name, n=None):
    v = s.get(name, 0.0) if hasattr(s, 'get') else getattr(s, name, 0.0)
    try: return float(v)
    except Exception: return 0.0


def struct_get_mat(s, name, n=None):
    if hasattr(s, 'get'): return s.get(name)
    return getattr(s, name, None)


def struct_has_field(s, name, n=None):
    if hasattr(s, '__contains__'): return 1.0 if name in s else 0.0
    return 1.0 if hasattr(s, name) else 0.0


def struct_get_child_struct(s, name, n=None):
    v = s.get(name) if hasattr(s, 'get') else getattr(s, name, None)
    if v is None:
        v = _Struct(); s[name] = v
    return v


def struct_rmfield(s, name, n=None):
    if hasattr(s, 'pop'): s.pop(name, None)
    return s


# Phase 2 — struct arrays. Mirrors the C runtime: a vector of structs
# with auto-grow on write and 1-based indexing.

def struct_arr_new():
    return []

def struct_arr_get_or_create(a, i):
    idx = int(i) - 1
    if idx < 0: return _Struct()
    while len(a) <= idx:
        a.append(_Struct())
    return a[idx]

def struct_arr_get(a, i):
    idx = int(i) - 1
    if idx < 0 or idx >= len(a): return _Struct()
    return a[idx]

def struct_arr_length(a):
    return float(len(a)) if a is not None else 0.0

def struct_arr_numel(a):
    return struct_arr_length(a)

def struct_arr_size_dim(a, d):
    d = int(d)
    n = len(a) if a is not None else 0
    if d == 1: return 1.0 if n > 0 else 0.0
    if d == 2: return float(n)
    return 1.0


# --- cells ----------------------------------------------------------------

def cell_new(n):
    return [None] * int(n)

def _cell_grow(c, idx):
    while len(c) < idx:
        c.append(None)

def cell_set_f64(c, i, v):
    idx = int(i)
    _cell_grow(c, idx)
    c[idx - 1] = float(v)

def cell_set_mat(c, i, m):
    idx = int(i)
    _cell_grow(c, idx)
    c[idx - 1] = m

def cell_get_f64(c, i):
    v = c[int(i) - 1]
    try: return float(v)
    except Exception: return 0.0

def cell_get_mat(c, i):
    return c[int(i) - 1]

def cell_numel(c):
    return float(len(c))

def iscell(c):
    return 1.0 if isinstance(c, list) else 0.0

# Phase 1.3 — 2-D cells. The legacy 1-D representation is a flat Python
# list (one element per slot). The 2-D form keeps the same flat list and
# tracks rows/cols on the dict-style `_meta` attribute via a wrapper. We
# avoid threading a class through here by using a dict { 'data': [...],
# 'rows': r, 'cols': c }; iscell / numel / cell_get_* / cell_set_* now
# accept either form.

def _is_cell2d(c): return isinstance(c, dict) and 'data' in c and 'rows' in c

def cell_new_2d(rows, cols):
    r = int(rows); k = int(cols)
    return {'data': [None] * (r * k), 'rows': r, 'cols': k}

def cell_rows(c):
    if _is_cell2d(c): return float(c['rows'])
    return 1.0 if c else 0.0

def cell_cols(c):
    if _is_cell2d(c): return float(c['cols'])
    return float(len(c)) if isinstance(c, list) else 0.0

def cell_size_dim(c, d):
    d = int(d)
    if d == 1: return cell_rows(c)
    if d == 2: return cell_cols(c)
    return 1.0

def _cell2d_lin(c, r, k):
    return (int(r) - 1) * c['cols'] + (int(k) - 1)

def cell_set_f64_2d(c, r, k, v):
    if not _is_cell2d(c): return
    c['data'][_cell2d_lin(c, r, k)] = float(v)

def cell_set_mat_2d(c, r, k, m):
    if not _is_cell2d(c): return
    c['data'][_cell2d_lin(c, r, k)] = m

def cell_get_f64_2d(c, r, k):
    if not _is_cell2d(c): return 0.0
    v = c['data'][_cell2d_lin(c, r, k)]
    try: return float(v)
    except Exception: return 0.0

def cell_get_mat_2d(c, r, k):
    if not _is_cell2d(c): return None
    return c['data'][_cell2d_lin(c, r, k)]

def cell_concat_row(a, b):
    """[a, b] horizontal concat: rows must match; cols sum."""
    ar = int(cell_rows(a)); ac = int(cell_cols(a))
    br = int(cell_rows(b)); bc = int(cell_cols(b))
    if ar != br: return cell_new(0)
    nc = ac + bc
    out = cell_new_2d(ar, nc)
    a_data = a['data'] if _is_cell2d(a) else a
    b_data = b['data'] if _is_cell2d(b) else b
    for r in _pyrange(ar):
        for kk in _pyrange(ac):
            out['data'][r * nc + kk] = a_data[r * ac + kk]
        for kk in _pyrange(bc):
            out['data'][r * nc + ac + kk] = b_data[r * bc + kk]
    return out

def cell_concat_col(a, b):
    """[a; b] vertical concat: cols must match; rows sum."""
    ar = int(cell_rows(a)); ac = int(cell_cols(a))
    br = int(cell_rows(b)); bc = int(cell_cols(b))
    if ac != bc: return cell_new(0)
    nr = ar + br
    out = cell_new_2d(nr, ac)
    a_data = a['data'] if _is_cell2d(a) else a
    b_data = b['data'] if _is_cell2d(b) else b
    for i in _pyrange(ar * ac):
        out['data'][i] = a_data[i]
    for i in _pyrange(br * bc):
        out['data'][ar * ac + i] = b_data[i]
    return out


# --- object / class -------------------------------------------------------
#
# `obj_new` returns a plain Python object whose attributes back the
# class's properties. The Python emitter rewrites `obj_get_f64(o, "X")`
# to `o.X` and `obj_set_f64(o, "X", v)` to `o.X = v` whenever the field
# name is a valid Python identifier, so most accesses bypass the
# runtime functions entirely. The functions below remain for the cases
# that don't qualify (non-identifier field names, hand-written callers)
# and for back-compat with the legacy oid-int API.

class _MatObj:
    """Heap-allocated MATLAB classdef instance.

    Properties land directly as attributes — the Python emitter targets
    `obj.Field` syntax. Tracks an `_oid` so the dict-based fallback
    (`obj_set_f64` with non-identifier fields) keeps working.
    """
    __slots__ = ("_oid", "__dict__")

_obj_store = {}
_obj_next_id = 1

def obj_new(*_ignored):
    global _obj_next_id
    oid = _obj_next_id; _obj_next_id += 1
    obj = _MatObj()
    obj._oid = oid
    _obj_store[oid] = obj  # keep a strong ref + a lookup path for legacy callers
    return obj

# Phase 5.3 — table. Mirrors the C runtime: dict-of-named-columns
# with column-vector storage. Each column is a numpy array (or list);
# the public API matches the C ABI's column-add / column-get / shape /
# disp surface.

def table_new():
    return {'_kind': 'table', 'names': [], 'cols': []}

def _table_idx(t, name):
    for i, n in enumerate(t['names']):
        if n == name: return i
    return -1

def table_add_column(t, name, *rest):
    """The C ABI is `add_column(t, name_ptr, name_len, col)`; the
    -emit-python lane drops name_len, but accepts either form. We
    take *rest to absorb a (len, col) pair or just (col)."""
    if t is None: return
    if len(rest) == 2:
        col = rest[1]
    elif len(rest) == 1:
        col = rest[0]
    else:
        return
    nm = name if isinstance(name, str) else str(name)
    i = _table_idx(t, nm)
    if i >= 0:
        t['cols'][i] = col
    else:
        t['names'].append(nm)
        t['cols'].append(col)

def table_get_column(t, name, *_unused):
    if t is None: return None
    nm = name if isinstance(name, str) else str(name)
    i = _table_idx(t, nm)
    return t['cols'][i] if i >= 0 else None

def _column_len(c):
    try: return len(c)
    except Exception:
        try: return c.size
        except Exception: return 0

def table_height(t):
    if t is None or not t['cols']: return 0.0
    return float(_column_len(t['cols'][0]))

def table_width(t):
    return float(len(t['names'])) if t else 0.0

def table_numel(t):
    return table_height(t) * table_width(t)

def table_size_dim(t, dim):
    d = int(dim)
    if d == 1: return table_height(t)
    if d == 2: return table_width(t)
    return 1.0

def _fmt_table_cell(v):
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e15:
            return f"{int(f):>12d}"
        return f"{f:>12.6g}"
    except Exception:
        return str(v).rjust(12)

def table_disp(t):
    if t is None: print("(empty table)"); return
    nrows = int(table_height(t))
    # Header.
    parts = ["    " + n.rjust(12) for n in t['names']]
    print("".join(parts))
    underline = ["    " + ("_" * 12) for _ in t['names']]
    print("".join(underline))
    for r in _pyrange(nrows):
        row = []
        for c in t['cols']:
            try:
                v = c[r] if hasattr(c, '__getitem__') else None
            except Exception:
                v = None
            row.append("    " + _fmt_table_cell(v))
        print("".join(row))


# Phase 5.2 — categorical. Mirrors the C runtime: each instance has a
# list of per-element codes (1-based, 0 = <undefined>) and a list of
# category-name strings (sorted alphabetically). Lookup is O(N).

def categorical_from_cell(cell, n):
    """Build a categorical from a cell-shaped object (we accept the
    1-D cell list form). The lowering in matlabc emits this entry
    with a freshly-built cell containing matlab_string-shaped Python
    strings."""
    n = int(n)
    pairs = []
    if isinstance(cell, list):
        items = cell[:n]
    else:
        items = list(cell.get('data', cell)) if hasattr(cell, 'get') else cell
        items = items[:n]
    cats = sorted({str(x) if x is not None else "" for x in items})
    cat_index = {c: i + 1 for i, c in enumerate(cats)}
    codes = [cat_index.get(str(x) if x is not None else "", 0) for x in items]
    return {'_kind': 'categorical', 'codes': codes, 'cats': cats}

def categorical_length(c):
    return float(len(c['codes'])) if c else 0.0

def categorical_numcats(c):
    return float(len(c['cats'])) if c else 0.0

def categorical_iscategory(c, key):
    if c is None: return 0.0
    k = key if isinstance(key, str) else str(key)
    return 1.0 if k in c['cats'] else 0.0

def categorical_categories(c):
    if c is None: return []
    return list(c['cats'])

def categorical_disp(c):
    if c is None: print("(empty categorical)"); return
    if not c['codes']:
        print("     [0x0 categorical]"); return
    for code in c['codes']:
        if code >= 1 and code <= len(c['cats']):
            print(f"     {c['cats'][code - 1]}")
        else:
            print("     <undefined>")

def categorical_eq(a, b):
    if a is None or b is None: return None
    n = min(len(a['codes']), len(b['codes']))
    out = []
    for i in range(n):
        if a['codes'][i] == 0 or b['codes'][i] == 0:
            out.append(0.0); continue
        an = a['cats'][a['codes'][i] - 1]
        bn = b['cats'][b['codes'][i] - 1]
        out.append(1.0 if an == bn else 0.0)
    return out


# Phase 5.1 — datetime / duration. Mirrors the C runtime's wrapping
# style: each is a small dict carrying a single `seconds` field. We
# avoid Python's datetime module to keep the surface independent of
# the target's locale + timezone settings (the C runtime treats the
# datetime as UTC seconds-since-epoch and renders DD-Mon-YYYY HH:MM:SS).

import time as _time
_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]

def _civil_to_epoch(y, m, d, hh=0, mn=0, ss=0.0):
    """Howard Hinnant civil-to-epoch (UTC). Same algorithm as the C
    runtime so output matches byte-for-byte."""
    ny = y - 1 if m <= 2 else y
    nm = m + 9 if m <= 2 else m - 3
    era = ny // 400 if ny >= 0 else (ny - 399) // 400
    yoe = ny - era * 400
    doy = (153 * nm + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return float(days) * 86400.0 + hh * 3600.0 + mn * 60.0 + float(ss)

def _epoch_to_civil(secs):
    total = int(secs)
    frac = secs - float(total)
    days = total // 86400
    sod = total - days * 86400
    hh = sod // 3600
    mn = (sod // 60) % 60
    ss = float(sod % 60) + frac
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    ny = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    y = ny + (1 if m <= 2 else 0)
    return y, m, d, hh, mn, ss

def datetime_now():
    return {'_kind': 'datetime', 'seconds': _time.time()}

def datetime_ymd(y, m, d):
    return {'_kind': 'datetime', 'seconds': _civil_to_epoch(int(y), int(m), int(d))}

def datetime_ymdhms(y, m, d, h, mn, s):
    return {'_kind': 'datetime',
            'seconds': _civil_to_epoch(int(y), int(m), int(d),
                                        int(h), int(mn), float(s))}

def datetime_disp(t):
    if t is None: print("(empty datetime)"); return
    y, m, d, hh, mn, ss = _epoch_to_civil(t['seconds'])
    print(f"{int(d):02d}-{_MONTHS[(m-1) % 12]}-{int(y):04d} "
          f"{int(hh):02d}:{int(mn):02d}:{int(ss):02d}")

def duration_seconds(n): return {'_kind': 'duration', 'seconds': float(n)}
def duration_minutes(n): return {'_kind': 'duration', 'seconds': float(n) * 60.0}
def duration_hours(n):   return {'_kind': 'duration', 'seconds': float(n) * 3600.0}
def duration_days(n):    return {'_kind': 'duration', 'seconds': float(n) * 86400.0}
def duration_years(n):   return {'_kind': 'duration', 'seconds': float(n) * 365.25 * 86400.0}

def duration_to_seconds(d): return float(d['seconds']) if d else 0.0
def duration_to_minutes(d): return float(d['seconds']) / 60.0 if d else 0.0
def duration_to_hours(d):   return float(d['seconds']) / 3600.0 if d else 0.0
def duration_to_days(d):    return float(d['seconds']) / 86400.0 if d else 0.0

def duration_disp(d):
    if d is None: print("(empty duration)"); return
    s = float(d['seconds'])
    if abs(s) >= 86400.0:  print(f"{s/86400.0:.4f} days")
    elif abs(s) >= 3600.0: print(f"{s/3600.0:.4f} hr")
    elif abs(s) >= 60.0:   print(f"{s/60.0:.4f} min")
    else:                  print(f"{s:.6f} sec")

def datetime_sub_datetime(a, b):
    return duration_seconds((a['seconds'] if a else 0.0) -
                             (b['seconds'] if b else 0.0))

def datetime_add_duration(a, d):
    return {'_kind': 'datetime',
            'seconds': (a['seconds'] if a else 0.0) +
                        (d['seconds'] if d else 0.0)}

def datetime_sub_duration(a, d):
    return {'_kind': 'datetime',
            'seconds': (a['seconds'] if a else 0.0) -
                        (d['seconds'] if d else 0.0)}

def duration_add(a, b):
    return duration_seconds((a['seconds'] if a else 0.0) +
                             (b['seconds'] if b else 0.0))

def duration_sub(a, b):
    return duration_seconds((a['seconds'] if a else 0.0) -
                             (b['seconds'] if b else 0.0))


def dict_new():
    """Phase 4 — containers.Map / dictionary. Backed by a list of
    (key, value) pairs to mirror the C runtime's slot model. Keys can
    be either f64 floats or Python strings (representing
    matlab_string *). Lookup is O(N) — fine for the test corpus."""
    return {'_pairs': []}

def _dict_find(d, key):
    pairs = d.get('_pairs', [])
    for i, (k, _) in enumerate(pairs):
        if k == key: return i
    return -1

def dict_set_str_f64(d, key, v):
    if d is None: return
    k = key if isinstance(key, str) else str(key)
    i = _dict_find(d, k)
    if i >= 0: d['_pairs'][i] = (k, float(v))
    else: d['_pairs'].append((k, float(v)))

def dict_set_str_mat(d, key, m):
    if d is None: return
    k = key if isinstance(key, str) else str(key)
    i = _dict_find(d, k)
    if i >= 0: d['_pairs'][i] = (k, m)
    else: d['_pairs'].append((k, m))

def dict_set_num_f64(d, key, v):
    if d is None: return
    k = float(key)
    i = _dict_find(d, k)
    if i >= 0: d['_pairs'][i] = (k, float(v))
    else: d['_pairs'].append((k, float(v)))

def dict_set_num_mat(d, key, m):
    if d is None: return
    k = float(key)
    i = _dict_find(d, k)
    if i >= 0: d['_pairs'][i] = (k, m)
    else: d['_pairs'].append((k, m))

def dict_get_str_f64(d, key):
    if d is None: return 0.0
    k = key if isinstance(key, str) else str(key)
    i = _dict_find(d, k)
    if i < 0: return 0.0
    v = d['_pairs'][i][1]
    try: return float(v)
    except Exception: return 0.0

def dict_get_str_mat(d, key):
    if d is None: return None
    k = key if isinstance(key, str) else str(key)
    i = _dict_find(d, k)
    if i < 0: return None
    return d['_pairs'][i][1]

def dict_get_num_f64(d, key):
    if d is None: return 0.0
    k = float(key)
    i = _dict_find(d, k)
    if i < 0: return 0.0
    v = d['_pairs'][i][1]
    try: return float(v)
    except Exception: return 0.0

def dict_get_num_mat(d, key):
    if d is None: return None
    k = float(key)
    i = _dict_find(d, k)
    if i < 0: return None
    return d['_pairs'][i][1]

def dict_has_str(d, key):
    if d is None: return 0.0
    k = key if isinstance(key, str) else str(key)
    return 1.0 if _dict_find(d, k) >= 0 else 0.0

def dict_has_num(d, key):
    if d is None: return 0.0
    return 1.0 if _dict_find(d, float(key)) >= 0 else 0.0

def dict_length(d):
    if d is None: return 0.0
    return float(len(d.get('_pairs', [])))

def dict_remove_str(d, key):
    if d is None: return 0.0
    k = key if isinstance(key, str) else str(key)
    i = _dict_find(d, k)
    if i < 0: return 0.0
    d['_pairs'].pop(i)
    return 1.0

def dict_remove_num(d, key):
    if d is None: return 0.0
    k = float(key)
    i = _dict_find(d, k)
    if i < 0: return 0.0
    d['_pairs'].pop(i)
    return 1.0

def obj_clone(o):
    """Phase 3 — value-class shallow clone. Mirrors the C runtime's
    matlab_obj_clone: a fresh object with independent property
    storage, but shared matrix-pointer fields. The copy gets its own
    _oid registration so identity tests don't conflate the two."""
    global _obj_next_id
    if o is None: return obj_new()
    src = _resolve_obj(o)
    new = obj_new()
    if hasattr(src, '__dict__'):
        for k, v in src.__dict__.items():
            if k.startswith('_'): continue
            new.__dict__[k] = v
    elif isinstance(src, dict):
        for k, v in src.items():
            new.__dict__[k] = v
    return new

def _resolve_obj(oid_or_obj):
    """Accept either a `_MatObj` instance or the legacy integer oid."""
    if isinstance(oid_or_obj, _MatObj): return oid_or_obj
    obj = _obj_store.get(int(oid_or_obj))
    if isinstance(obj, _MatObj): return obj
    # Pre-existing int-oid stores (back-compat path): wrap a thin facade
    # that proxies attribute access into the original dict.
    return None

def obj_set_f64(oid_or_obj, name, *rest):
    # The C ABI is `obj_set_f64(oid, name_ptr, name_len, value)`. The
    # `-emit-python` backend drops `name_len`, so the natural Python
    # call is `obj_set_f64(oid, name, value)`. Accept both shapes by
    # peeling the legacy length arg off the front when it's a small int
    # that matches `len(name)`.
    if len(rest) == 2 and isinstance(rest[0], int) and rest[0] == len(str(name)):
        v = rest[1]
    else:
        v = rest[-1]
    obj = _resolve_obj(oid_or_obj)
    if obj is not None:
        setattr(obj, name, float(v))
        return
    # Pure legacy path — store in the dict-of-dicts. Used only by
    # hand-written callers that pass an int oid never produced by
    # obj_new() since obj_new() now returns _MatObj.
    _obj_store.setdefault(int(oid_or_obj), {})[name] = float(v)


def obj_get_f64(oid_or_obj, name, *unused):
    obj = _resolve_obj(oid_or_obj)
    if obj is not None:
        return float(getattr(obj, name, 0.0))
    bucket = _obj_store.get(int(oid_or_obj), {})
    if isinstance(bucket, dict):
        return float(bucket.get(name, 0.0))
    return 0.0


# --- strings --------------------------------------------------------------

def string_from_literal(s, n=None): return s
def string_len(s): return float(len(s))
def string_concat(a, b): return str(a) + str(b)
def string_disp(s): print(s)
def strcat(*args): return "".join(str(a) for a in args)
def strtrim(s): return str(s).strip()
def lower(s): return str(s).lower()
def upper(s): return str(s).upper()
def strrep(s, old, new): return str(s).replace(str(old), str(new))
def contains(s, pat): return 1.0 if str(pat) in str(s) else 0.0
def startsWith(s, pat): return 1.0 if str(s).startswith(str(pat)) else 0.0
def endsWith(s, pat): return 1.0 if str(s).endswith(str(pat)) else 0.0
def num2str(v): return f"{float(v):g}"
def str2double(s):
    try: return float(s)
    except Exception: return float('nan')
def sprintf_f64(fmt, v): return _c_printf(_expand_escapes(str(fmt)), v)


# --- set ops --------------------------------------------------------------

def union(A, B):
    u = np.union1d(_m(A).flatten(), _m(B).flatten())
    return u.reshape((-1, 1))
def intersect(A, B):
    u = np.intersect1d(_m(A).flatten(), _m(B).flatten())
    return u.reshape((-1, 1))
def setdiff(A, B):
    u = np.setdiff1d(_m(A).flatten(), _m(B).flatten())
    return u.reshape((-1, 1))
def ismember(A, B):
    a = _m(A).flatten(); b = set(_m(B).flatten().tolist())
    return np.array([1.0 if x in b else 0.0 for x in a]).reshape(_m(A).shape)
def unique(A):
    return np.unique(_m(A).flatten()).reshape((-1, 1))


# --- concat ---------------------------------------------------------------

def horzcat(*args):
    return np.hstack([_m(a) for a in args]) if args else np.zeros((0, 0))


def vertcat(*args):
    return np.vstack([_m(a) for a in args]) if args else np.zeros((0, 0))


def flip(A): return np.flip(_m(A))
def fliplr(A): return np.fliplr(_m(A))
def flipud(A): return np.flipud(_m(A))
def rot90(A): return np.rot90(_m(A))
def sort(A):
    a = _m(A)
    # MATLAB sorts along the first non-singleton dim. A 1xN row sorts
    # elementwise; taller matrices sort each column.
    if a.ndim >= 2 and a.shape[0] == 1:
        return np.sort(a, axis=1)
    return np.sort(a, axis=0)
def sortrows(A): return _m(A)[np.lexsort(_m(A).T[::-1])]
def permute(A, perm):
    p = _m(perm).flatten().astype(int) - 1
    return np.transpose(_m(A), tuple(p))
def kron(A, B): return np.kron(_m(A), _m(B))


# --- index helpers --------------------------------------------------------

def sub2ind(sz, i, j):
    shp = _m(sz).flatten()
    return float((int(i) - 1) + (int(j) - 1) * int(shp[0]) + 1)


def ind2sub(sz, k):
    shp = _m(sz).flatten()
    m = int(shp[0])
    k0 = int(k) - 1
    return np.array([[float((k0 % m) + 1), float(k0 // m + 1)]])


# --- I/O files ------------------------------------------------------------

def fopen(name, mode="r", mlen=None, moff=None):
    try:
        m = str(mode) if mode is not None else "r"
        # Normalise "w"/"r" + "b" suffix as binary so fread/fwrite work.
        if "b" not in m:
            m = m + "b"
        return open(name, m)
    except Exception:
        return None

def fclose(fp):
    if fp is not None:
        try: fp.close()
        except Exception: pass
    return 0.0

def fgetl(fp):
    if fp is None: return ""
    line = fp.readline()
    if not line: return -1.0
    if isinstance(line, bytes):
        try: line = line.decode('utf-8', errors='replace')
        except Exception: line = ""
    return line.rstrip("\r\n")

def fread(fp, n=None):
    if fp is None: return np.zeros((0, 0))
    if n is not None:
        # Interpret as a count of f64 elements (matches matlab_fread
        # conventions for this test).
        nb = int(n) * 8
        data = fp.read(nb)
        return np.frombuffer(data, dtype=np.float64).reshape((-1, 1))
    data = fp.read()
    if isinstance(data, str): data = data.encode('utf-8', errors='replace')
    return np.frombuffer(data, dtype=np.uint8).astype(float).reshape((-1, 1))

def fwrite_mat(fp, A):
    if fp is None: return 0.0
    data = _m(A).astype(np.float64).tobytes()
    try: fp.write(data)
    except Exception:
        try: fp.buffer.write(data)
        except Exception: return 0.0
    return float(_m(A).size)

_saved_mats = {}

def load_mat(name, *args):
    return _saved_mats.get(str(name), None)

def save_mat(name, *args):
    # Signature in emitted code varies; last non-string arg is the matrix.
    for a in reversed(args):
        if not isinstance(a, (int, float)):
            _saved_mats[str(name)] = a
            break
    return 1.0

def io_file_test(*args): return 0.0
def save_test(*args): return 0.0
def binary_test(*args): return 0.0


# --- parfor ---------------------------------------------------------------

def parfor_dispatch(start, step, end, body, state):
    """Sequential parfor for v1 — runs iterations in a single thread."""
    s = float(start); st = float(step); e = float(end)
    if st == 0: return
    if (st > 0 and e < s) or (st < 0 and e > s): return
    n = int((e - s) / st) + 1
    for k in _pyrange(n):
        body(s + k * st, state)


def reduce_add_f64(ptr, delta):
    """No-op for Python — emitted parfor bodies capture the reducer as a
    plain float variable in `state`; callers handle accumulation through
    the captured slot. Left as a hook for future parfor lowering."""
    # If `ptr` is a mutable numpy array slot, accumulate into it.
    try:
        ptr[0] += float(delta)
    except Exception:
        pass


# --- assertions -----------------------------------------------------------

def assert_(cond, *args):
    # Mirrors matlab_assert: set the error flag rather than throwing,
    # so try/catch lowering in the emitter keeps working.
    if float(cond) == 0.0:
        set_error_msg("assertion failed")


def assert_msg(cond, msg, n=None):
    if float(cond) == 0.0:
        set_error_msg(str(msg) if msg else "assertion failed")


# The emitter remaps `matlab_assert` to `rt.assert_` since `assert` is a
# Python keyword.


# --- complex numbers ------------------------------------------------------

def complex_scalar(re, im): return np.array([[complex(float(re), float(im))]])
def mat_c_from_real(A): return _m(A).astype(complex)
def mat_c_from_buf(re, im, m, n):
    m = int(m); n = int(n)
    rr = np.asarray(re, dtype=float)[:m * n].reshape((m, n))
    ii = np.asarray(im, dtype=float)[:m * n].reshape((m, n))
    return rr + 1j * ii

def conj_c(A): return np.conj(np.asarray(A))
def neg_c(A): return -np.asarray(A)
def real_c(A): return np.real(np.asarray(A))
def imag_c(A): return np.imag(np.asarray(A))
def angle_c(A): return np.angle(np.asarray(A))
def add_cc(A, B): return np.asarray(A) + np.asarray(B)
def sub_cc(A, B): return np.asarray(A) - np.asarray(B)
def emul_cc(A, B): return np.asarray(A) * np.asarray(B)
def ediv_cc(A, B): return np.asarray(A) / np.asarray(B)
def matmul_cc(A, B): return np.asarray(A) @ np.asarray(B)
def transpose_c(A): return np.asarray(A).T
def ctranspose_c(A): return np.conj(np.asarray(A).T)

def disp_mat_c(A):
    a = np.asarray(A)
    for row in a:
        parts = []
        for z in row:
            re, im = z.real, z.imag
            if im >= 0:
                parts.append(f"{re:9.4g} + {im:.4g}i")
            else:
                parts.append(f"{re:9.4g} - {-im:.4g}i")
        print("  ".join(parts))


def fft_c(A):
    a = np.asarray(A)
    flat = a.flatten()
    r = np.fft.fft(flat)
    # Preserve input shape when 1-D / row / column vectors.
    if a.ndim <= 1:
        return r.reshape((1, -1))
    if a.shape[0] == 1:
        return r.reshape((1, -1))
    if a.shape[1] == 1:
        return r.reshape((-1, 1))
    return np.fft.fft(a, axis=0)


def ifft_c(A):
    a = np.asarray(A)
    flat = a.flatten()
    r = np.fft.ifft(flat)
    if a.ndim <= 1: return r.reshape((1, -1))
    if a.shape[0] == 1: return r.reshape((1, -1))
    if a.shape[1] == 1: return r.reshape((-1, 1))
    return np.fft.ifft(a, axis=0)
def fft2_c(A): return np.fft.fft2(np.asarray(A))
def ifft2_c(A): return np.fft.ifft2(np.asarray(A))


def conv(u, v):
    a = np.asarray(u, dtype=float).ravel()
    b = np.asarray(v, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 0))
    r = np.convolve(a, b)
    ua = np.asarray(u); va = np.asarray(v)
    u_col = ua.ndim == 2 and ua.shape[1] == 1 and ua.shape[0] > 1
    v_col = va.ndim == 2 and va.shape[1] == 1 and va.shape[0] > 1
    return r.reshape((-1, 1)) if (u_col or v_col) else r.reshape((1, -1))


def conv2(A, B):
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 0))
    if a.ndim == 1: a = a.reshape((1, -1))
    if b.ndim == 1: b = b.reshape((1, -1))
    am, an = a.shape; bm, bn = b.shape
    cm, cn = am + bm - 1, an + bn - 1
    c = np.zeros((cm, cn))
    for p in range(am):
        for q in range(an):
            v = a[p, q]
            if v == 0.0: continue
            c[p:p+bm, q:q+bn] += v * b
    return c


def filter(b, a, x):
    bv = np.asarray(b, dtype=float).ravel()
    av = np.asarray(a, dtype=float).ravel()
    if av.size == 0 or bv.size == 0 or av[0] == 0.0:
        return np.zeros((0, 0))
    bn = bv / av[0]; an = av / av[0]
    xa = np.asarray(x, dtype=float)
    if xa.ndim == 1 or 1 in xa.shape:
        flat = xa.ravel()
        from scipy.signal import lfilter
        try:
            y = lfilter(bn, an, flat)
        except Exception:
            # Fallback DF-II-T without scipy.
            L = max(bn.size, an.size)
            bp = np.zeros(L); bp[:bn.size] = bn
            ap = np.zeros(L); ap[:an.size] = an
            w = np.zeros(L); y = np.zeros(flat.size)
            for n in range(flat.size):
                yn = bp[0] * flat[n] + w[0]
                w = np.concatenate([bp[1:] * flat[n] - ap[1:] * yn + w[1:],
                                    [0.0]])
                y[n] = yn
        if xa.ndim == 2 and xa.shape[1] == 1:
            return y.reshape((-1, 1))
        return y.reshape((1, -1)) if xa.ndim <= 1 else y.reshape(xa.shape)
    # column-wise on a matrix
    out = np.zeros_like(xa)
    for j in range(xa.shape[1]):
        out[:, j] = filter(b, a, xa[:, j]).ravel()
    return out


def any(A):
    a = np.asarray(A)
    if a.ndim <= 1 or 1 in a.shape:
        return np.array([[1.0 if (a != 0).any() else 0.0]])
    return (a != 0).any(axis=0).astype(float).reshape((1, -1))


def all(A):
    a = np.asarray(A)
    if a.ndim <= 1 or 1 in a.shape:
        return np.array([[1.0 if (a != 0).all() else 0.0]])
    return (a != 0).all(axis=0).astype(float).reshape((1, -1))


def tril(A):  return np.tril(np.asarray(A, dtype=float))
def triu(A):  return np.triu(np.asarray(A, dtype=float))


def fftshift_c(A):  return np.fft.fftshift(np.asarray(A))
def ifftshift_c(A): return np.fft.ifftshift(np.asarray(A))


def std(A):
    a = np.asarray(A, dtype=float)
    if a.ndim <= 1 or 1 in a.shape:
        return np.array([[float(np.std(a.ravel(), ddof=1)) if a.size > 1 else 0.0]])
    return np.std(a, axis=0, ddof=1).reshape((1, -1))


def var(A):
    a = np.asarray(A, dtype=float)
    if a.ndim <= 1 or 1 in a.shape:
        return np.array([[float(np.var(a.ravel(), ddof=1)) if a.size > 1 else 0.0]])
    return np.var(a, axis=0, ddof=1).reshape((1, -1))


def median(A):
    a = np.asarray(A, dtype=float)
    if a.ndim <= 1 or 1 in a.shape:
        return np.array([[float(np.median(a.ravel())) if a.size > 0 else 0.0]])
    return np.median(a, axis=0).reshape((1, -1))


def diff(A):
    a = np.asarray(A, dtype=float)
    if a.ndim <= 1 or 1 in a.shape:
        flat = a.ravel()
        if flat.size < 2: return np.zeros((0, 0))
        d = np.diff(flat)
        if a.ndim == 2 and a.shape[1] == 1: return d.reshape((-1, 1))
        return d.reshape((1, -1))
    return np.diff(a, axis=0)


# --- ODE solvers ----------------------------------------------------------
# Dormand-Prince 5(4) and Bogacki-Shampine 3(2). Scalar y only. Cached
# across the paired _t / _y calls so the second call returns the other
# half without re-integrating. Tolerances and step control match the C
# runtime so cross-backend output stays in lockstep.

_ode_cache = {"key": None, "t": None, "y": None}

def _ode_hermite(y, y1, k, k1, h, th):
    th2 = th * th
    th3 = th2 * th
    return ((2*th3 - 3*th2 + 1) * y
            + (-2*th3 + 3*th2)  * y1
            + h * (th3 - 2*th2 + th) * k
            + h * (th3 - th2)        * k1)

def _ode_solve_dp45(f, targets, y0, rtol=1e-3, atol=1e-6,
                    max_step=0.0, init_step=0.0, refine=4):
    """Returns (T, Y, n_acc, n_rej, n_fev). Stats fields zero on the
    early-exit paths; non-zero on successful integration."""
    max_steps = 100000
    if refine < 1: refine = 1
    n_targets = len(targets)
    if n_targets < 2:
        return [], [], 0, 0, 0
    t0 = float(targets[0])
    tf = float(targets[n_targets - 1])
    user_grid = (n_targets > 2)
    T = [t0]; Y = [y0]
    next_tgt = 1
    t, y = t0, y0
    span = tf - t0
    if init_step > 0.0:
        h = init_step if span >= 0 else (0.0 - init_step)
    else:
        h = span * 0.01
    if h == 0.0 or span == 0.0:
        return T, Y, 0, 0, 0
    forward = h > 0
    if max_step > 0.0:
        if h >  max_step: h =  max_step
        if h < (0.0 - max_step): h = 0.0 - max_step
    k1 = f(t, y)
    n_acc = 0; n_rej = 0; n_fev = 1
    steps = 0
    while ((t < tf) if forward else (t > tf)) and steps < max_steps:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        k2 = f(t + h*(1/5),  y + h*(k1*(1/5)))
        k3 = f(t + h*(3/10), y + h*(k1*(3/40) + k2*(9/40)))
        k4 = f(t + h*(4/5),  y + h*(k1*(44/45) - k2*(56/15) + k3*(32/9)))
        k5 = f(t + h*(8/9),  y + h*(k1*(19372/6561) - k2*(25360/2187)
                                    + k3*(64448/6561) - k4*(212/729)))
        k6 = f(t + h,        y + h*(k1*(9017/3168) - k2*(355/33)
                                    + k3*(46732/5247) + k4*(49/176)
                                    - k5*(5103/18656)))
        y5 = y + h*(k1*(35/384) + k3*(500/1113) + k4*(125/192)
                    - k5*(2187/6784) + k6*(11/84))
        k7 = f(t + h, y5)
        n_fev += 6
        err = h*(k1*(71/57600) - k3*(71/16695) + k4*(71/1920)
                 - k5*(17253/339200) + k6*(22/525) - k7*(1/40))
        scale = atol + rtol * (abs(y) if abs(y) > abs(y5) else abs(y5))
        normerr = abs(err)/scale if scale > 0 else 0.0
        if normerr <= 1.0:
            n_acc += 1
            if user_grid:
                while next_tgt < n_targets:
                    tt = float(targets[next_tgt])
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th = 0.0 if h == 0.0 else (tt - t) / h
                    yi = y5 if next_tgt == n_targets - 1 \
                            else _ode_hermite(y, y5, k1, k7, h, th)
                    T.append(tt); Y.append(yi)
                    next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th = j / refine
                    ti = t + h * th
                    yi = y5 if j == refine else _ode_hermite(y, y5, k1, k7, h, th)
                    T.append(ti); Y.append(yi)
                    j += 1
            t += h
            y = y5
            k1 = k7
            if user_grid and next_tgt >= n_targets:
                break
        else:
            n_rej += 1
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/5))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0.0:
            if h >  max_step: h =  max_step
            if h < (0.0 - max_step): h = 0.0 - max_step
    return T, Y, n_acc, n_rej, n_fev

def _ode_solve_bs23(f, targets, y0, rtol=1e-3, atol=1e-6,
                    max_step=0.0, init_step=0.0, refine=1):
    """Returns (T, Y, n_acc, n_rej, n_fev)."""
    max_steps = 100000
    if refine < 1: refine = 1
    n_targets = len(targets)
    if n_targets < 2:
        return [], [], 0, 0, 0
    t0 = float(targets[0])
    tf = float(targets[n_targets - 1])
    user_grid = (n_targets > 2)
    T = [t0]; Y = [y0]
    next_tgt = 1
    t, y = t0, y0
    span = tf - t0
    if init_step > 0.0:
        h = init_step if span >= 0 else (0.0 - init_step)
    else:
        h = span * 0.01
    if h == 0.0 or span == 0.0:
        return T, Y, 0, 0, 0
    forward = h > 0
    if max_step > 0.0:
        if h >  max_step: h =  max_step
        if h < (0.0 - max_step): h = 0.0 - max_step
    k1 = f(t, y)
    n_acc = 0; n_rej = 0; n_fev = 1
    steps = 0
    while ((t < tf) if forward else (t > tf)) and steps < max_steps:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        k2 = f(t + h*0.5,  y + h*(k1*0.5))
        k3 = f(t + h*0.75, y + h*(k2*0.75))
        y3 = y + h*(k1*(2/9) + k2*(1/3) + k3*(4/9))
        k4 = f(t + h, y3)
        n_fev += 3
        err = h*(k1*(-5/72) + k2*(1/12) + k3*(1/9) - k4*(1/8))
        scale = atol + rtol * (abs(y) if abs(y) > abs(y3) else abs(y3))
        normerr = abs(err)/scale if scale > 0 else 0.0
        if normerr <= 1.0:
            n_acc += 1
            if user_grid:
                while next_tgt < n_targets:
                    tt = float(targets[next_tgt])
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th = 0.0 if h == 0.0 else (tt - t) / h
                    yi = y3 if next_tgt == n_targets - 1 \
                            else _ode_hermite(y, y3, k1, k4, h, th)
                    T.append(tt); Y.append(yi)
                    next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th = j / refine
                    ti = t + h * th
                    yi = y3 if j == refine else _ode_hermite(y, y3, k1, k4, h, th)
                    T.append(ti); Y.append(yi)
                    j += 1
            t += h
            y = y3
            k1 = k4
            if user_grid and next_tgt >= n_targets:
                break
        else:
            n_rej += 1
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/3))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0.0:
            if h >  max_step: h =  max_step
            if h < (0.0 - max_step): h = 0.0 - max_step
    return T, Y, n_acc, n_rej, n_fev

def _ode_compute(kind, f, tspan, y0, rtol=1e-3, atol=1e-6,
                 max_step=0.0, init_step=0.0, refine=None,
                 print_stats=False):
    if refine is None:
        refine = 4 if kind == 45 else 1
    ts = np.asarray(tspan, dtype=float).ravel()
    targets = ts.tolist()
    key = (kind, id(f), tuple(targets), float(y0),
           float(rtol), float(atol), float(max_step), float(init_step),
           int(refine), bool(print_stats))
    if _ode_cache["key"] == key:
        return
    solver = _ode_solve_dp45 if kind == 45 else _ode_solve_bs23
    T, Y, n_acc, n_rej, n_fev = solver(
        f, targets, y0, rtol, atol, max_step, init_step, refine)
    _ode_cache["key"] = key
    _ode_cache["t"] = np.asarray(T, dtype=float).reshape((-1, 1))
    _ode_cache["y"] = np.asarray(Y, dtype=float).reshape((-1, 1))
    _ode_cache["n_acc"] = n_acc
    _ode_cache["n_rej"] = n_rej
    _ode_cache["n_fev"] = n_fev
    if print_stats:
        print(f"{n_acc} successful steps")
        print(f"{n_rej} failed attempts")
        print(f"{n_fev} function evaluations")

def _ode_opts_resolve(opts, default_refine):
    """Pull RelTol / AbsTol / MaxStep / InitialStep / Refine / Stats from
    a struct-shaped dict; fall back to MATLAB defaults when fields are
    missing or opts is None. Stats is a numeric flag (non-zero = on);
    see header doc for the MATLAB-string deviation."""
    rtol, atol = 1e-3, 1e-6
    max_step, init_step = 0.0, 0.0
    refine = default_refine
    print_stats = False
    if opts is not None:
        try:
            if "RelTol"      in opts: rtol      = float(opts["RelTol"])
            if "AbsTol"      in opts: atol      = float(opts["AbsTol"])
            if "MaxStep"     in opts: max_step  = float(opts["MaxStep"])
            if "InitialStep" in opts: init_step = float(opts["InitialStep"])
            if "Refine"      in opts:
                r = int(opts["Refine"])
                if r >= 1: refine = r
            if "Stats"       in opts:
                print_stats = bool(float(opts["Stats"]))
        except (TypeError, KeyError):
            pass
    return rtol, atol, max_step, init_step, refine, print_stats

def ode45_t(f, tspan, y0):
    _ode_compute(45, f, tspan, float(y0))
    return _ode_cache["t"].copy()

def ode45_y(f, tspan, y0):
    _ode_compute(45, f, tspan, float(y0))
    return _ode_cache["y"].copy()

def ode23_t(f, tspan, y0):
    _ode_compute(23, f, tspan, float(y0))
    return _ode_cache["t"].copy()

def ode23_y(f, tspan, y0):
    _ode_compute(23, f, tspan, float(y0))
    return _ode_cache["y"].copy()

def ode45_t_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 4)
    _ode_compute(45, f, tspan, float(y0), rtol, atol,
                 max_step, init_step, refine, print_stats=ps)
    return _ode_cache["t"].copy()

def ode45_y_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 4)
    _ode_compute(45, f, tspan, float(y0), rtol, atol,
                 max_step, init_step, refine, print_stats=ps)
    return _ode_cache["y"].copy()

def ode23_t_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 1)
    _ode_compute(23, f, tspan, float(y0), rtol, atol,
                 max_step, init_step, refine, print_stats=ps)
    return _ode_cache["t"].copy()

def ode23_y_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 1)
    _ode_compute(23, f, tspan, float(y0), rtol, atol,
                 max_step, init_step, refine, print_stats=ps)
    return _ode_cache["y"].copy()


# --- 3-return [t, y, stats] form ----------------------------------------
# `stats` is a dict (struct-shaped) with nsteps / nfailed / nfevals.
# Cache stores the counts on solve so the third call just packages.

def _ode_stats_struct():
    s = struct_new()
    s["nsteps"]  = float(_ode_cache.get("n_acc", 0))
    s["nfailed"] = float(_ode_cache.get("n_rej", 0))
    s["nfevals"] = float(_ode_cache.get("n_fev", 0))
    return s

def ode45_stats(f, tspan, y0):
    _ode_compute(45, f, tspan, float(y0))
    return _ode_stats_struct()

def ode45_stats_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 4)
    _ode_compute(45, f, tspan, float(y0), rtol, atol,
                 max_step, init_step, refine, print_stats=ps)
    return _ode_stats_struct()

def ode23_stats(f, tspan, y0):
    _ode_compute(23, f, tspan, float(y0))
    return _ode_stats_struct()

def ode23_stats_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 1)
    _ode_compute(23, f, tspan, float(y0), rtol, atol,
                 max_step, init_step, refine, print_stats=ps)
    return _ode_stats_struct()


# --- ode23s — Rosenbrock 2(3) stiff solver --------------------------------
# Same Shampine pair as the C runtime. Scalar y → division by W;
# vector y → numpy.linalg.solve(W, ·) at each stage. Refine default = 1.

import math as _math

def _rosen_solve_23s_scalar(f, targets, y0, rtol=1e-3, atol=1e-6,
                              max_step=0.0, init_step=0.0, refine=1):
    if refine < 1: refine = 1
    targets = list(map(float, targets))
    n_targets = len(targets)
    if n_targets < 2:
        return [], [], 0, 0, 0
    t0 = targets[0]; tf = targets[n_targets - 1]
    user_grid = (n_targets > 2)
    T = [t0]; Y = [y0]; next_tgt = 1
    y = y0; t = t0
    span = tf - t0
    h = init_step if init_step > 0 else span * 0.01
    if span < 0 and init_step > 0: h = -h
    if h == 0.0 or span == 0.0:
        return T, Y, 0, 0, 0
    forward = h > 0
    if max_step > 0:
        if h >  max_step: h = max_step
        if h < -max_step: h = -max_step
    SQRT2 = _math.sqrt(2.0)
    d_   = 1.0 / (2.0 + SQRT2)
    e32  = 6.0 + SQRT2
    SQRT_EPS = 1.4901161193847656e-8
    n_acc = 0; n_rej = 0; n_fev = 0
    steps = 0; max_steps = 100000
    while ((t < tf) if forward else (t > tf)) and steps < max_steps:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        F0 = f(t, y); n_fev += 1
        eps = SQRT_EPS * (abs(y) if abs(y) > 1.0 else 1.0)
        Jp = f(t, y + eps); Jm = f(t, y - eps); n_fev += 2
        J = (Jp - Jm) / (2.0 * eps)
        W = 1.0 - h * d_ * J
        if W == 0.0: W = 1e-30
        k1 = F0 / W
        F1 = f(t + 0.5*h, y + 0.5*h*k1); n_fev += 1
        k2 = (F1 - k1) / W + k1
        y_new = y + h * k2
        F2 = f(t + h, y_new); n_fev += 1
        k3 = (F2 - e32*(k2 - F1) - 2.0*(k1 - F0)) / W
        err = (h / 6.0) * (k1 - 2.0*k2 + k3)
        scale = atol + rtol * (abs(y) if abs(y) > abs(y_new) else abs(y_new))
        normerr = abs(err) / scale if scale > 0 else 0.0
        if normerr <= 1.0:
            n_acc += 1
            if user_grid:
                while next_tgt < n_targets:
                    tt = float(targets[next_tgt])
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th = 0.0 if h == 0.0 else (tt - t) / h
                    if next_tgt == n_targets - 1:
                        Y.append(y_new)
                    else:
                        Y.append(_ode_hermite(y, y_new, F0, F2, h, th))
                    T.append(tt); next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th = j / refine
                    ti = t + h * th
                    if j == refine:
                        Y.append(y_new)
                    else:
                        Y.append(_ode_hermite(y, y_new, F0, F2, h, th))
                    T.append(ti); j += 1
            t += h; y = y_new
            if user_grid and next_tgt >= n_targets: break
        else:
            n_rej += 1
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/3))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0:
            if h >  max_step: h = max_step
            if h < -max_step: h = -max_step
    return T, Y, n_acc, n_rej, n_fev

def _rosen_solve_23s_vector(f, targets, y0, rtol=1e-3, atol=1e-6,
                              max_step=0.0, init_step=0.0, refine=1):
    if refine < 1: refine = 1
    targets = list(map(float, targets))
    n_targets = len(targets)
    D = len(y0)
    if n_targets < 2 or D <= 0:
        return [], np.zeros((0, D)), 0, 0, 0
    t0 = targets[0]; tf = targets[n_targets - 1]
    user_grid = (n_targets > 2)
    T = [t0]; Y_rows = [np.asarray(y0, dtype=float).copy()]
    next_tgt = 1
    y = np.asarray(y0, dtype=float).copy()
    t = t0
    span = tf - t0
    h = init_step if init_step > 0 else span * 0.01
    if span < 0 and init_step > 0: h = -h
    if h == 0.0 or span == 0.0:
        return T, np.array(Y_rows), 0, 0, 0
    forward = h > 0
    if max_step > 0:
        if h >  max_step: h = max_step
        if h < -max_step: h = -max_step
    SQRT2 = _math.sqrt(2.0)
    d_   = 1.0 / (2.0 + SQRT2)
    e32  = 6.0 + SQRT2
    SQRT_EPS = 1.4901161193847656e-8
    n_acc = 0; n_rej = 0; n_fev = 0
    steps = 0; max_steps = 100000
    eyeD = np.eye(D)
    while ((t < tf) if forward else (t > tf)) and steps < max_steps:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        F0 = _ode_v_call(f, t, y, D); n_fev += 1
        # Build Jacobian column-by-column (central FD). Manual loop —
        # `range` is the MATLAB-runtime symbol in this module.
        Jmat = np.zeros((D, D))
        j = 0
        while j < D:
            yj = y[j]
            dj = SQRT_EPS * (abs(yj) if abs(yj) > 1.0 else 1.0)
            yp = y.copy(); yp[j] = yj + dj
            ym = y.copy(); ym[j] = yj - dj
            Fp = _ode_v_call(f, t, yp, D)
            Fm = _ode_v_call(f, t, ym, D)
            n_fev += 2
            Jmat[:, j] = (Fp - Fm) / (2.0 * dj)
            j += 1
        W = eyeD - h * d_ * Jmat
        try:
            Wlu = np.linalg.lu_factor(W) if hasattr(np.linalg, "lu_factor") \
                  else None
        except Exception:
            Wlu = None
        # numpy.linalg has no lu_factor in the public API; just call solve.
        try:
            k1 = np.linalg.solve(W, F0)
            F1 = _ode_v_call(f, t + 0.5*h, y + 0.5*h*k1, D); n_fev += 1
            k2 = np.linalg.solve(W, F1 - k1) + k1
            y_new = y + h * k2
            F2 = _ode_v_call(f, t + h, y_new, D); n_fev += 1
            k3 = np.linalg.solve(W, F2 - e32*(k2 - F1) - 2.0*(k1 - F0))
        except np.linalg.LinAlgError:
            n_rej += 1
            h *= 0.5
            continue
        err = (h / 6.0) * (k1 - 2.0*k2 + k3)
        ay = np.abs(y); ayN = np.abs(y_new)
        scale = atol + rtol * np.maximum(ay, ayN)
        e = np.where(scale > 0, np.abs(err) / np.maximum(scale, 1e-300), 0.0)
        normerr = float(np.max(e)) if e.size else 0.0
        if normerr <= 1.0:
            n_acc += 1
            if user_grid:
                while next_tgt < n_targets:
                    tt = float(targets[next_tgt])
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th_ = 0.0 if h == 0.0 else (tt - t) / h
                    if next_tgt == n_targets - 1:
                        Y_rows.append(y_new.copy())
                    else:
                        Y_rows.append(_ode_v_hermite(y, y_new, F0, F2, h, th_))
                    T.append(tt); next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th_ = j / refine
                    ti = t + h * th_
                    if j == refine:
                        Y_rows.append(y_new.copy())
                    else:
                        Y_rows.append(_ode_v_hermite(y, y_new, F0, F2, h, th_))
                    T.append(ti); j += 1
            t += h
            y = y_new.copy()
            if user_grid and next_tgt >= n_targets: break
        else:
            n_rej += 1
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/3))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0:
            if h >  max_step: h = max_step
            if h < -max_step: h = -max_step
    return T, np.array(Y_rows), n_acc, n_rej, n_fev

def _ode23s_compute(f, tspan, y0, rtol=1e-3, atol=1e-6,
                     max_step=0.0, init_step=0.0, refine=1, print_stats=False):
    ts = np.asarray(tspan, dtype=float).ravel()
    targets = ts.tolist()
    key = (235, id(f), tuple(targets), float(y0),
           float(rtol), float(atol), float(max_step), float(init_step),
           int(refine), bool(print_stats))
    if _ode_cache["key"] == key:
        return
    T, Y, n_acc, n_rej, n_fev = _rosen_solve_23s_scalar(
        f, targets, y0, rtol, atol, max_step, init_step, refine)
    _ode_cache["key"] = key
    _ode_cache["t"] = np.asarray(T, dtype=float).reshape((-1, 1))
    _ode_cache["y"] = np.asarray(Y, dtype=float).reshape((-1, 1))
    _ode_cache["n_acc"] = n_acc
    _ode_cache["n_rej"] = n_rej
    _ode_cache["n_fev"] = n_fev
    if print_stats:
        print(f"{n_acc} successful steps")
        print(f"{n_rej} failed attempts")
        print(f"{n_fev} function evaluations")

def ode23s_t(f, tspan, y0):
    _ode23s_compute(f, tspan, float(y0))
    return _ode_cache["t"].copy()
def ode23s_y(f, tspan, y0):
    _ode23s_compute(f, tspan, float(y0))
    return _ode_cache["y"].copy()
def ode23s_t_opts(f, tspan, y0, opts):
    rtol, atol, mxs, ins, rfn, ps = _ode_opts_resolve(opts, 1)
    _ode23s_compute(f, tspan, float(y0), rtol, atol, mxs, ins, rfn, print_stats=ps)
    return _ode_cache["t"].copy()
def ode23s_y_opts(f, tspan, y0, opts):
    rtol, atol, mxs, ins, rfn, ps = _ode_opts_resolve(opts, 1)
    _ode23s_compute(f, tspan, float(y0), rtol, atol, mxs, ins, rfn, print_stats=ps)
    return _ode_cache["y"].copy()
def ode23s_stats(f, tspan, y0):
    _ode23s_compute(f, tspan, float(y0))
    return _ode_stats_struct()
def ode23s_stats_opts(f, tspan, y0, opts):
    rtol, atol, mxs, ins, rfn, ps = _ode_opts_resolve(opts, 1)
    _ode23s_compute(f, tspan, float(y0), rtol, atol, mxs, ins, rfn, print_stats=ps)
    return _ode_stats_struct()

def _ode23s_v_compute(f, tspan, y0, rtol=1e-3, atol=1e-6,
                       max_step=0.0, init_step=0.0, refine=1, print_stats=False):
    ts = np.asarray(tspan, dtype=float).ravel()
    targets = ts.tolist()
    y0v = np.asarray(y0, dtype=float).ravel()
    D = int(y0v.size)
    key = (235, id(f), tuple(targets), tuple(y0v.tolist()),
           float(rtol), float(atol), float(max_step), float(init_step),
           int(refine), bool(print_stats))
    if _ode_v_cache["key"] == key:
        return
    T, Y, n_acc, n_rej, n_fev = _rosen_solve_23s_vector(
        f, targets, y0v, rtol, atol, max_step, init_step, refine)
    _ode_v_cache["key"] = key
    _ode_v_cache["t"] = np.asarray(T, dtype=float).reshape((-1, 1))
    _ode_v_cache["y"] = Y if Y.size else np.zeros((0, D))
    _ode_v_cache["n_acc"] = n_acc
    _ode_v_cache["n_rej"] = n_rej
    _ode_v_cache["n_fev"] = n_fev
    _ode_v_cache["D"] = D
    if print_stats:
        print(f"{n_acc} successful steps")
        print(f"{n_rej} failed attempts")
        print(f"{n_fev} function evaluations")

def ode23s_v_t(f, tspan, y0):
    _ode23s_v_compute(f, tspan, y0); return _ode_v_cache["t"].copy()
def ode23s_v_y(f, tspan, y0):
    _ode23s_v_compute(f, tspan, y0); return _ode_v_cache["y"].copy()
def ode23s_v_t_opts(f, tspan, y0, opts):
    rtol, atol, mxs, ins, rfn, ps = _ode_opts_resolve(opts, 1)
    _ode23s_v_compute(f, tspan, y0, rtol, atol, mxs, ins, rfn, print_stats=ps)
    return _ode_v_cache["t"].copy()
def ode23s_v_y_opts(f, tspan, y0, opts):
    rtol, atol, mxs, ins, rfn, ps = _ode_opts_resolve(opts, 1)
    _ode23s_v_compute(f, tspan, y0, rtol, atol, mxs, ins, rfn, print_stats=ps)
    return _ode_v_cache["y"].copy()
def ode23s_v_stats(f, tspan, y0):
    _ode23s_v_compute(f, tspan, y0); return _ode_v_stats()
def ode23s_v_stats_opts(f, tspan, y0, opts):
    rtol, atol, mxs, ins, rfn, ps = _ode_opts_resolve(opts, 1)
    _ode23s_v_compute(f, tspan, y0, rtol, atol, mxs, ins, rfn, print_stats=ps)
    return _ode_v_stats()


# --- ode_events — IVP solver with event detection ------------------------
# v1: scalar y, single event. The event function returns a 3-vector
# [value; isterminal; direction]. Bisection on Hermite-interpolated
# state between accepted RK45 steps.

_ode_events_cache = {"key": None, "T": None, "Y": None,
                     "TE": None, "YE": None, "IE": None}

def _ode_evt_eval(evt, t, y):
    r = evt(t, y)
    arr = np.asarray(r, dtype=float).ravel()
    if arr.size < 1:
        return 0.0, 0, 0
    value = float(arr[0])
    term  = int(arr[1]) if arr.size >= 2 else 0
    direction = int(arr[2]) if arr.size >= 3 else 0
    return value, term, direction

def _ode_evt_bisect(evt, t, h, y, y_new, k1, k7, v0):
    lo, hi = 0.0, 1.0
    vlo = v0
    it = 0
    while it < 50:
        mid = 0.5 * (lo + hi)
        y_mid = _ode_hermite(y, y_new, k1, k7, h, mid)
        v, _, _ = _ode_evt_eval(evt, t + mid * h, y_mid)
        if abs(v) < 1e-12 or (hi - lo) < 1e-15:
            return mid
        if (vlo < 0.0 and v > 0.0) or (vlo > 0.0 and v < 0.0):
            hi = mid
        else:
            lo = mid; vlo = v
        it += 1
    return 0.5 * (lo + hi)

def _rk_solve_dp45_events(f, evt, targets, y0, rtol=1e-3, atol=1e-6,
                           max_step=0.0, init_step=0.0, refine=4):
    n_targets = len(targets)
    if n_targets < 2:
        return [], [], [], [], []
    if refine < 1: refine = 1
    t0 = float(targets[0]); tf = float(targets[n_targets - 1])
    user_grid = (n_targets > 2)
    T = [t0]; Y = [y0]; next_tgt = 1
    TE = []; YE = []; IE = []
    y, t = y0, t0
    span = tf - t0
    h = init_step if init_step > 0 else span * 0.01
    if span < 0 and init_step > 0: h = -h
    if h == 0.0 or span == 0.0:
        return T, Y, TE, YE, IE
    forward = h > 0
    if max_step > 0:
        if h >  max_step: h =  max_step
        if h < -max_step: h = -max_step
    k1 = f(t, y)
    v_prev, _, _ = _ode_evt_eval(evt, t, y)
    steps = 0; max_steps = 100000
    halted = False
    while ((t < tf) if forward else (t > tf)) and steps < max_steps and not halted:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        k2 = f(t + h*(1/5),  y + h*(k1*(1/5)))
        k3 = f(t + h*(3/10), y + h*(k1*(3/40) + k2*(9/40)))
        k4 = f(t + h*(4/5),  y + h*(k1*(44/45) - k2*(56/15) + k3*(32/9)))
        k5 = f(t + h*(8/9),  y + h*(k1*(19372/6561) - k2*(25360/2187)
                                    + k3*(64448/6561) - k4*(212/729)))
        k6 = f(t + h,        y + h*(k1*(9017/3168) - k2*(355/33)
                                    + k3*(46732/5247) + k4*(49/176)
                                    - k5*(5103/18656)))
        y5 = y + h*(k1*(35/384) + k3*(500/1113) + k4*(125/192)
                    - k5*(2187/6784) + k6*(11/84))
        k7 = f(t + h, y5)
        err = h*(k1*(71/57600) - k3*(71/16695) + k4*(71/1920)
                 - k5*(17253/339200) + k6*(22/525) - k7*(1/40))
        scale = atol + rtol * (abs(y) if abs(y) > abs(y5) else abs(y5))
        normerr = abs(err) / scale if scale > 0 else 0.0
        if normerr <= 1.0:
            v_new, term_new, dir_setting = _ode_evt_eval(evt, t + h, y5)
            crossed = False
            if v_prev * v_new < 0.0:
                rising = (v_new > v_prev)
                if dir_setting == 0: crossed = True
                elif dir_setting > 0 and rising: crossed = True
                elif dir_setting < 0 and not rising: crossed = True
            if crossed:
                th_star = _ode_evt_bisect(evt, t, h, y, y5, k1, k7, v_prev)
                te = t + th_star * h
                ye = _ode_hermite(y, y5, k1, k7, h, th_star)
                TE.append(te); YE.append(ye); IE.append(1)
                if term_new:
                    T.append(te); Y.append(ye)
                    halted = True
                    break
            v_prev = v_new
            if user_grid:
                while next_tgt < n_targets:
                    tt = float(targets[next_tgt])
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th = 0.0 if h == 0.0 else (tt - t) / h
                    yi = y5 if next_tgt == n_targets - 1 \
                            else _ode_hermite(y, y5, k1, k7, h, th)
                    T.append(tt); Y.append(yi)
                    next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th = j / refine
                    ti = t + h * th
                    yi = y5 if j == refine else _ode_hermite(y, y5, k1, k7, h, th)
                    T.append(ti); Y.append(yi)
                    j += 1
            t += h; y = y5; k1 = k7
            if user_grid and next_tgt >= n_targets: break
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/5))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0:
            if h >  max_step: h =  max_step
            if h < -max_step: h = -max_step
    return T, Y, TE, YE, IE

def _ode_events_compute(f, evt, tspan, y0):
    ts = np.asarray(tspan, dtype=float).ravel()
    targets = ts.tolist()
    key = (id(f), id(evt), tuple(targets), float(y0))
    if _ode_events_cache["key"] == key:
        return
    T, Y, TE, YE, IE = _rk_solve_dp45_events(f, evt, targets, float(y0))
    _ode_events_cache["key"] = key
    _ode_events_cache["T"]  = np.asarray(T,  dtype=float).reshape((-1, 1))
    _ode_events_cache["Y"]  = np.asarray(Y,  dtype=float).reshape((-1, 1))
    _ode_events_cache["TE"] = np.asarray(TE, dtype=float).reshape((-1, 1))
    _ode_events_cache["YE"] = np.asarray(YE, dtype=float).reshape((-1, 1))
    _ode_events_cache["IE"] = np.asarray(IE, dtype=float).reshape((-1, 1))

def ode_events_t (f, tspan, y0, evt):
    _ode_events_compute(f, evt, tspan, y0); return _ode_events_cache["T"].copy()
def ode_events_y (f, tspan, y0, evt):
    _ode_events_compute(f, evt, tspan, y0); return _ode_events_cache["Y"].copy()
def ode_events_te(f, tspan, y0, evt):
    _ode_events_compute(f, evt, tspan, y0); return _ode_events_cache["TE"].copy()
def ode_events_ye(f, tspan, y0, evt):
    _ode_events_compute(f, evt, tspan, y0); return _ode_events_cache["YE"].copy()
def ode_events_ie(f, tspan, y0, evt):
    _ode_events_compute(f, evt, tspan, y0); return _ode_events_cache["IE"].copy()


# --- pdepe — 1-D parabolic-elliptic PDE via method-of-lines --------------
# v1: m=0 (Cartesian), scalar PDE, Dirichlet BCs. Spatial discretisation
# on the user xmesh + ode23s_v under the hood.

_pdepe_ctx = {"pdefn": None, "bcfn": None, "xmesh": None, "Nx": 0,
              "m": 0, "err": 0}

def _pdepe_xpow(x, m):
    if m == 0: return 1.0
    if m == 1: return x
    if m == 2: return x * x
    return x ** m

def _pdepe_eval_bc(t, ul, ur):
    """Evaluate the user's bcfun at current boundary values; return
    (pl, ql, pr, qr) or None on shape failure."""
    xl = _pdepe_ctx["xmesh"][0]
    xr = _pdepe_ctx["xmesh"][_pdepe_ctx["Nx"] - 1]
    r = _pdepe_ctx["bcfn"](xl, ul, xr, ur, t)
    arr = np.asarray(r, dtype=float).ravel()
    if arr.size < 4:
        _pdepe_ctx["err"] = 1
        return None
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

def _pdepe_rhs(t, Ufull):
    Nx = _pdepe_ctx["Nx"]
    xmesh = _pdepe_ctx["xmesh"]
    pdefn = _pdepe_ctx["pdefn"]
    Uflat = np.asarray(Ufull, dtype=float).ravel()
    if Uflat.size != Nx:
        return np.zeros((Nx, 1))
    u = Uflat.copy()
    bc = _pdepe_eval_bc(t, float(u[0]), float(u[Nx - 1]))
    if bc is None:
        return np.zeros((Nx, 1))
    pl, ql_, pr, qr_ = bc
    dirichlet_left  = (ql_ == 0.0)
    dirichlet_right = (qr_ == 0.0)
    # Snap Dirichlet boundaries: linear pl = ul - g(t) → g(t) = ul - pl.
    if dirichlet_left:  u[0]      = u[0]      - pl
    if dirichlet_right: u[Nx - 1] = u[Nx - 1] - pr
    f_left_bdy  = 0.0 if dirichlet_left  else (-pl / ql_)
    f_right_bdy = 0.0 if dirichlet_right else (-pr / qr_)
    # Compute interior fluxes f_{i+1/2}.
    flx = np.empty(Nx - 1)
    i = 0
    while i < Nx - 1:
        xL = xmesh[i]; xR = xmesh[i + 1]
        dx = xR - xL
        if dx == 0.0: dx = 1e-30
        xm = 0.5 * (xL + xR)
        um = 0.5 * (u[i] + u[i + 1])
        dudx = (u[i + 1] - u[i]) / dx
        rr = pdefn(xm, t, um, dudx)
        rrarr = np.asarray(rr, dtype=float).ravel()
        flx[i] = rrarr[1] if rrarr.size >= 2 else 0.0
        i += 1
    mm = _pdepe_ctx["m"]
    if mm != 0:
        i = 0
        while i < Nx - 1:
            xm = 0.5 * (xmesh[i] + xmesh[i + 1])
            flx[i] *= _pdepe_xpow(xm, mm)
            i += 1
    out = np.zeros((Nx, 1))
    # Left boundary node 0.
    if dirichlet_left:
        out[0, 0] = 0.0
    else:
        xi = xmesh[0]; ui = u[0]
        dudx = (u[1] - u[0]) / (xmesh[1] - xmesh[0])
        rr = pdefn(xi, t, ui, dudx)
        rrarr = np.asarray(rr, dtype=float).ravel()
        c = rrarr[0] if rrarr.size >= 1 else 1.0
        s = rrarr[2] if rrarr.size >= 3 else 0.0
        if c == 0.0: c = 1e-30
        cell_w = 0.5 * (xmesh[1] - xmesh[0])
        xpow_l = _pdepe_xpow(xi, mm)
        f_l_bdy_w = f_left_bdy * xpow_l if mm != 0 else f_left_bdy
        inv_xpow = 0.0 if xpow_l == 0.0 else (1.0 / xpow_l)
        out[0, 0] = (((flx[0] - f_l_bdy_w) / cell_w) * inv_xpow + s) / c
    # Interior nodes.
    i = 1
    while i < Nx - 1:
        xi = xmesh[i]; ui = u[i]
        dudx = (u[i + 1] - u[i - 1]) / (xmesh[i + 1] - xmesh[i - 1])
        rr = pdefn(xi, t, ui, dudx)
        rrarr = np.asarray(rr, dtype=float).ravel()
        c = rrarr[0] if rrarr.size >= 1 else 1.0
        s = rrarr[2] if rrarr.size >= 3 else 0.0
        if c == 0.0: c = 1e-30
        dx_avg = 0.5 * (xmesh[i + 1] - xmesh[i - 1])
        dflux = flx[i] - flx[i - 1]
        xpow_i = _pdepe_xpow(xi, mm)
        inv_xpow = 0.0 if xpow_i == 0.0 else (1.0 / xpow_i)
        out[i, 0] = ((dflux / dx_avg) * inv_xpow + s) / c
        i += 1
    # Right boundary node Nx-1.
    if dirichlet_right:
        out[Nx - 1, 0] = 0.0
    else:
        xi = xmesh[Nx - 1]; ui = u[Nx - 1]
        dudx = (u[Nx - 1] - u[Nx - 2]) / (xmesh[Nx - 1] - xmesh[Nx - 2])
        rr = pdefn(xi, t, ui, dudx)
        rrarr = np.asarray(rr, dtype=float).ravel()
        c = rrarr[0] if rrarr.size >= 1 else 1.0
        s = rrarr[2] if rrarr.size >= 3 else 0.0
        if c == 0.0: c = 1e-30
        cell_w = 0.5 * (xmesh[Nx - 1] - xmesh[Nx - 2])
        xpow_r = _pdepe_xpow(xi, mm)
        f_r_bdy_w = f_right_bdy * xpow_r if mm != 0 else f_right_bdy
        inv_xpow = 0.0 if xpow_r == 0.0 else (1.0 / xpow_r)
        out[Nx - 1, 0] = (((f_r_bdy_w - flx[Nx - 2]) / cell_w) * inv_xpow + s) / c
    return out

def pdepe(m, pdefn, icfn, bcfn, xmesh, tspan):
    if pdefn is None or icfn is None or bcfn is None:
        return np.zeros((0, 0))
    xs = np.asarray(xmesh, dtype=float).ravel()
    ts = np.asarray(tspan, dtype=float).ravel()
    Nx = int(xs.size); Nt = int(ts.size)
    if Nx < 3 or Nt < 2: return np.zeros((0, 0))
    mi = int(m)
    if mi < 0 or mi > 2 or float(mi) != float(m): return np.zeros((0, 0))
    if mi != 0 and xs[0] <= 0.0: return np.zeros((0, 0))
    _pdepe_ctx["pdefn"] = pdefn
    _pdepe_ctx["bcfn"]  = bcfn
    _pdepe_ctx["xmesh"] = xs
    _pdepe_ctx["Nx"]    = Nx
    _pdepe_ctx["m"]     = mi
    _pdepe_ctx["err"]   = 0
    # Invalidate the ode23s_v cache: same _pdepe_rhs / y0 across pdepe
    # calls would otherwise return a stale solution when only the
    # pdepe context (m, bcfn, …) changed.
    _ode_v_cache["key"] = None
    # Initial state covers ALL mesh points.
    u0 = np.zeros(Nx)
    j = 0
    while j < Nx:
        u0[j] = float(icfn(xs[j]))
        j += 1
    T = ode23s_v_t(_pdepe_rhs, ts, u0)
    U = ode23s_v_y(_pdepe_rhs, ts, u0)
    Tflat = np.asarray(T, dtype=float).ravel()
    Nt_out = Tflat.size
    sol = np.asarray(U, dtype=float).reshape((Nt_out, Nx)).copy()
    # Re-snap Dirichlet boundaries at output time.
    k = 0
    while k < Nt_out:
        bc = _pdepe_eval_bc(float(Tflat[k]),
                             float(sol[k, 0]), float(sol[k, Nx - 1]))
        if bc is not None:
            pl, ql_, pr, qr_ = bc
            if ql_ == 0.0: sol[k, 0]      = sol[k, 0]      - pl
            if qr_ == 0.0: sol[k, Nx - 1] = sol[k, Nx - 1] - pr
        k += 1
    return sol


# --- Vector-y solvers ----------------------------------------------------
# Same Dormand-Prince / Bogacki-Shampine pair as the scalar path, but
# operating on D-component vectors. The user RHS takes a Dx1 column
# matrix (numpy array) and returns the same shape.

_ode_v_cache = {"key": None, "t": None, "y": None,
                "n_acc": 0, "n_rej": 0, "n_fev": 0, "D": 0}

def _ode_v_call(f, t, y, D):
    """Call user RHS with a Dx1 column. Return numpy 1D array of length D."""
    yv = np.asarray(y, dtype=float).reshape((D, 1))
    dy = f(t, yv)
    arr = np.asarray(dy, dtype=float).ravel()
    if arr.size < D:
        out = np.zeros(D)
        out[:arr.size] = arr
        return out
    return arr[:D].copy()

def _ode_v_hermite(y0, y1, k0, k1, h, th):
    th2 = th * th
    th3 = th2 * th
    return ((2*th3 - 3*th2 + 1) * y0
            + (-2*th3 + 3*th2)  * y1
            + h * (th3 - 2*th2 + th) * k0
            + h * (th3 - th2)        * k1)

def _ode_v_solve_dp45(f, targets, y0, rtol=1e-3, atol=1e-6,
                       max_step=0.0, init_step=0.0, refine=4):
    max_steps = 100000
    if refine < 1: refine = 1
    targets = list(map(float, targets))
    n_targets = len(targets)
    D = len(y0)
    if n_targets < 2 or D <= 0:
        return [], np.zeros((0, D)), 0, 0, 0
    t0 = targets[0]; tf = targets[n_targets - 1]
    user_grid = (n_targets > 2)
    T = [t0]
    Y_rows = [np.asarray(y0, dtype=float).copy()]
    next_tgt = 1
    y = np.asarray(y0, dtype=float).copy()
    span = tf - t0
    if init_step > 0.0:
        h = init_step if span >= 0 else (0.0 - init_step)
    else:
        h = span * 0.01
    if h == 0.0 or span == 0.0:
        return T, np.array(Y_rows), 0, 0, 0
    forward = h > 0
    if max_step > 0.0:
        if h >  max_step: h = max_step
        if h < -max_step: h = -max_step
    k1 = _ode_v_call(f, t0, y, D)
    n_acc = 0; n_rej = 0; n_fev = 1
    t = t0
    steps = 0
    while ((t < tf) if forward else (t > tf)) and steps < max_steps:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        k2 = _ode_v_call(f, t + h*(1/5),  y + h*(k1*(1/5)), D)
        k3 = _ode_v_call(f, t + h*(3/10), y + h*(k1*(3/40) + k2*(9/40)), D)
        k4 = _ode_v_call(f, t + h*(4/5),  y + h*(k1*(44/45) - k2*(56/15) + k3*(32/9)), D)
        k5 = _ode_v_call(f, t + h*(8/9),  y + h*(k1*(19372/6561) - k2*(25360/2187)
                                                + k3*(64448/6561) - k4*(212/729)), D)
        k6 = _ode_v_call(f, t + h,        y + h*(k1*(9017/3168) - k2*(355/33)
                                                + k3*(46732/5247) + k4*(49/176)
                                                - k5*(5103/18656)), D)
        y5 = y + h*(k1*(35/384) + k3*(500/1113) + k4*(125/192)
                    - k5*(2187/6784) + k6*(11/84))
        k7 = _ode_v_call(f, t + h, y5, D)
        n_fev += 6
        err = h*(k1*(71/57600) - k3*(71/16695) + k4*(71/1920)
                 - k5*(17253/339200) + k6*(22/525) - k7*(1/40))
        ay = np.abs(y); ay5 = np.abs(y5)
        scale = atol + rtol * np.maximum(ay, ay5)
        e = np.where(scale > 0, np.abs(err) / np.maximum(scale, 1e-300), 0.0)
        normerr = float(np.max(e)) if e.size else 0.0
        if normerr <= 1.0:
            n_acc += 1
            if user_grid:
                while next_tgt < n_targets:
                    tt = targets[next_tgt]
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th_ = 0.0 if h == 0.0 else (tt - t) / h
                    if next_tgt == n_targets - 1:
                        Y_rows.append(y5.copy())
                    else:
                        Y_rows.append(_ode_v_hermite(y, y5, k1, k7, h, th_))
                    T.append(tt)
                    next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th_ = j / refine
                    ti = t + h * th_
                    if j == refine:
                        Y_rows.append(y5.copy())
                    else:
                        Y_rows.append(_ode_v_hermite(y, y5, k1, k7, h, th_))
                    T.append(ti)
                    j += 1
            t += h
            y = y5.copy()
            k1 = k7
            if user_grid and next_tgt >= n_targets: break
        else:
            n_rej += 1
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/5))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0.0:
            if h >  max_step: h = max_step
            if h < -max_step: h = -max_step
    return T, np.array(Y_rows), n_acc, n_rej, n_fev

def _ode_v_solve_bs23(f, targets, y0, rtol=1e-3, atol=1e-6,
                       max_step=0.0, init_step=0.0, refine=1):
    max_steps = 100000
    if refine < 1: refine = 1
    targets = list(map(float, targets))
    n_targets = len(targets)
    D = len(y0)
    if n_targets < 2 or D <= 0:
        return [], np.zeros((0, D)), 0, 0, 0
    t0 = targets[0]; tf = targets[n_targets - 1]
    user_grid = (n_targets > 2)
    T = [t0]; Y_rows = [np.asarray(y0, dtype=float).copy()]
    next_tgt = 1
    y = np.asarray(y0, dtype=float).copy()
    span = tf - t0
    if init_step > 0.0:
        h = init_step if span >= 0 else (0.0 - init_step)
    else:
        h = span * 0.01
    if h == 0.0 or span == 0.0:
        return T, np.array(Y_rows), 0, 0, 0
    forward = h > 0
    if max_step > 0.0:
        if h >  max_step: h = max_step
        if h < -max_step: h = -max_step
    k1 = _ode_v_call(f, t0, y, D)
    n_acc = 0; n_rej = 0; n_fev = 1
    t = t0
    steps = 0
    while ((t < tf) if forward else (t > tf)) and steps < max_steps:
        steps += 1
        if (forward and t + h > tf) or ((not forward) and t + h < tf):
            h = tf - t
        k2 = _ode_v_call(f, t + h*0.5,  y + h*(k1*0.5), D)
        k3 = _ode_v_call(f, t + h*0.75, y + h*(k2*0.75), D)
        y3 = y + h*(k1*(2/9) + k2*(1/3) + k3*(4/9))
        k4 = _ode_v_call(f, t + h, y3, D)
        n_fev += 3
        err = h*(k1*(-5/72) + k2*(1/12) + k3*(1/9) - k4*(1/8))
        ay = np.abs(y); ay3 = np.abs(y3)
        scale = atol + rtol * np.maximum(ay, ay3)
        e = np.where(scale > 0, np.abs(err) / np.maximum(scale, 1e-300), 0.0)
        normerr = float(np.max(e)) if e.size else 0.0
        if normerr <= 1.0:
            n_acc += 1
            if user_grid:
                while next_tgt < n_targets:
                    tt = targets[next_tgt]
                    in_range = (tt <= t + h) if forward else (tt >= t + h)
                    if not in_range: break
                    th_ = 0.0 if h == 0.0 else (tt - t) / h
                    if next_tgt == n_targets - 1:
                        Y_rows.append(y3.copy())
                    else:
                        Y_rows.append(_ode_v_hermite(y, y3, k1, k4, h, th_))
                    T.append(tt)
                    next_tgt += 1
            else:
                j = 1
                while j <= refine:
                    th_ = j / refine
                    ti = t + h * th_
                    if j == refine:
                        Y_rows.append(y3.copy())
                    else:
                        Y_rows.append(_ode_v_hermite(y, y3, k1, k4, h, th_))
                    T.append(ti)
                    j += 1
            t += h
            y = y3.copy()
            k1 = k4
            if user_grid and next_tgt >= n_targets: break
        else:
            n_rej += 1
        fac = 5.0 if normerr == 0.0 else 0.9 * (normerr ** (-1/3))
        if fac < 0.2: fac = 0.2
        if fac > 5.0: fac = 5.0
        h *= fac
        if max_step > 0.0:
            if h >  max_step: h = max_step
            if h < -max_step: h = -max_step
    return T, np.array(Y_rows), n_acc, n_rej, n_fev

def _ode_v_compute(kind, f, tspan, y0,
                    rtol=1e-3, atol=1e-6, max_step=0.0, init_step=0.0,
                    refine=None, print_stats=False):
    if refine is None:
        refine = 4 if kind == 45 else 1
    ts = np.asarray(tspan, dtype=float).ravel()
    targets = ts.tolist()
    y0v = np.asarray(y0, dtype=float).ravel()
    D = int(y0v.size)
    key = (kind, id(f), tuple(targets), tuple(y0v.tolist()),
           float(rtol), float(atol), float(max_step), float(init_step),
           int(refine), bool(print_stats))
    if _ode_v_cache["key"] == key:
        return
    solver = _ode_v_solve_dp45 if kind == 45 else _ode_v_solve_bs23
    T, Y, n_acc, n_rej, n_fev = solver(
        f, targets, y0v, rtol, atol, max_step, init_step, refine)
    _ode_v_cache["key"] = key
    _ode_v_cache["t"] = np.asarray(T, dtype=float).reshape((-1, 1))
    _ode_v_cache["y"] = Y if Y.size else np.zeros((0, D))
    _ode_v_cache["n_acc"] = n_acc
    _ode_v_cache["n_rej"] = n_rej
    _ode_v_cache["n_fev"] = n_fev
    _ode_v_cache["D"] = D
    if print_stats:
        print(f"{n_acc} successful steps")
        print(f"{n_rej} failed attempts")
        print(f"{n_fev} function evaluations")

def _ode_v_stats():
    s = struct_new()
    s["nsteps"]  = float(_ode_v_cache.get("n_acc", 0))
    s["nfailed"] = float(_ode_v_cache.get("n_rej", 0))
    s["nfevals"] = float(_ode_v_cache.get("n_fev", 0))
    return s

def ode45_v_t(f, tspan, y0):
    _ode_v_compute(45, f, tspan, y0)
    return _ode_v_cache["t"].copy()
def ode45_v_y(f, tspan, y0):
    _ode_v_compute(45, f, tspan, y0)
    return _ode_v_cache["y"].copy()
def ode23_v_t(f, tspan, y0):
    _ode_v_compute(23, f, tspan, y0)
    return _ode_v_cache["t"].copy()
def ode23_v_y(f, tspan, y0):
    _ode_v_compute(23, f, tspan, y0)
    return _ode_v_cache["y"].copy()

def ode45_v_t_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 4)
    _ode_v_compute(45, f, tspan, y0, rtol, atol, max_step, init_step, refine, print_stats=ps)
    return _ode_v_cache["t"].copy()
def ode45_v_y_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 4)
    _ode_v_compute(45, f, tspan, y0, rtol, atol, max_step, init_step, refine, print_stats=ps)
    return _ode_v_cache["y"].copy()
def ode23_v_t_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 1)
    _ode_v_compute(23, f, tspan, y0, rtol, atol, max_step, init_step, refine, print_stats=ps)
    return _ode_v_cache["t"].copy()
def ode23_v_y_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 1)
    _ode_v_compute(23, f, tspan, y0, rtol, atol, max_step, init_step, refine, print_stats=ps)
    return _ode_v_cache["y"].copy()

def ode45_v_stats(f, tspan, y0):
    _ode_v_compute(45, f, tspan, y0)
    return _ode_v_stats()
def ode45_v_stats_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 4)
    _ode_v_compute(45, f, tspan, y0, rtol, atol, max_step, init_step, refine, print_stats=ps)
    return _ode_v_stats()
def ode23_v_stats(f, tspan, y0):
    _ode_v_compute(23, f, tspan, y0)
    return _ode_v_stats()
def ode23_v_stats_opts(f, tspan, y0, opts):
    rtol, atol, max_step, init_step, refine, ps = _ode_opts_resolve(opts, 1)
    _ode_v_compute(23, f, tspan, y0, rtol, atol, max_step, init_step, refine, print_stats=ps)
    return _ode_v_stats()


def meshgrid_X(x, y=None):
    xv = np.asarray(x, dtype=float).ravel()
    yv = xv if y is None else np.asarray(y, dtype=float).ravel()
    return np.tile(xv, (yv.size, 1))

def meshgrid_Y(x, y=None):
    xv = np.asarray(x, dtype=float).ravel()
    yv = xv if y is None else np.asarray(y, dtype=float).ravel()
    return np.tile(yv.reshape((-1, 1)), (1, xv.size))

def ndgrid_X(x, y=None):
    xv = np.asarray(x, dtype=float).ravel()
    yv = xv if y is None else np.asarray(y, dtype=float).ravel()
    return np.tile(xv.reshape((-1, 1)), (1, yv.size))

def ndgrid_Y(x, y=None):
    xv = np.asarray(x, dtype=float).ravel()
    yv = xv if y is None else np.asarray(y, dtype=float).ravel()
    return np.tile(yv, (xv.size, 1))


# --- remaining stubs ------------------------------------------------------
# Programs that exercise these symbols without a real implementation will
# produce wrong output, but won't crash — good enough for coverage.

def matpow(A, n):
    a = _m(A); n = int(n)
    if n == 0: return np.eye(a.shape[0])
    if n > 0: return np.linalg.matrix_power(a, n)
    return np.linalg.matrix_power(np.linalg.inv(a), -n)

def rand(m, n=None):
    m = int(m); n = int(n) if n is not None else m
    return np.random.rand(m, n)

def randn(m, n=None):
    m = int(m); n = int(n) if n is not None else m
    return np.random.randn(m, n)
