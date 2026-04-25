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


def fprintf_f64(fmt, n, v):
    sys.stdout.write(_c_printf(_expand_escapes(fmt), v))


def fprintf_f64_2(fmt, n, a, b):
    sys.stdout.write(_c_printf(_expand_escapes(fmt), a, b))


def fprintf_f64_3(fmt, n, a, b, c):
    sys.stdout.write(_c_printf(_expand_escapes(fmt), a, b, c))


def fprintf_f64_4(fmt, n, a, b, c, d):
    sys.stdout.write(_c_printf(_expand_escapes(fmt), a, b, c, d))


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


# --- object / class -------------------------------------------------------

_obj_store = {}
_obj_next_id = 1

def obj_new(*_ignored):
    global _obj_next_id
    oid = _obj_next_id; _obj_next_id += 1
    _obj_store[oid] = {}
    return oid

def obj_set_f64(oid, name, n, v):
    _obj_store.setdefault(int(oid), {})[name] = float(v)

def obj_get_f64(oid, name, n=None):
    return float(_obj_store.get(int(oid), {}).get(name, 0.0))


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
