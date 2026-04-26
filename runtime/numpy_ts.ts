// numpy-ts — a minimal NumPy-like API for the TypeScript emission lane
// of matlab_llvm. The shape mirrors `numpy` closely enough that the
// generated source from `matlabc -emit-typescript` reads as the natural
// translation of the MATLAB original:
//
//     A = [1 2; 3 4];           ->   const A = np.array([[1, 2], [3, 4]]);
//     B = A * 2;                ->   const B = A.mul(2);
//     C = A * B;                ->   const C = np.matmul(A, B);
//     d = A.';                  ->   const d = A.T;
//     e = inv(A);               ->   const e = np.linalg.inv(A);
//
// We back arrays with a flat Float64Array so the cost model is close to
// numpy's contiguous buffers. Ops produce fresh arrays (no in-place
// mutation), so the emitted source can treat them as MATLAB values.
//
// One file on purpose: simpler to ship and easy to drop into a Bun /
// tsx project alongside the matlab_runtime.ts shim.

export class NDArray {
  // Row-major flat storage. `shape.length` is always >= 1; scalars
  // are represented as `shape = [1, 1]` so the 2-D matrix algebra
  // reads consistently.
  readonly data: Float64Array;
  readonly shape: number[];

  constructor(data: Float64Array | number[], shape: number[]) {
    this.data = data instanceof Float64Array
      ? data
      : Float64Array.from(data);
    this.shape = shape;
  }

  // --- shape helpers ------------------------------------------------------

  get ndim(): number { return this.shape.length; }
  get size(): number { return this.data.length; }
  get rows(): number { return this.shape[0] ?? 1; }
  get cols(): number { return this.shape[1] ?? 1; }

  // Transpose. Returns a fresh contiguous matrix — no view trickery,
  // since the rest of the runtime assumes row-major layout.
  get T(): NDArray {
    if (this.ndim < 2) return new NDArray(this.data.slice(), [1, this.size]);
    const [m, n] = this.shape;
    const out = new Float64Array(m * n);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        out[j * m + i] = this.data[i * n + j];
    return new NDArray(out, [n, m]);
  }

  // --- element access -----------------------------------------------------

  // `at(i, j)` mirrors numpy's `arr[i, j]` (0-indexed).
  at(i: number, j: number = 0): number {
    if (this.ndim < 2) return this.data[i];
    return this.data[i * this.cols + j];
  }

  set(i: number, j: number, v: number): void {
    if (this.ndim < 2) { this.data[i] = v; return; }
    this.data[i * this.cols + j] = v;
  }

  // Convert to a plain (nested) JS array — handy when callers want to
  // round-trip through JSON or pass to a non-numpy consumer.
  tolist(): any {
    if (this.ndim <= 1) return Array.from(this.data);
    if (this.ndim === 2) {
      const out: number[][] = [];
      for (let i = 0; i < this.rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.cols; j++) row.push(this.data[i * this.cols + j]);
        out.push(row);
      }
      return out;
    }
    return Array.from(this.data);
  }

  reshape(shape: number[]): NDArray {
    const n = shape.reduce((a, b) => a * b, 1);
    if (n !== this.data.length)
      throw new Error(`reshape: size ${this.data.length} != ${n}`);
    return new NDArray(this.data.slice(), shape);
  }

  // --- elementwise binary ops --------------------------------------------
  //
  // Each accepts either an NDArray (same shape — broadcasting is
  // limited) or a scalar (number). The factor-out helper keeps the
  // four ops as single-line method bodies, which is what readers expect.

  add(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => a + b); }
  sub(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => a - b); }
  mul(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => a * b); }
  div(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => a / b); }
  pow(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => a ** b); }

  neg(): NDArray {
    const out = new Float64Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) out[i] = -this.data[i];
    return new NDArray(out, this.shape.slice());
  }

  // --- comparisons --------------------------------------------------------
  // Return 0/1 ndarrays, matching MATLAB's logical-as-double convention.

  eq(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => (a === b ? 1 : 0)); }
  ne(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => (a !== b ? 1 : 0)); }
  lt(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => (a < b ? 1 : 0)); }
  le(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => (a <= b ? 1 : 0)); }
  gt(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => (a > b ? 1 : 0)); }
  ge(other: NDArray | number): NDArray { return ewBin(this, other, (a, b) => (a >= b ? 1 : 0)); }
}

// Elementwise binary helper. Broadcasts a scalar against any shape and
// pairs same-shape NDArrays elementwise. We deliberately keep the
// fast-path inline (no try/catch) — callers that pass mismatched shapes
// get a clear `Error` immediately instead of a NaN result.
function ewBin(a: NDArray, b: NDArray | number,
               op: (x: number, y: number) => number): NDArray {
  if (typeof b === "number") {
    const out = new Float64Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = op(a.data[i], b);
    return new NDArray(out, a.shape.slice());
  }
  if (a.data.length !== b.data.length) {
    throw new Error(`shape mismatch: ${a.shape} vs ${b.shape}`);
  }
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = op(a.data[i], b.data[i]);
  return new NDArray(out, a.shape.slice());
}

// Coerce a value into an NDArray. Mirrors `_m(x)` in matlab_runtime.py.
export function asArray(x: any): NDArray {
  if (x instanceof NDArray) return x;
  if (typeof x === "number" || typeof x === "boolean") {
    return new NDArray(new Float64Array([Number(x)]), [1, 1]);
  }
  if (Array.isArray(x)) return arrayFromNested(x);
  throw new Error(`asArray: cannot coerce ${typeof x}`);
}

// Build an NDArray from a nested number[][] / number[]. Detects the
// 2-D case and stores row-major; 1-D inputs become a 1xN row vector to
// match MATLAB's vector-as-row convention.
function arrayFromNested(values: any[]): NDArray {
  if (values.length > 0 && Array.isArray(values[0])) {
    const m = values.length;
    const n = (values[0] as any[]).length;
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      const row = values[i] as any[];
      if (row.length !== n) throw new Error("ragged matrix literal");
      for (let j = 0; j < n; j++) data[i * n + j] = Number(row[j]);
    }
    return new NDArray(data, [m, n]);
  }
  const n = values.length;
  return new NDArray(Float64Array.from(values.map(Number)), [1, n]);
}

// --- top-level constructors -------------------------------------------------

export function array(values: any): NDArray { return asArray(values); }

export function zeros(m: number, n?: number): NDArray {
  m = m | 0; n = n === undefined ? m : (n | 0);
  return new NDArray(new Float64Array(m * n), [m, n]);
}

export function ones(m: number, n?: number): NDArray {
  m = m | 0; n = n === undefined ? m : (n | 0);
  const buf = new Float64Array(m * n);
  buf.fill(1);
  return new NDArray(buf, [m, n]);
}

export function eye(m: number, n?: number): NDArray {
  m = m | 0; n = n === undefined ? m : (n | 0);
  const buf = new Float64Array(m * n);
  const k = Math.min(m, n);
  for (let i = 0; i < k; i++) buf[i * n + i] = 1;
  return new NDArray(buf, [m, n]);
}

export function arange(start: number, stop?: number, step: number = 1): NDArray {
  if (stop === undefined) { stop = start; start = 0; }
  const out: number[] = [];
  if (step > 0) for (let v = start; v < stop; v += step) out.push(v);
  else for (let v = start; v > stop; v += step) out.push(v);
  return new NDArray(Float64Array.from(out), [1, out.length]);
}

export function linspace(a: number, b: number, n: number = 100): NDArray {
  n = n | 0;
  if (n <= 0) return zeros(1, 0);
  if (n === 1) return new NDArray(new Float64Array([a]), [1, 1]);
  const out = new Float64Array(n);
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  return new NDArray(out, [1, n]);
}

// --- linear algebra --------------------------------------------------------

export function matmul(A: NDArray | number[][] | number,
                       B: NDArray | number[][] | number): NDArray {
  const a = asArray(A); const b = asArray(B);
  const am = a.rows, ak = a.cols, bk = b.rows, bn = b.cols;
  if (ak !== bk) throw new Error(`matmul: ${a.shape} vs ${b.shape}`);
  const out = new Float64Array(am * bn);
  for (let i = 0; i < am; i++) {
    for (let j = 0; j < bn; j++) {
      let s = 0;
      for (let k = 0; k < ak; k++) s += a.data[i * ak + k] * b.data[k * bn + j];
      out[i * bn + j] = s;
    }
  }
  return new NDArray(out, [am, bn]);
}

export function transpose(A: NDArray | number[][] | number): NDArray {
  return asArray(A).T;
}

export function trace(A: NDArray | number[][] | number): number {
  const a = asArray(A);
  const k = Math.min(a.rows, a.cols);
  let s = 0;
  for (let i = 0; i < k; i++) s += a.data[i * a.cols + i];
  return s;
}

export const linalg = {
  // Compute determinant via LU with partial pivoting. Tracks parity of
  // row swaps in `sign` so the determinant comes out signed correctly.
  det(A: NDArray | number[][] | number): number {
    const a = asArray(A);
    const n = a.rows;
    if (n !== a.cols) throw new Error("det: not square");
    const m = new Float64Array(a.data);
    let sign = 1;
    for (let k = 0; k < n; k++) {
      let pivot = k;
      for (let i = k + 1; i < n; i++)
        if (Math.abs(m[i * n + k]) > Math.abs(m[pivot * n + k])) pivot = i;
      if (m[pivot * n + k] === 0) return 0;
      if (pivot !== k) {
        for (let j = 0; j < n; j++) {
          const t = m[k * n + j]; m[k * n + j] = m[pivot * n + j]; m[pivot * n + j] = t;
        }
        sign = -sign;
      }
      for (let i = k + 1; i < n; i++) {
        const f = m[i * n + k] / m[k * n + k];
        for (let j = k; j < n; j++) m[i * n + j] -= f * m[k * n + j];
      }
    }
    let d = sign;
    for (let i = 0; i < n; i++) d *= m[i * n + i];
    return d;
  },

  // Inverse via Gauss–Jordan on `[A | I]`. Square matrices only.
  inv(A: NDArray | number[][] | number): NDArray {
    const a = asArray(A);
    const n = a.rows;
    if (n !== a.cols) throw new Error("inv: not square");
    const m = new Float64Array(2 * n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) m[i * 2 * n + j] = a.data[i * n + j];
      m[i * 2 * n + (n + i)] = 1;
    }
    for (let k = 0; k < n; k++) {
      let pivot = k;
      for (let i = k + 1; i < n; i++)
        if (Math.abs(m[i * 2 * n + k]) > Math.abs(m[pivot * 2 * n + k])) pivot = i;
      if (m[pivot * 2 * n + k] === 0) throw new Error("inv: singular");
      if (pivot !== k) {
        for (let j = 0; j < 2 * n; j++) {
          const t = m[k * 2 * n + j]; m[k * 2 * n + j] = m[pivot * 2 * n + j]; m[pivot * 2 * n + j] = t;
        }
      }
      const piv = m[k * 2 * n + k];
      for (let j = 0; j < 2 * n; j++) m[k * 2 * n + j] /= piv;
      for (let i = 0; i < n; i++) {
        if (i === k) continue;
        const f = m[i * 2 * n + k];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) m[i * 2 * n + j] -= f * m[k * 2 * n + j];
      }
    }
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) out[i * n + j] = m[i * 2 * n + (n + j)];
    return new NDArray(out, [n, n]);
  },

  // Solve `A x = b` by Gauss elimination with partial pivoting. `b` may
  // be a column vector or a multi-column right-hand side.
  solve(A: NDArray | number[][] | number, B: NDArray | number[][] | number): NDArray {
    const a = asArray(A); const b = asArray(B);
    const n = a.rows;
    const k = b.cols;
    if (n !== a.cols) throw new Error("solve: A not square");
    if (n !== b.rows) throw new Error("solve: A/b shape mismatch");
    const m = new Float64Array(n * (n + k));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) m[i * (n + k) + j] = a.data[i * n + j];
      for (let j = 0; j < k; j++) m[i * (n + k) + (n + j)] = b.data[i * k + j];
    }
    for (let p = 0; p < n; p++) {
      let pivot = p;
      for (let i = p + 1; i < n; i++)
        if (Math.abs(m[i * (n + k) + p]) > Math.abs(m[pivot * (n + k) + p])) pivot = i;
      if (m[pivot * (n + k) + p] === 0) throw new Error("solve: singular");
      if (pivot !== p) {
        for (let j = 0; j < n + k; j++) {
          const t = m[p * (n + k) + j]; m[p * (n + k) + j] = m[pivot * (n + k) + j]; m[pivot * (n + k) + j] = t;
        }
      }
      for (let i = p + 1; i < n; i++) {
        const f = m[i * (n + k) + p] / m[p * (n + k) + p];
        for (let j = p; j < n + k; j++) m[i * (n + k) + j] -= f * m[p * (n + k) + j];
      }
    }
    const x = new Float64Array(n * k);
    for (let i = n - 1; i >= 0; i--) {
      for (let j = 0; j < k; j++) {
        let s = m[i * (n + k) + (n + j)];
        for (let q = i + 1; q < n; q++) s -= m[i * (n + k) + q] * x[q * k + j];
        x[i * k + j] = s / m[i * (n + k) + i];
      }
    }
    return new NDArray(x, [n, k]);
  },

  norm(A: NDArray | number[][] | number): number {
    const a = asArray(A);
    let s = 0;
    for (let i = 0; i < a.data.length; i++) s += a.data[i] * a.data[i];
    return Math.sqrt(s);
  },
};

// --- elementwise unary math (numpy-style) ---------------------------------
//
// Each accepts an NDArray (returns an NDArray) or a scalar (returns a
// scalar). The dual-shape keeps call-sites compact in the emitted
// source: `np.sin(x)` works whether `x` is a matrix or a number.

function unary(op: (x: number) => number) {
  return (a: NDArray | number): NDArray | number => {
    if (typeof a === "number") return op(a);
    const out = new Float64Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = op(a.data[i]);
    return new NDArray(out, a.shape.slice());
  };
}

export const sqrt   = unary(Math.sqrt);
export const exp    = unary(Math.exp);
export const log    = unary(Math.log);
export const log2_  = unary(Math.log2);
export const log10_ = unary(Math.log10);
export const sin    = unary(Math.sin);
export const cos    = unary(Math.cos);
export const tan    = unary(Math.tan);
export const asin   = unary(Math.asin);
export const acos   = unary(Math.acos);
export const atan   = unary(Math.atan);
export const sinh   = unary(Math.sinh);
export const cosh   = unary(Math.cosh);
export const tanh   = unary(Math.tanh);
export const abs    = unary(Math.abs);
export const sign   = unary(Math.sign);
export const floor  = unary(Math.floor);
export const ceil   = unary(Math.ceil);
export const round  = unary((x) => Math.round(x));

// Re-export under the names emit-typescript wants (`np.log2`, `np.log10`)
// — `log2`/`log10` are reserved-ish but still legal as JS property names,
// so we re-bind them on the namespace export below.
export { log2_ as log2, log10_ as log10 };
