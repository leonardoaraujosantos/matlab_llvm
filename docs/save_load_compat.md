# `save` / `load` — MATLAB API + `.mat` Format Compatibility Plan

This document scopes bringing `save` and `load` from the current
single-matrix custom format (`MLB1`) up to the real MATLAB API and
on-disk format, in two phases.

Today, after `runtime/matlab_runtime.c:5321`:

- `save(path, A)` takes a *value*, writes one matrix per file.
- `B = load(path)` returns the matrix directly.
- The on-disk layout is `"MLB1" + int64 rows + int64 cols + doubles`.
- It is **not** readable by MathWorks MATLAB.

The end goal is that `.mat` files written here open in MATLAB and vice
versa, and that user code like

```matlab
A = [1 2 3; 4 5 6];
b = pi;
save('out.mat', 'A', 'b');

S = load('out.mat');     % S.A, S.b
T = load('out.mat', 'A'); % T.A only
```

works the same way it does in MATLAB.

## 0. Constraints (what we won't do)

These behaviors of real MATLAB don't fit a compiled language with no
runtime workspace and are explicitly out of scope:

- **Workspace injection.** `load('f.mat')` with no LHS in real MATLAB
  drops variables straight into the caller's workspace via `evalin`.
  We have SSA values, not a name-addressable workspace. The struct
  return form (`S = load(...)`) covers the same use case.
- **Whole-workspace save.** `save('f.mat')` (no var list) saves
  every workspace variable. There is no enumerable workspace at
  runtime; users must list variables explicitly.
- **Command syntax** (`save filename A B`, `load filename`). Possible,
  but separate parser work; track in a follow-up.
- **MAT-File v7.3** (HDF5). Different format entirely; needs an HDF5
  dependency. Out of scope — we target v5 uncompressed in Phase 2 and
  v7 (zlib-wrapped v5) as a stretch.

## 1. Phase 1 — API alignment (no `.mat` interop yet)

Goal: source-compatible API with real MATLAB. File format stays our
own, but is extended to a multi-record format (`MLB2`) so multiple
named variables fit in one file.

### 1.1 Frontend / Sema changes

`save` and `load` need special-case treatment because their string
arguments are *names* of variables in the current scope, not regular
string values.

- **Recognize the call sites.** In `Resolver::resolveCallee` (or a
  small post-pass after resolution), when the callee is the builtin
  `save` or `load` and arguments include string literals, mark the
  call as a `BuiltinSpecial` form. Add a tag on `CallOrIndex` (e.g.
  `SpecialKind { None, SaveByName, LoadByName }`).
- **For `save`:** for every string-literal arg after the path, look
  up the name in the current scope. If found, replace the literal
  with a synthetic `(name_string, name_ref)` pair the lowerer can
  consume. If not found, emit `undefined name 'X' in save(...)`.
  The first argument (path) stays a normal expression.
- **For `load`:** if the call has a single LHS in an `AssignStmt`,
  type the result as a `Struct` (existing `Type::Kind::Struct`,
  unknown fields). String-literal arguments after the path are
  field-name filters and need no scope lookup — they're just stored
  on the call node.
- **Diagnostics.** Reject non-string-literal arguments after the
  path with `save/load: expected string literal name`. Real MATLAB
  errors similarly when you pass a non-char.

Files touched: `lib/Sema/Resolver.cpp`, `include/matlab/AST/AST.h`
(new `SpecialKind` enum on `CallOrIndex`), and
`lib/Sema/TypeInference.cpp` for the struct-result typing of `load`.

### 1.2 IR / Lowering changes

In `lib/MLIR/Passes/LowerTensorOps.cpp` around line 1528, replace
the current single-pair lowering with variadic-shaped runtime calls:

- `save(path, 'A', 'B', ...)` → emit `matlab_save_named_begin(path)`,
  then per name `matlab_save_named_add(handle, name_str, name_len,
  value_ptr, kind)`, then `matlab_save_named_end(handle)`. `kind`
  is a small enum (matrix / scalar f64 / string / struct).
- `S = load(path, 'A', 'B', ...)` → emit
  `matlab_load_named(path, n_filters, filter_names...)` returning a
  `matlab_struct *`. Filter list of length 0 means "all variables".

The variadic-on-stack approach (build a small array of structs in IR
and pass `count + ptr`) keeps the runtime ABI stable. Match the
pattern already used by `matlab_struct_*` for argument plumbing.

### 1.3 Runtime changes — new `MLB2` format

Replace the body of `matlab_save_mat` / `matlab_load_mat` and add the
`_named` entry points. The on-disk layout for `MLB2`:

```
"MLB2"                4 bytes magic
int32 version         = 1
int32 record_count

per record:
  int32 name_len
  char[name_len] name
  int32 kind          (1=mat double, 2=scalar f64, 3=string, 4=struct)
  payload:
    kind=1: int64 rows, int64 cols, rows*cols doubles
    kind=2: f64
    kind=3: int32 len, char[len]
    kind=4: nested record_count + records (recursive)
```

Records are streamed in declaration order. The reader builds a fresh
`matlab_struct *` and returns it; on filter args, only matching names
are inserted.

This is still our format — old `MLB1` files become a one-off shim
that the reader detects and converts to a single-record struct with
field name `"data"` (mirrors what `load` of a v4 .mat file does in
real MATLAB: it returns a struct with the matrix under its declared
variable name).

Files touched: `runtime/matlab_runtime.{c,h,hpp}`,
`runtime/matlab_runtime.py`, `runtime/matlab_runtime.ts`.

### 1.4 Tests (Phase 1)

- `test/Run/io_save_load.m` — update to the new API; keep the
  existing semantic checks.
- `test/Run/io_save_load_multi.m` — new: save two vars, load all,
  load with filter.
- `test/Run/io_save_load_struct.m` — new: round-trip a `struct(...)`.
- `test/Sema/save_load_undef.m` — new: `save('f', 'X')` where X is
  not in scope must error.

## 2. Phase 2 — `.mat` v5 interop

Goal: files we write are openable in real MATLAB; files written by
MATLAB load here. Uncompressed v5 first; zlib (v7) optional.

### 2.1 MAT-File v5 layout (reference)

128-byte header:

```
offset  size  field
0       116   text description (printable ASCII, padded with spaces)
116     8     subsystem-specific data offset (zero for our writes)
124     2     version          = 0x0100
126     2     endian indicator = 'M','I' (big) or 'I','M' (little)
```

Followed by a sequence of **Data Elements**, each with an 8-byte tag:

```
int32 type
int32 num_bytes_of_payload
... payload, padded to 8-byte boundary ...
```

A "small data element" form packs `type` + `nbytes` + up to 4 bytes
of payload into one 8-byte word for short payloads (used heavily).

Top-level variables are `miMATRIX` (type=14) elements. A `miMATRIX`
payload is itself a sequence of sub-elements:

1. **Array Flags** (`miUINT32`, 8 bytes): low byte = `mxCLASS`,
   next byte = flags (complex / global / logical bits).
2. **Dimensions Array** (`miINT32`, N\*4 bytes).
3. **Array Name** (`miINT8`, name_len bytes).
4. **Real part** (e.g. `miDOUBLE`, rows\*cols\*8 bytes).
5. **Imaginary part** if complex.

Classes we need at minimum:

- `mxDOUBLE_CLASS` = 6
- `mxCHAR_CLASS` = 4 (for MATLAB strings)
- `mxSTRUCT_CLASS` = 2 (for struct)
- `mxLOGICAL_CLASS` = 9 (encoded via a flag bit on numeric class)
- Later: `mxCELL_CLASS` = 1, integer classes 8..15, `mxSINGLE_CLASS = 7`.

`mxSTRUCT_CLASS` payload extends the standard form with two extra
sub-elements after the name: a `miINT32` "field name length" cap, and
a `miINT8` block of concatenated field-name C-strings padded to that
length. Then one nested `miMATRIX` per field per element.

### 2.2 Implementation order

1. **Writer, doubles only.** Replace `MLB2` payload encoding with v5
   bytes for `mxDOUBLE_CLASS` matrices. Header + miMATRIX per record.
   Verify by opening files in MATLAB / `scipy.io.loadmat`.
2. **Reader, doubles only.** Parse header, walk top-level miMATRIX
   elements, decode dims + name + double payload. Reject unsupported
   classes with a clear diagnostic.
3. **Strings.** Encode `matlab_string` as `mxCHAR_CLASS` with UTF-16
   payload (`miUINT16`). MATLAB will display them as char arrays;
   to round-trip *string scalars* (the `"..."` type) we also need
   support for the v7.3 string container or the ad-hoc cell-array
   trick MathWorks uses. Defer string-scalar parity; ship char first.
4. **Structs.** `mxSTRUCT_CLASS` writer + reader. Recurse on field
   payloads.
5. **Logicals + integer classes.** Cheap once doubles work — same
   encoder, different `mxCLASS` byte and element type tag.
6. **Complex.** Set the complex flag in array flags, write a second
   payload sub-element. Pulls from the existing complex runtime.
7. **(Optional) v7 / zlib.** Wrap each top-level miMATRIX in a
   `miCOMPRESSED` element; deflate with a small zlib helper. Adds a
   `-lz` link and one new code path; gated by a flag on the writer.

### 2.3 Endianness + alignment

- Always write little-endian (host on supported targets) and set the
  header indicator to `'I','M'`. The reader honors the flag and
  byte-swaps if needed.
- Every Data Element payload is padded to 8 bytes. Use a small
  `pad_to_8(FILE *)` helper.

### 2.4 Cells, sparse, objects, function handles

Out of scope for v1 of Phase 2. Reader emits `unsupported MAT class
%d for variable '%s'` and skips. Writer refuses with a typed
diagnostic when asked to save these kinds.

### 2.5 Tests (Phase 2)

- Round-trip via `scipy.io.savemat` / `loadmat` against our reader
  and writer (Python tooling lives in `runtime/matlab_runtime.py`;
  reuse).
- Golden `.mat` files committed under `test/Inputs/mat/` covering
  doubles, char, struct, logical, int32, complex, multi-var.
- Negative tests: malformed header, truncated payload, unsupported
  class — each must produce a clean diagnostic, not a crash.

## 3. Sequencing + checkpoints

1. **Phase 1.1–1.3** behind feature flag `--mat-api-v2` in the
   driver. Default off for one release while the migration lands.
2. Cut over `examples/` and `test/Run/io_save_load.m` to the new
   API. Remove the flag.
3. **Phase 2.2 step 1** (doubles writer + reader) lands as a separate
   PR with `.mat` golden tests and a MATLAB-side smoke test if
   MATLAB is available; otherwise validate against `scipy.io`.
4. Subsequent Phase 2 steps land incrementally; each is independently
   useful and can ship without the others.

## 4. References

- MAT-File Format specification: MathWorks document `matfile_format.pdf`
  (versions 5/7). The bytes-on-disk reference for Phase 2.
- `scipy.io.matlab` — pure-Python implementation, useful as a known-
  good cross-check.
- `libmatio` — C library implementing v5 / v7 / v7.3. Useful if we
  ever decide to vendor instead of writing our own; HDF5 dependency
  is the main reason we don't.
