# The MATLAB Language — Tutorial

`matlab_llvm` compiles a **practical subset of the MATLAB language** to
native code through MLIR and LLVM (and emits C / C++ / Python / TypeScript /
SystemVerilog from the same source). This is the front-door tutorial: it
walks the core `.m` language — scalars, matrices, control flow, functions,
`classdef`, I/O, and plotting — *before* any toolbox. Every code excerpt
below comes from a real, runnable file under `examples/`.

The authoritative inventory of what is supported, partial, or missing is
[`../feature_status.md`](../feature_status.md); read it when you need to
confirm whether a specific feature exists. For the toolbox surfaces built
on top of this language (Signal, Control, Optimization, …) see the toolbox
index in [`README.md`](README.md).

---

## Running your first program

The canonical first program, `examples/hello.m`:

```matlab
% The canonical first program.
disp('Hello, world!');
fprintf('Greetings from matlab_llvm!\n');
```

Compile it to LLVM IR, link against the runtime archive, and run:

```bash
build/matlabc -emit-llvm examples/hello.m > /tmp/hello.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/hello.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/hello
/tmp/hello
```

That two-command flow (`matlabc -emit-llvm …` then `clang++ … -Wl,-dead_strip`)
is the same for every example in this tutorial — only the filename changes.
The linker dead-strips unused runtime code, so a `disp(1)` program links to
about 50 KB. Full details, including how Cairo (plotting) and SymPP (symbolic)
are `dlopen`'d lazily, are in [`../build_and_run.md`](../build_and_run.md).

`matlabc` is also an interactive REPL (`matlabc -repl`) backed by the MLIR
ExecutionEngine with a persistent workspace and `who` / `whos` / `clear`. The
REPL is **experimental** (🟡) — no line editing or live user-function
definitions yet (see [`../repl.md`](../repl.md)).

The same `.m` source compiles to several backends via `-emit-*` flags:
`-emit-mlir`, `-emit-c`, `-emit-cpp`, `-emit-python`, `-emit-typescript`,
`-emit-systemverilog`. The default LLVM lane is used throughout this tutorial;
the other backends are covered in [Output backends](#output-backends).

---

## Scalars, matrices & operators

Everything numeric is a dense matrix; a scalar is a 1×1 matrix. Build a
matrix literal with spaces (or commas) between columns and semicolons (or
newlines) between rows. From `examples/matrix_mult.m`:

```matlab
A = [1 2 3;
     4 5 6;
     7 8 10];

B = [1 0 0;
     0 2 0;
     0 0 3];

disp('A * B =');
disp(A * B);        % matrix multiply

disp('A .* B =');
disp(A .* B);       % element-wise multiply

disp('A'' =');
disp(A');           % transpose (use '' for a literal quote in a char array)
```

The full operator set is supported: `+ - * / \ ^` and their element-wise
forms `.* ./ .\ .^`, transpose `'` and non-conjugate transpose `.'`,
comparisons (`== ~= < <= > >=`), and logical operators (`& | && || ~`).

The backslash `\` is MATLAB's left-divide — it **solves** `A x = b` rather
than inverting. From `examples/solve_linear.m`:

```matlab
A = [2 1 1;
     1 3 2;
     1 1 1];
b = [5; 10; 6];

x = A \ b;          % solve A x = b
disp('A * x =');
disp(A * x);        % reproduces b
```

Core linear algebra ships as built-ins. From `examples/eigendecomp.m`:

```matlab
disp(eig(A));                 % eigenvalues as a column vector
[V, D] = eig(A);              % eigenvectors V, diagonal eigenvalue matrix D
disp(V * D * V');             % reconstructs A
disp(det(A));                 % determinant
disp(inv(A));                 % inverse
```

`eig` accepts both the single-return (eigenvalues) and two-return
(`[V, D]`) shapes. The broader linear-algebra surface — `lu`, `qr`, `chol`,
`pinv`, `norm`, `trace`, `kron`, `expm`, `rank`, `null`, `orth` and more —
is listed in [`../feature_status.md`](../feature_status.md) §4.

Ranges use the colon: `0:0.05:10` is `start:step:stop` (the step defaults to
1, and may be negative). Ranges fold to a concrete vector at compile time.

---

## Control flow

`if / elseif / else`, `for`, `while`, `switch`, `break`, `continue`, and
`return` all work. From `examples/even_odd.m` — `if`/`else` inside a `for`:

```matlab
N = 6;
even_count = 0;
odd_count = 0;
for i = 1:N
    if mod(i, 2) == 0
        fprintf('%g is even\n', i);
        even_count = even_count + 1;
    else
        fprintf('%g is odd\n', i);
        odd_count = odd_count + 1;
    end
end
```

`while` loops with `break` and convergence tests, from `examples/while_loop.m`:

```matlab
% Break out early once a running product exceeds a threshold.
n = 1;
p = 1;
while 1 == 1
    p = p * n;
    if p > 1000
        break;
    end
    n = n + 1;
end

% Newton-Raphson driven by a convergence check rather than a fixed count.
x = 1.0;
while abs(x * x - 2) > 1e-12
    x = (x + 2 / x) / 2;
end
```

Loop-carried state is the idiomatic iterative pattern — `examples/fibonacci.m`
carries `a`, `b` across iterations:

```matlab
a = 0;
b = 1;
i = 0;
while i < n
    disp(a);
    t = a + b;
    a = b;
    b = t;
    i = i + 1;
end
```

**Logical indexing** is a first-class way to select and count elements. From
`examples/logical_mask.m`:

```matlab
disp(A(A > 0));               % flatten the positive entries
disp(A(A < 0));               % flatten the negative entries

% sum-of-logical: count entries strictly above the mean
disp(sum(A(:) > mean(A(:))));
```

`A(A > 0)` indexes `A` with a logical mask; `A(:)` flattens to a column.
Summing a logical array counts the `true` entries.

> **Note on short-circuiting**: `&&` and `||` parse and evaluate, but the
> frontend emits *both* operands eagerly — there is no runtime short-circuit
> skip. Don't rely on the right-hand side being guarded against an error by
> the left (`feature_status.md` §1, lexical row).

---

## Functions & handles

A `.m` file can be a script (top-level statements) or define `function`s at
the bottom. Functions support recursion and multiple returns. From
`examples/factorial.m`:

```matlab
function y = fact(n)
    if n <= 1
        y = 1;
    else
        y = n * fact(n - 1);
    end
end
```

A predicate function returning a logical, from `examples/is_old.m`:

```matlab
function r = is_old(age)
    r = age > 18;
end
```

Multiple return values use the `[a, b] = f(...)` form on the call side (you
saw `[V, D] = eig(A)` above). Dispatch on `nargin` / `nargout` is supported.

**Function handles** take a pointer to a builtin or a user function. From
`examples/func_handles.m`:

```matlab
f = @sin;
disp(f(0));                   % call sin through the handle
g = @sqrt;
disp(g(16));

p = @mySq;                    % handle to a user function
disp(p(6));

function y = mySq(x)
    y = x * x;
end
```

Supported builtin handles are the scalar `f64 -> f64` math functions
(`@sin @cos @tan @exp @log @sqrt @abs`); user-function handles resolve to
direct calls at compile time.

**Anonymous functions** close over values by snapshot at the moment `@(...)`
is written — reassigning a captured variable afterwards does not change the
handle. From `examples/anon_capture.m`:

```matlab
k = 5;
f = @(x) x + k;
disp(f(3));                   % 8

a = 2; b = 3;
g = @(x) a * x + b;           % two captures in one expression
disp(g(5));                   % 13

k = 100;
disp(f(0));                   % still 5 — capture is by value

A = [1 2 3; 4 5 6; 7 8 9];
diagi = @(i) A(i, i);         % matrix capture (by pointer at @-time)
disp(diagi(2));               % 5
```

**Persistent** variables are function-local state that survives across calls
— MATLAB's equivalent of C's `static` locals. From
`examples/persistent_counter.m`:

```matlab
function y = count()
    persistent n;
    n = n + 1;
    y = n;
end
```

Calling `count()` three times yields 1, 2, 3. `persistent` (and `global`)
are scalar `f64` today. The first-call-init idiom
`if isempty(x); x = init; end` is recognized specially by the C/C++/Python/TS
backends and lowered to a static initializer.

---

## Structs, strings & cells

**Structs** support scalar fields, nesting (`s.a.b`), dynamic fields
(`s.(name)`), `isstruct` / `isfield` / `rmfield`, and **struct arrays**
(`s(i).x`):

```matlab
s.name = "rotor";
s.rpm  = 3600;
disp(s.rpm);

p(1).x = 1;                   % struct array — auto-promotes the binding
p(2).x = 4;
disp(numel(p));               % 2
```

Scalar struct fields work end-to-end; matrix-valued struct fields carry a
known tensor→pointer conversion gap (see
[`../feature_status.md`](../feature_status.md) §4 heterogeneous-data rows).

**Strings**: double-quoted `"..."` strings and single-quoted `'...'` char
arrays are both supported, along with concatenation (`[s1 s2]`, `strcat`,
`s1 + s2`), `num2str`, `str2double`, `strtrim`, `strrep`, `upper`, `lower`,
`startsWith`, `endsWith`, and `contains`. Not yet supported: `strsplit`,
`strjoin`, `regexp`, `regexprep`, `str2num` (all ❌).

**Cell arrays**: 1-D and 2-D cell literals (`{a, b}`, `{a, b; c, d}`), read
and write via `C{i}` / `C{r, k}`, `numel`, `iscell`, and bracket
concatenation. `containers.Map` / `dictionary` are also shipped:

```matlab
C = {1, "two", [3 3 3]};
disp(C{2});

m = dictionary("a", 1, "b", 2);
disp(m("a"));                 % 1
disp(isKey(m, "b"));          % 1
```

`cellfun` / `arrayfun` are registered but only partially wired (🟡).

---

## classdef

Object-oriented programming is supported as an MVP: `properties`, `methods`,
constructors that read `nargin`, instance methods via dot-syntax,
`Dependent` properties with `get.Prop`, single inheritance (`< Parent`),
static methods, and operator overloading (`plus`, `minus`, `mtimes`, `eq`, …).
From `examples/bank_account.m`:

```matlab
classdef BankAccount
    properties
        Id
        Balance
    end
    properties (Dependent)
        Overdrawn
    end
    methods
        function obj = BankAccount(id, bal)
            if nargin == 2
                obj.Id = id;
                obj.Balance = bal;
            end
        end
        function deposit(obj, amt)
            obj.Balance = obj.Balance + amt;
        end
        function f = get.Overdrawn(obj)
            if obj.Balance < 0
                f = 1;
            else
                f = 0;
            end
        end
        function r = eq(a, b)            % overloads ==
            if a.Id == b.Id
                r = 1;
            else
                r = 0;
            end
        end
    end
end

classdef Savings < BankAccount          % inheritance
    properties
        Rate
    end
    methods
        function obj = Savings(id, bal, rate)
            if nargin == 3
                obj.Id = id; obj.Balance = bal; obj.Rate = rate;
            end
        end
        function i = interest(obj)
            i = obj.Balance * obj.Rate;
        end
    end
end
```

The driving script reads `acc.Id`, calls `acc.deposit(250)`, reads the
dependent `acc.Overdrawn`, compares objects with `acc == other`, and uses
the inherited methods from `Savings`.

Currently all objects are **handle-shaped** at runtime (value-class copy
semantics are partially shipped — `b = a` clones, but in-method value
semantics are not yet enabled). Events/listeners, property validators
(parsed but not enforced), and `handle` destructors are not yet implemented
— see [`../feature_status.md`](../feature_status.md) §1 / §8.

---

## Built-ins & I/O

Display and formatting: `disp` (scalar, vector, matrix, string), `fprintf`
(with escape sequences, up to 4 numeric args), `sprintf` (literal +
single-f64 form), plus `error`, `warning`, `assert`, and `input`. The
`%g` / `%d` / `%.0f` conversions are the common ones; see `examples/even_odd.m`
and `examples/hello.m`.

A wide built-in library ships in the runtime, including:

- **Creation & shape**: `zeros`, `ones`, `eye`, `rand`, `randn`, `magic`,
  `diag`, `reshape`, `repmat`, `linspace`, `size`, `length`, `numel`,
  `ndims`, `flip` / `fliplr` / `flipud`, `rot90`.
- **Element-wise math**: `abs`, `sqrt`, `exp`, `log`, the trig family
  (incl. degree variants `sind`/`cosd`/…), `floor`, `ceil`, `round`, `fix`,
  `mod`, `rem`, `sign`, and `fft` / `ifft`.
- **Reductions**: `sum`, `min`, `max`, `mean`, `prod`, `cumsum`, `cumprod`
  — all with the all-elements, column-wise, and `(A, dim)` forms.
- **Indexing / search**: `find`, `isempty`, `isequal`, `sort`, `sortrows`,
  `unique`, `ismember`, `setdiff`, `intersect`, `union`, `sub2ind`, `ind2sub`.

> Note: `std`, `var`, `median`, `mode` are **not** core built-ins (❌ in
> §4 reductions); they are provided by the Statistics toolbox surface.
> `examples/tier1_demo.m`–`tier3_demo.m` exercise a broad spread of the
> array/DSP built-ins (`filter`, `any`, `all`, `tril`, `triu`, `meshgrid`,
> `polyval`, `interp1`, `trapz`, …) with hand-verified expected outputs.

**Timing**: `tic` / `toc` / `pause` are available (🟡 — implementation
varies). From `examples/tic_toc_pause.m`:

```matlab
tic;
pause(0.25);
elapsed = toc();
fprintf('measured %g s (expect ~0.25)\n', elapsed);
```

**File / CSV I/O**: text and binary single-matrix round-trips work via
`fopen` / `fclose` / `fprintf(fid,…)` / `fgetl` / `fread` / `fwrite` /
`save` / `load` (🟡 — `save`/`load` use a custom `MLB1` header, **not**
MATLAB's `.mat` format). Higher-level tabular readers like `readtable`
belong to the table/data toolbox surface (see `examples/csv_stats.m`, which
loads a CSV into a `table` and runs per-column stats), not the core language.

---

## Plotting

Plotting is **headless**: a pure-C++/Cairo backend renders `plot` / `bar` /
`surf` / `contour` / `imagesc` / … to PNG, SVG, or PDF with no display server.
From `examples/plot/sine_wave.m`:

```matlab
x = 0:0.05:10;
y = sin(x);

figure;
plot(x, y, 'r--');
title('sin(x) on [0, 10]');
xlabel('x');
ylabel('sin(x)');
grid on;
saveas(gcf, '/tmp/plot_sine.png');
```

Build it the usual way (plotting requires a runtime configured with
`-DMATLAB_LLVM_WITH_PLOT=ON`); the first `plot`/`figure`/`saveas` call
`dlopen`s Cairo lazily, so non-plotting programs need no Cairo at all.
The full plotting surface and roadmap are in [`../plotting.md`](../plotting.md).

Interactive figures, mouse picking, `ginput`, pan/zoom/rotate, and App
Designer are **out of scope** — output is files, not windows.

---

## Output backends

The same `.m` source lowers through one shared pipeline to multiple targets,
selected by an `-emit-*` flag:

| Flag | Backend | Status |
|---|---|:-:|
| `-emit-llvm` | LLVM IR → native (default lane in this tutorial) | ✅ |
| `-emit-c` | self-contained C | ✅ |
| `-emit-cpp` | C++ (classes + inheritance preserved) | ✅ |
| `-emit-python` | NumPy-backed Python | ✅ |
| `-emit-typescript` | TypeScript | 🟡 |
| `-emit-systemverilog` | synthesizable SystemVerilog (ASIC) | ✅ |
| `-emit-mlir` | MLIR (with `-opt` for the optimized form) | ✅ |

Multi-return maps idiomatically per backend (out-pointer params in C,
`std::tuple` in C++, tuple unpacking in Python, array destructuring in TS).
The C/C++/Python/TS emit lanes are validated against the LLVM lane for
byte-identical output across the run-test corpus. See
[`../build_and_run.md`](../build_and_run.md) for the compile flow and the
`emit_*.md` docs for backend specifics.

---

## What's supported (and what isn't)

**In scope** — the practical subset this tutorial covers:

- dense numeric (`double`) matrices, `logical`, `char`, `string`, and
  `complex`; `int32` / `uint8` typed matrix lanes (other int widths are
  f64-shadowed);
- the full operator set, ranges, scalar/slice/logical indexing;
- `if` / `for` / `while` / `switch` / `try`–`catch`, `break` / `continue` /
  `return`, `parfor`;
- named functions (recursion, multi-return), function handles, anonymous
  functions with captures, `global` / `persistent` (scalar);
- structs (incl. struct arrays), 1-D/2-D cells, `containers.Map`/`dictionary`,
  strings;
- `classdef` with properties, methods, constructors, single inheritance,
  `Dependent` properties, static methods, and operator overloading;
- a large built-in / linear-algebra library, headless plotting, and basic
  file I/O;
- 2-D arrays everywhere; arbitrary-depth 3-D arrays for `zeros`/`ones` +
  scalar indexing + `size`/`numel`/`ndims`.

**Partial (🟡)** — parsed/modelled but with incomplete runtime or backend
coverage: the REPL/LSP/DAP tooling, narrower/wider integer lanes, `single`,
N-D (>2D) slicing, `fi` function-internal typing, `cellfun`/`arrayfun`,
matrix-valued struct fields, `fieldnames`, `tic`/`toc`/`pause`,
`save`/`load` (custom format, not `.mat`).

**Out of scope (deliberate non-goals)**:

- interactive UIs / GUIs — App Designer, Live Editor inline plots, `ginput`,
  pan/zoom/rotate (headless plotting *is* shipped);
- full MATLAB numerical bit-exactness (pure-C linalg vs LAPACK differs in
  the last few ULPs — correct to tolerance, not to bit);
- `.mat` file compatibility, `.mlx` Live Scripts, MEX interop, `gpuArray`,
  `eval` / `evalin` / `assignin`, `matlab.unittest`;
- N-D arrays beyond 3-D, sparse matrices, the complex-linalg tail.

The complete, authoritative status — feature by feature, with notes — is in
[`../feature_status.md`](../feature_status.md).

---

## See also

- Full feature inventory: [`../feature_status.md`](../feature_status.md)
- Compile & run details: [`../build_and_run.md`](../build_and_run.md)
- Plotting surface: [`../plotting.md`](../plotting.md)
- Toolbox tutorials: [`README.md`](README.md)
