# Bioinformatics Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL/JIT**, and **demo** Bioinformatics-Toolbox programs.

Source: *Bioinformatics Toolbox User's Guide* (R2026a, 6 chapters:
Getting Started · High-Throughput Sequence Analysis · Sequence Analysis ·
Microarray Analysis · Phylogenetic Analysis · Mass Spectrometry and
Bioanalytics).

This is a **high-reuse, low-keystone toolbox** — unlike the numeric
toolboxes (Control / PDE) that needed a new linear-algebra primitive
first, Bioinformatics is almost entirely *string/character algorithms +
lookup tables + reuse of already-shipped kernels*. The substrate it
needs is already in the tree:

- **Sequences are char vectors / strings** — the lane shipped with the
  recent char-vector work (`feat/char-vector-234`). A DNA/RNA/protein
  sequence is a `1×N char`; a sequence set is a cell array of chars or a
  struct array (`Header`/`Sequence`). No new container type.
- **Pairwise/multiple alignment is dynamic programming over a score
  matrix** — pure matrix-lane fill + traceback, the same idiom as the
  shipped `dtw`/edit-distance and `seqpdist`-style routines. No new
  primitive.
- **Scoring matrices (`blosum`/`pam`/`dayhoff`/`gonnet`/`nuc44`) are
  hard-coded lookup tables** — the exact precedent of the Comm 5G-NR base
  matrices, the Image `fspecial` kernels and the Wavelet family-filter
  catalogue.
- **Profile HMMs reuse the shipped Stats HMM engine** (`hmmviterbi` /
  `hmmdecode` / `hmmtrain` — confirmed in `Resolver.cpp`): a profile HMM
  is a position-specific emission/transition matrix run through the same
  Viterbi / forward-backward / Baum-Welch machinery.
- **Phylogenetic distance + clustering reuse `pdist`/`squareform`**
  (shipped) — `seqlinkage` is UPGMA/WPGMA agglomerative clustering and
  `seqneighjoin` is neighbor-joining over a distance matrix; the
  `phytree` object is the alloc-then-populate + class-pinned-dispatch
  classdef pattern proven by `tf`/`ss`/`LinearModel`/`ClassificationModel`.
  **NOTE:** Stats carved `linkage` out, so the agglomerative step is
  hand-rolled here (small — it is the standard min-distance merge loop).
- **Microarray clustering + viz reuse `kmeans`/`pca`/`fitcsvm`** (shipped)
  and the Image heatmap/colormap path; `clustergram` is a dendrogram +
  reordered heatmap (hand-rolled UPGMA + the shipped `imagesc`).
- **Mass-spec preprocessing reuses the shipped Signal surface** —
  `resample`/`sgolayfilt`/`findpeaks`/baseline smoothing (all confirmed
  in `Resolver.cpp`): `msresample`/`msgolay`/`mspeaks`/`msbackadj` are
  thin wrappers with bio-domain defaults.
- **Statistical learning is a direct call into the shipped Stats/ML
  toolbox** — `rankfeatures`/`randfeatures`/`knnimpute`/`classperf`/
  `crossvalind` sit on top of `kmeans`/`fitc*`/`pca`/`pdist`.

**No external dependency** (no BioPerl, no NCBI E-utilities at runtime,
no libsam/htslib) — every algorithm is a hand-coded routine over the
shipped kernel, and every biological constant (codon tables, scoring
matrices, amino-acid masses/pK values) is a baked-in lookup table.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/bioinfo/align_globin_pair.m`](../examples/bioinfo/align_globin_pair.m):
*load two hemoglobin-beta protein sequences from a bundled FASTA file
(`fastaread`), globally align them with `nwalign` under `BLOSUM62`,
report the alignment score + percent identity, and render the
`seqdotplot` self-similarity map*. This exercises the
`fastaread` → sequence-char-lane → DP-alignment → score/identity arc
end-to-end; achieving it closes **Bio-Tier-1/2** (the foundation + the
single most common reason anyone reaches for this toolbox). The companion
[`examples/bioinfo/primate_phylotree.m`](../examples/bioinfo/primate_phylotree.m)
(`seqpdist` → `seqlinkage` → `phytree` → `plot` cladogram of primate
mitochondrial sequences) is the **Bio-Tier-4** tracer-bullet, and
[`examples/bioinfo/yeast_clustergram.m`](../examples/bioinfo/yeast_clustergram.m)
(microarray expression → `quantilenorm` → gene filtering → `clustergram`)
is the **Bio-Tier-6** one.

Companion docs:
[`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md) (HMM engine,
`kmeans`/`pca`/`fitcsvm`/`pdist` — the phylo/microarray/learning reuse
base), [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) (the
mass-spec preprocessing surface — `resample`/`sgolayfilt`/`findpeaks`),
[`image_toolbox_roadmap.md`](image_toolbox_roadmap.md) (heatmap / colormap
/ `imagesc` for clustergram & `msheatmap`),
[`plotting.md`](plotting.md) (dot plots, phylogram/cladogram, scalogram
all route through the Cairo backend), [`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the sequence lane: representation, alphabets, conversion, statistics,
  and FASTA I/O. **Tier-2** is pairwise alignment + scoring matrices (the
  headline pillar). **Tier-3** is multiple-sequence alignment + sequence
  profiles + profile HMMs. **Tier-4** is phylogenetics (`phytree` +
  distance builders). **Tier-5** is protein property + structural analysis
  (`molweight`/`isoelectric`/in-silico digestion/PDB). **Tier-6** is the
  high-reuse application layer: microarray normalization/clustering, mass
  spectrometry preprocessing, and statistical learning helpers, plus the
  local-file format parsers.
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~1.5 wk · T2
  ~1.5 wk · T3 ~2 wk · T4 ~2 wk · T5 ~1 wk · T6 ~3 wk (~11 wk full)**.
  Each tier is independently shippable and demoable; **T1 + T2 alone
  (~3 wk) close the 80% sequence-alignment workflow** — the canonical
  reason anyone reaches for this toolbox. Badge would advance by one.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started. **ALL 6 TIERS
  SHIPPED 2026-06-07 — the Bioinformatics Toolbox is complete** (Phases A+B+C),
  `runtime/toolbox/bioinfo/runtime_bioinfo.cpp` (~1.5 kLOC) + `bioinfo_classdefs.m`
  (`phytree` + `DataMatrix`). 8 gating tests
  (`test/Run/bioinfo_{seqstats,nwalign,fasta,msa,phylo,protein,microarray,massspec}.m`)
  + 7 examples (`examples/bioinfo/`). Full suite: **Run 750/0, frontend 83/0,
  emit-c/py/ts 324/266/231 /0**.
  **Phase C (Tier-5 protein + Tier-6 application layer) ✅ SHIPPED 2026-06-07**:
  T5 `molweight`/`atomiccomp`/`isoelectric`/`aminolookup`/`cleave`(protease)/
  `restrict`(restriction enzyme) over baked AA-mass/pK/atomic + enzyme tables;
  T6 `quantilenorm`/`manorm`/`genevarfilter`/`generangefilter`/`genelowvalfilter`/
  `clustergram`(UPGMA leaf order) + `msnorm`/`mslowess`/`msbackadj`/`mspeaks`/
  `msresample` + `rankfeatures`/`knnimpute`/`crossvalind` (matrix lane) + the
  second classdef **`DataMatrix`** (pure-`.m` arg-constructor, the `mpc(...)`
  precedent — no runtime populate). Carve-downs (documented): PDB structural
  (`pdbread`/`ramachandran`/`pdbdistplot`), `proteinplot`, profile-HMM,
  `phytreeread`, `plot(phytree)`/clustergram heatmap render, web-DB readers,
  NGS/BioMap.
  **Phase B (Tier-3 MSA + Tier-4 phylogenetics) ✅ SHIPPED 2026-06-07** — adds
  `multialign`/`profalign`/`seqconsensus`/`seqprofile` (progressive MSA over
  the shipped NW core; sequence-set input = a **cell array of char**, aligned
  output = a **newline-joined string**), `seqpdist` (p-distance + Jukes-Cantor
  over cell input → pdist row vector), and the first classdef **`phytree`**
  (`runtime/toolbox/bioinfo/bioinfo_classdefs.m`) built by `seqlinkage`
  (UPGMA/WPGMA/single/complete) and `seqneighjoin` (neighbor-joining), with
  `getnewickstr`/`pdist`(patristic)/property-read methods + `phytreewrite`.
  2 gating tests (`bioinfo_{msa,phylo}.m`) + 2 examples
  (`primate_phylotree.m` headline, `msa_consensus.m`). Suite: Run 747/0,
  frontend 83/0, emit-c/py/ts 324/266/231 /0. **Phase-B wiring notes/traps**:
  classdef = `bioinfo_classdefs.m` (phytree shell + method forwarders);
  `seqlinkage`/`seqneighjoin` are alloc-then-populate constructor-intercepts
  in Lowering.cpp (mirror `fit`), class-pinned via Resolver `inferClassForCall`
  (`phytree`); the prelude is wired in BOTH main.cpp paths — REPL `Want` table
  (~2417) + static `extClassLeaf`/detection-names (~11974/~12247) — AND
  `bioinfo` added to BOTH `kToolboxDirs` lists; method names called
  free-function (`getnewickstr`) MUST be registered as builtins in Resolver
  (the `coeffvalues` precedent) else "undefined name"; use `tr.NumLeaves`
  property reads (avoid a generic `get` builtin); populate fns return
  `matlab_mat*` (result ignored via NoneType); cell-of-strings read via
  `matlab_cell_numel`/`_get_mat` (elements = char-code rows). Profile-HMM
  (3.6-3.8), `phytreeread`, `plot(phytree)` deferred to a Tier-3/4 follow-on.
  **Phase A (Tier-1 + Tier-2) ✅ SHIPPED 2026-06-07** in
  [`runtime/toolbox/bioinfo/runtime_bioinfo.cpp`](../runtime/toolbox/bioinfo/runtime_bioinfo.cpp);
  Tiers 3–6 🔵 (not started). The reuse anchors
  (`hmmviterbi`/`hmmdecode`/`hmmtrain`, `pdist`/`squareform`,
  `kmeans`/`fitcsvm`, `regexp`, `resample`/`sgolayfilt`/`findpeaks`) are all
  ✅ shipped. **Phase A shipped surface**: Tier-1 `nt2int`/`int2nt`/`aa2int`/
  `int2aa`/`nt2aa`/`dna2rna`/`rna2dna`/`seqcomplement`/`seqrcomplement`/
  `seqreverse`/`basecount`/`aacount`/`randseq`/`fastaread`/`fastawrite`;
  Tier-2 `blosum`(62)/`nuc44`/`nwalign`/`swalign`/`seqdotplot`. 3 gating
  tests (`test/Run/bioinfo_{seqstats,nwalign,fasta}.m`) + 3 examples
  (`examples/bioinfo/{align_globin_pair,orf_translate,fasta_align}.m`).
  **Phase-A wiring notes & traps** (for Phase B/C):
  - Sequences flow as `matlab_string*` (the `'ACGT'` lane). Sequence
    transforms register in the LowerTensorOps single-return spec table
    (PtrTy in/out); `nwalign`/`swalign` are multi-return (`wmret`) — each
    output is an independent runtime call, score = 1×1 `matlab_mat*`,
    alignment = a 3-row newline-joined `matlab_string*`.
  - String-returning sequence functions MUST be added to
    `Lowerer::isStringReturningBuiltin` (Lowering.cpp) or `length()` on the
    result reads the `matlab_string` as a matrix and returns garbage
    (`disp`/`strcmp` survive via the runtime string registry, but `length`
    does not).
  - `basecount`/`aacount` return a `matlab_struct*` — add the name to the
    `RhsIsStruct` list (Lowering.cpp) for `b.A` field access.
  - `fastaread` returns a `matlab_struct_arr*`. Tag the LHS into
    `StructArrayBindings` (new `RhsIsStructArray` flag) AND record its
    `Header`/`Sequence` fields in `MatStructFields`, and extend the
    `s(i).Field` read path (Lowering.cpp ~14188) to consult `MatStructFields`
    via the array binding — else element fields default to `get_f64`→0.
    Store the fields with `matlab_struct_set_string` (kind=3), NOT
    `set_mat` (kind=1 segfaults the reader).
  - **Known gap**: `fprintf('%s', s(i).Sequence)` crashes (matrix-typed
    struct-array field fed to `%s`); use `disp(s(i).Sequence)` instead.
    Custom numeric scoring matrices and BLOSUM-N other than 62 are
    carve-downs (the named-string 3rd arg path is shipped).
- **Sequences live in the char/string lane** — proven by
  `feat/char-vector-234`. A single sequence is `1×N char`; a sequence set
  is read into a struct array `s(i).Header` / `s(i).Sequence` (FASTA) or a
  cell array of chars. Conversion functions (`nt2int` etc.) bridge to the
  numeric matrix lane so the DP fills and statistics run on `double`
  matrices.
- **Scoring matrices and biological constants are baked-in tables** —
  `blosum(62)` returns the 24×24 BLOSUM62 matrix from a hard-coded
  catalogue; codon→amino-acid maps, amino-acid molecular weights and pK
  values, the standard/mitochondrial genetic codes are all lookup tables
  in the runtime, fetched by a caller-supplied key (the Image
  `imread('f.png')` / Wavelet `wfilters('db4')` precedent).
- **Only `phytree` (Tier-4) and `DataMatrix` (Tier-6) need the classdef
  descriptor** (alloc-then-populate + class-pinned dispatch + REPL persist
  + DAP render); everything else is plain matrix/char/struct-in →
  matrix/char/struct-out builtins. This keeps Tiers 1–3 (the alignment
  headline) entirely out of the classdef machinery.
- **Web database access and apps are carved out** (see §9): `getgenbank`/
  `getpdb`/`getgeodata`/`blastncbi`/NCBI E-utilities need live network and
  are stubbed (clear "not supported in offline build" error); the
  interactive apps (`seqalignviewer`/`phytreeviewer`/`msviewer`/Sequence
  Alignment App / Genomics Viewer / Biopipeline Designer) are out of
  scope. The corresponding *local-file* readers (`fastaread`/`genbankread`/
  `pdbread`/`geosoftread`) **are** in scope.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Bioinformatics code yet)

| Group | Surface (already shipped) | Location | How Bioinformatics uses it |
|---|---|---|---|
| Char / string lane | `1×N char`, string ops, indexing, concat | `feat/char-vector-234`, `lib/Sema`, `runtime/matlab_runtime.cpp` | The sequence container — DNA/RNA/protein sequences are char vectors; headers are strings (Tier-1). |
| Regular expressions | `regexp`, `regexprep`, `strfind`, `strsplit` | `lib/Sema/Resolver.cpp` (`regexp` ✅) | `seq2regexp` motif → regex, `restrict`/`cleave` recognition-site search, `palindromes`, FASTA/GenBank record parsing (Tier-1/5). |
| Struct / cell arrays | struct-array fields, cell-of-char | `lib/Sema`, `runtime` | FASTA `s(i).Header`/`s(i).Sequence`, multi-record file reads, alignment result structs (Tier-1/3). |
| Dense linear algebra / reductions | `sum`/`min`/`max`/`cumsum`, `mldivide` | `runtime/matlab_runtime.cpp` | DP score-matrix fill + traceback (Tier-2/3); profile column statistics (Tier-3). |
| Distances / clustering | `pdist`, `squareform` | `lib/Sema/Resolver.cpp` (✅) | `seqpdist` Hamming/Jukes-Cantor distances; UPGMA/NJ tree building (Tier-4); microarray sample distances (Tier-6). |
| HMM engine | `hmmviterbi`, `hmmdecode`, `hmmtrain`, `hmmgenerate` | `lib/Sema/Resolver.cpp` (✅, Stats T6) | Profile-HMM align/score/train — position-specific emission/transition matrices run through the shipped Viterbi / forward-backward / Baum-Welch (Tier-3). |
| Stats / ML | `kmeans`, `pca`, `fitcsvm`, `fitcknn`, `fitctree`, `classify` | `runtime/toolbox/stats/runtime_stats.cpp` (✅) | `clustergram` clustering, `mapcaplot` PCA, `rankfeatures`/`randfeatures`/`classperf`/`crossvalind`/`knnimpute` (Tier-6). |
| Signal preprocessing | `resample`, `sgolayfilt`, `findpeaks`, smoothing | `runtime/matlab_runtime.cpp` (Signal, ✅) | `msresample`/`msgolay`/`mslowess`/`mspeaks`/`msbackadj` mass-spec preprocessing (Tier-6). |
| Plotting | Cairo `plot`/`imagesc`/`stem`/`bar`/`contourf` | `runtime/plot/` | `seqdotplot` (Tier-2), phylogram/cladogram (Tier-4), `clustergram`/`msheatmap`/`redgreencmap` (Tier-6), `ntdensity`/`proteinplot` (Tier-1/5). |
| Classdef plumbing | `matlab_obj_new`/`_set_*`/`_get_mat`, kwarg-ctor, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | The `phytree` object (Tier-4) and `DataMatrix` (Tier-6). |
| Name/value option parsing | option-string read in runtime (`fspecial`/`wdenoise` path) | `lib/MLIR/LowerTensorOps.cpp` | `nwalign(...,'ScoringMatrix','BLOSUM62','GapOpen',8)`, `wdenoise`-style kwargs across the toolbox. |

**Net assessment**: the *compute substrate* (char lane, regex, struct
arrays, DP-able reductions, the HMM engine, Stats/ML, Signal
preprocessing, plotting, classdef plumbing) is **already shipped**. The
genuinely new code is (a) the **biological alphabets + conversion + codon
tables** (`nt2aa`/`aa2int`/`geneticcode` — lookup tables), (b) the
**sequence statistics** (`aacount`/`basecount`/`codonbias`/`cpgisland`),
(c) the **FASTA/GenBank/PDB local-file parsers** (regex over text), (d)
the **DP alignment engine** (`nwalign`/`swalign` + the scoring-matrix
catalogue), (e) the **MSA + profile layer** (`multialign`/`seqprofile` +
profile-HMM wrappers over the shipped HMM engine), (f) the **`phytree`
classdef + distance builders** (`seqpdist`/`seqlinkage`/`seqneighjoin`),
(g) the **protein property tables** (`molweight`/`isoelectric`), and (h)
the **application-layer wrappers** (microarray normalization, mass-spec
preprocessing, learning helpers — thin shims over the shipped kernels).
Each is a self-contained hand-coded routine — none requires a new numeric
primitive or an external library.

---

## 2. Bio-Tier-1 — Sequence lane: representation, alphabets, statistics, I/O ✅

Goal: get biological sequences into and out of the system, convert
between alphabets, and compute the basic compositional statistics. The
foundation every later tier stands on.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `nt2int`/`int2nt`/`aa2int`/`int2aa` | Map nucleotide/amino-acid chars ↔ integer codes (the numeric lane the DP fills run on). Lookup tables. | char lane |
| 1.2 | `nt2aa`/`aa2nt` | Codon → amino-acid translation via the genetic-code table (`'Frame'`/`'AlternativeStartCodons'`/`'GeneticCode'` options); reverse-translate. | 1.1, codon table |
| 1.3 | `dna2rna`/`rna2dna`/`seqcomplement`/`seqrcomplement`/`seqreverse` | T↔U substitution; Watson-Crick complement / reverse-complement; reverse. Pure char transforms. | char lane |
| 1.4 | `aminolookup`/`baselookup`/`geneticcode`/`revgeneticcode` | Code-table lookups (1-letter↔3-letter↔full-name; standard + mitochondrial codes). Baked-in tables. | tables |
| 1.5 | `basecount`/`aacount`/`dimercount`/`codoncount`/`nmercount` | Composition counts → struct of per-symbol/per-pair/per-codon frequencies; optional bar chart. | reductions, `bar` |
| 1.6 | `ntdensity`/`codonbias`/`cpgisland`/`oligoprop`/`gccontent` | Sliding-window nucleotide density plot; codon-usage bias per amino acid; CpG-island detection (moving GC% + obs/exp ratio); oligo properties (Tm, GC, MW). | windowed reductions, `plot` |
| 1.7 | `seqwordcount`/`seqmatch`/`seq2regexp`/`palindromes` | Word/pattern counts; library string match; IUPAC ambiguity → regex; reverse-complement palindrome search. | `regexp` |
| 1.8 | `randseq`/`seqdisp` | Random sequence generator (uniform / weighted / from profile); formatted multi-row display. | RNG, char fmt |
| 1.9 | `fastaread`/`fastawrite` | Parse `>header\nSEQ` records → struct array `s.Header`/`s.Sequence`; write the same. The headline I/O. | `regexp`, struct array |
| 1.10 | `oligoprop`/`isoelectric` stub hooks | Property scaffolding shared with Tier-5. | — |

**Headline-within-tier**: read a bundled FASTA, translate an ORF —
`s = fastaread('gene.fa'); aa = nt2aa(s(1).Sequence); aacount(aa)`.

**Compile/Execute wiring**: new
`runtime/toolbox/bioinfo/runtime_bioinfo.cpp`; register the conversion +
statistics builtins in `Resolver.cpp`; the alphabet/codon/lookup tables
are static arrays in the runtime keyed by a caller string (the Image
`imread('f.png')` path); `fastaread` returns a struct array (existing
struct-array machinery); composition functions return either a struct or
a `double` count vector depending on `nargout`.

**REPL/JIT + Debug**: sequences are chars → already render in the REPL and
DAP variable panes; the struct-array FASTA result renders via the shipped
struct pretty-printer. No new debug surface.

---

## 3. Bio-Tier-2 — Pairwise alignment + scoring matrices (the headline) ✅

Goal: globally / locally align two sequences and score them — the single
most common Bioinformatics-Toolbox task. Closes the headline.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `blosum`/`pam`/`dayhoff`/`gonnet`/`nuc44` | Scoring-matrix catalogue → labeled 24×24 (AA) / 15×15 (NT) matrices from baked-in tables (`blosum(62)`, `pam(250)`, ...). | lookup tables |
| 2.2 | `nwalign` | Needleman-Wunsch global alignment: DP score-matrix fill with affine gap (`GapOpen`/`ExtendGap`) + traceback → `[score, alignment, start]`. | DP fill, 2.1 |
| 2.3 | `swalign` | Smith-Waterman local alignment: same DP with a 0-floor + max-cell traceback. | 2.2 |
| 2.4 | `seqdotplot` | Self/cross k-mer match matrix → `imagesc`/`spy` dot plot (window + threshold options). | `imagesc` |
| 2.5 | `showalignment`/`seqalignment` display | Pretty-print the aligned pair with match/mismatch markers + identity/similarity %. | char fmt |
| 2.6 | `seqpdist` (pairwise mode) | Pairwise distance between two sequences (`'Hamming'`/`'Jukes-Cantor'`/`'alignment-score'`) — bridges to Tier-4. | 2.2, `pdist` |
| 2.7 | `localalign`/`nt2aa`-aware scoring | Multiple local alignments above a score threshold; codon-aware scoring option. | 2.3 |

**Headline-within-tier (whole-roadmap tracer-bullet)**: `align_globin_pair.m`
— `fastaread` two globins → `nwalign(s1,s2,'ScoringMatrix','BLOSUM62')` →
report score + identity → `seqdotplot`.

**Compile/Execute wiring**: `nwalign`/`swalign` are multi-return builtins
(`[score, aln, start] = ...`) via the existing multi-output splitter; the
scoring matrix is resolved from the catalogue when a string is passed or
taken directly when a numeric matrix is passed (`'ScoringMatrix'`
name/value read in the runtime, the `fspecial`-option path); the DP fill
+ traceback is a plain matrix routine in `runtime_bioinfo.cpp`.

---

## 4. Bio-Tier-3 — Multiple-sequence alignment + profiles + profile HMMs ✅ (MSA core; profile-HMM deferred)

Goal: align and profile a *set* of sequences; score a sequence against a
family HMM. Reuses the shipped Stats HMM engine.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `multialign` | Progressive MSA: guide tree (from `seqpdist`+`seqlinkage`) → successive profile-profile alignment. | 2.2, 4.x guide tree |
| 3.2 | `profalign` | Align two profiles (or a sequence to a profile) with affine gaps. | DP fill |
| 3.3 | `seqprofile`/`seqconsensus` | Column residue-frequency profile (with/without gaps/ambiguous); majority/score consensus sequence. | reductions |
| 3.4 | `seqlogo` | Information-content (bits) sequence logo → stacked-letter bar plot. | `seqprofile`, `bar` |
| 3.5 | `multialignread`/`seqdisp` | Read ClustalW/GCG/MSF alignment files; formatted alignment display. | `regexp` |
| 3.6 | `hmmprofstruct`/`showhmmprof` | Build / inspect a profile-HMM struct (match/insert/delete emission + transition matrices). | struct |
| 3.7 | `hmmprofalign`/`hmmprofestimate` | Align a sequence to a profile HMM (Viterbi) / estimate profile parameters from an MSA (Baum-Welch). | Stats `hmmviterbi`/`hmmtrain` |
| 3.8 | `hmmprofgenerate`/`hmmprofmerge`/`pfamhmmread` | Sample from a profile; merge profile alignments; read a PFAM-HMM file. | Stats `hmmgenerate` |

**Headline-within-tier**: `multialign` a small protein family → `seqlogo`
of the conserved columns + `seqconsensus`.

**Compile/Execute wiring**: `multialign`/`profalign`/`seqprofile` are
matrix/cell-in → cell/matrix-out builtins; the profile HMM is a *struct*
(not a classdef) whose matrices are handed straight to the shipped
`hmmviterbi`/`hmmtrain`/`hmmgenerate` — the only new code is the
profile↔HMM-matrix translation + PFAM-file parser.

---

## 5. Bio-Tier-4 — Phylogenetic analysis (`phytree` + distance builders) ✅ (core)

Goal: build, manipulate and draw phylogenetic trees from sequence
distances. The first classdef-bearing tier.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | `seqpdist` (matrix mode) | All-pairs distance matrix over a sequence set (`'Jukes-Cantor'`/`'Kimura'`/`'p-distance'`/`'alignment-score'`); optional `pdist`-vector output. | 2.2/2.6, `pdist`/`squareform` |
| 5.2 | `dnds`/`dndsml` | Synonymous/nonsynonymous substitution-rate ratio between coding sequences (counting + ML methods). | codon table, 5.1 |
| 5.3 | `seqlinkage` | UPGMA / WPGMA / single / complete agglomerative tree from a distance matrix → `phytree`. **Hand-rolled** (Stats `linkage` is carved out — standard min-distance merge loop). | distance matrix |
| 5.4 | `seqneighjoin` | Neighbor-joining tree (Saitou-Nei) → `phytree`. | distance matrix |
| 5.5 | `phytree` (classdef) | Tree object: pointer/branch-length arrays + leaf names. Constructed by 5.3/5.4 or from `(pointers, distances, names)`. | classdef plumbing |
| 5.6 | `phytreeread`/`phytreewrite`/`getnewickstr` | Newick-format parse / write / serialize. | `regexp` |
| 5.7 | phytree methods | `get`/`getbyname`/`pdist`/`weights`/`select`/`subtree`/`prune`/`reroot`/`getcanonical`. | class-pinned dispatch |
| 5.8 | `plot(phytree)` | Phylogram / cladogram / radial tree rendering. | `runtime/plot/` |

**Headline-within-tier (Tier-4 tracer-bullet)**: `primate_phylotree.m` —
`seqpdist` over primate mtDNA → `seqlinkage` → `phytree` → `plot`
cladogram.

**Compile/Execute wiring**: `phytree` follows the
`tf`/`ss`/`LinearModel` alloc-then-populate + class-pinned-dispatch
pattern, auto-prepended via `bioinfo_classdefs.m`; the builders return a
`phytree` object; `plot` dispatches on the pinned class; the distance
matrix bridges through the shipped `pdist`/`squareform`.

**REPL/JIT + Debug**: the `phytree` object persists across REPL
statements and renders in the DAP variable pane via the shipped classdef
render path — mind the recurring **ReplMode `ws_get_mat` round-trip**
trap (a `phytree` produced in one REPL statement and consumed by `plot`
in the next must survive the workspace round-trip; see
[`jit_pipeline_divergence.md`](repl_jit_cross_unit_gap.md)).

---

## 6. Bio-Tier-5 — Protein property + structural analysis ✅ (properties + digestion; PDB deferred)

Goal: physico-chemical properties of proteins + in-silico digestion + PDB
structural plots.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `molweight`/`atomiccomp` | Molecular weight + atomic composition from residue mass tables. | AA mass table |
| 6.2 | `isoelectric` | Isoelectric point (pI) — bisection over net charge from pK tables; charge-vs-pH curve plot. | pK table |
| 6.3 | `aacount`/`aminolookup` | (shared with Tier-1) amino-acid statistics + code lookup. | Tier-1 |
| 6.4 | `cleave`/`rebasecuts`/`restrict` | In-silico protease cleavage (`'trypsin'` etc. + regex rules) / restriction-enzyme cut sites (REBASE) / fragment generation. | `regexp` |
| 6.5 | `proteinplot` | Per-residue property profile (hydrophobicity / charge / sliding window) → `plot`. | `plot`, property tables |
| 6.6 | `pdbread`/`getpdb`(local) | Parse a local PDB file → struct (atoms / chains / coordinates). `getpdb` web fetch is carved (stubbed). | `regexp`, struct |
| 6.7 | `pdbdistplot`/`ramachandran` | Cα-distance heatmap; φ/ψ Ramachandran scatter from PDB coordinates. | `imagesc`/`scatter` |

**Headline-within-tier**: peptide-mass fingerprint —
`cleave(prot,'trypsin')` → `molweight` per fragment → mass histogram.

**Compile/Execute wiring**: all matrix/struct-in → matrix/struct-out
builtins; the mass/pK/atomic tables are baked-in arrays keyed by residue;
`pdbread` returns a struct (existing struct machinery); the digestion
functions are regex over the sequence char.

---

## 7. Bio-Tier-6 — Application layer: microarray + mass spec + learning ✅ (core)

Goal: the high-reuse application surface — most of the compute delegates
to already-shipped Stats / Signal / Image kernels. The largest tier by
function count, smallest by new algorithm.

### 7a. Microarray
| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 7.1 | `DataMatrix` (classdef) | Labeled expression matrix (data + row/col names); `dmarrayfun`/`dmbsxfun` element-wise. | classdef plumbing |
| 7.2 | `quantilenorm`/`manorm`/`malowess` | Quantile / mean / lowess normalization across arrays. | sort, `sgolayfilt`-style smoothing |
| 7.3 | `geneentropyfilter`/`genelowvalfilter`/`generangefilter`/`genevarfilter` + `exprprofrange`/`exprprofvar` | Gene-expression filtering by entropy / value / range / variance. | reductions |
| 7.4 | `clustergram` | Hierarchical-clustering dendrogram + reordered heatmap. **UPGMA hand-rolled** (reuse 5.3) + shipped `imagesc`. | 5.3, `pdist`, `imagesc` |
| 7.5 | `redgreencmap`/`maimage`/`maboxplot`/`mairplot`/`malogplot`/`mapcaplot` | Microarray colormaps + spatial/box/intensity-ratio/loglog plots; PCA scatter (`pca`). | `runtime/plot/`, `pca` |

### 7b. Mass spectrometry
| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 7.6 | `msresample`/`msppresample` | Resample m/z signal to lower/uniform resolution. | Signal `resample` |
| 7.7 | `msbackadj`/`msnorm`/`mslowess`/`msgolay` | Baseline correction / normalization / lowess + Savitzky-Golay smoothing. | `sgolayfilt`, smoothing |
| 7.8 | `mspeaks`/`msdotplot`/`msheatmap`/`msalign` | Peak detection / dot plot / heatmap / spectrum alignment to reference masses. | `findpeaks`, `imagesc` |
| 7.9 | `jcampread`/`mzxmlread`/`mzxml2peaks`(local) | Parse local JCAMP-DX / mzXML spectra files. | `regexp`, struct |

### 7c. Statistical learning + local-file readers
| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 7.10 | `rankfeatures`/`randfeatures` | Feature ranking (t-test / entropy / Bhattacharyya) / random feature subset selection. | Stats reductions |
| 7.11 | `knnimpute`/`classperf`/`crossvalind` | KNN missing-value imputation / classifier-performance object / cross-validation index generation. | `pdist`, Stats |
| 7.12 | `genbankread`/`genpeptread`/`emblread`/`geosoftread`/`gprread`/`galread`/`sptread`/`imageneread` | Local-file parsers for the remaining sequence/expression formats. | `regexp`, struct |
| 7.13 | web-DB stubs | `getgenbank`/`getgenpept`/`getembl`/`getpdb`/`getgeodata`/`blastncbi`/`getblast`/E-utilities → clear "offline build, use the local-file reader" error. | — |

**Headline-within-tier (Tier-6 tracer-bullet)**: `yeast_clustergram.m` —
bundled yeast expression `DataMatrix` → `quantilenorm` →
`genevarfilter` → `clustergram` heatmap + dendrogram.

**Compile/Execute wiring**: `DataMatrix` is the second classdef
(alloc-then-populate); the normalization/filter/learning functions are
matrix-in → matrix-out shims that mostly call the shipped Stats/Signal
kernels; `clustergram` hand-rolls UPGMA (reusing the Tier-4 merge loop)
then renders via `imagesc` + a dendrogram overlay; the local-file readers
are regex parsers returning structs; the web-DB functions are explicit
stubs.

---

## 8. Phasing & effort summary

The roadmap groups the six tiers into **three shippable phases**, each
independently demoable and each advancing the toolbox badge story:

| Phase | Tiers | Theme | New algorithm | Effort | Closes |
|---|---|---|---|---|---|
| **A — Sequence core + alignment** | T1 + T2 | The 80% workflow: read sequences, convert, align, score | alphabets + codon tables, FASTA parser, DP alignment, scoring-matrix catalogue | **~3 wk** | `align_globin_pair.m` headline |
| **B — Comparative genomics** | T3 + T4 | MSA, profiles, profile HMMs, phylogenetic trees | progressive MSA, profile↔HMM bridge, `phytree` classdef, UPGMA/NJ builders | **~4 wk** | `primate_phylotree.m` |
| **C — Application layer** | T5 + T6 | Protein properties, microarray, mass spec, learning | property tables, in-silico digestion, `DataMatrix`, normalization/clustering shims, local-file parsers | **~4 wk** | `yeast_clustergram.m` + peptide-mass demo |

**Full toolbox ≈ 11 weeks.** **Phase A alone (~3 wk) is the recommended
first cut** — it is self-contained (no classdef, pure char/matrix lane),
closes the canonical sequence-alignment workflow, and unblocks the rest
(Phase B's guide tree depends on Phase A's `nwalign`+`seqpdist`).

**Per-tier dependency notes**:
- T2 depends on T1 (sequences must exist before they can be aligned).
- T3 + T4 depend on T2 (`multialign`'s guide tree and `seqpdist` ride
  `nwalign`); T4 introduces the first classdef (`phytree`).
- T5 is independent of T3/T4 (only needs the T1 sequence lane) — can ship
  in parallel.
- T6 reuses the most (Stats/Signal/Image) and introduces the second
  classdef (`DataMatrix`); the web-DB stubs are trivial.

---

## 9. Carve-outs (explicitly out of scope)

- **Interactive apps**: Sequence Alignment App, `seqalignviewer`,
  `phytreeviewer`/`view`, `msviewer`, Genomics Viewer app, **Biopipeline
  Designer** (the entire chapter-2 NGS pipeline-builder UI), Clustergram
  *window* interactivity (static heatmap only).
- **Live network / web databases**: `getgenbank`/`getgenpept`/`getembl`/
  `getpdb`/`getgeodata`/`gethmmprof`/`gethmmalignment`/`gethmmtree`/
  `blastncbi`/`getblast`/`blastread`/NCBI E-utilities — stubbed with a
  clear error; the *local-file* readers for the same formats are in scope.
- **High-throughput NGS / structural objects**: `BioRead`/`BioMap`/
  `BioIndexedFile`/`GFFAnnotation`/`GTFAnnotation`, BAM/SAM/FASTQ indexed
  access, `featurecount`, RNA-Seq differential expression, ChIP-Seq
  genome-wide analysis, `seqfilter`/SAM-flag filtering (chapter 2 is
  almost entirely carved — it depends on the apps + indexed-file I/O +
  network).
- **Microarray experiment objects**: `ExptData`/`MetaData`/`MIAME`/
  `ExpressionSet`/GEO Series, Illumina bead-summary, array-CGH copy-number
  + Bayesian-HMM CGH, attractor metagenes (`DataMatrix` core only).
- **BioPerl** function calls, Spreadsheet-Link Excel exchange.
- **Full 3-D molecular rendering** (`molviewer`-style) — `pdbdistplot`/
  `ramachandran` 2-D plots only; `proteinplot` GUI → function form only.
- **Genetic-algorithm mass-spec feature search** (chapter 6) — reuses
  GADS `ga` (shipped) but the demo wiring is deferred.
- **RNA secondary structure** (`rnafold`/`rnaconvert`/`rnaplot`) and
  **mass-spec parallel/batch** (chapter 6) — deferred to a follow-on.

---

## 10. Compiler traps to watch (from sibling-toolbox experience)

- **Scoring-matrix string vs numeric arg**: `nwalign(s1,s2,'BLOSUM62')`
  passes a string that must resolve to the catalogue; `nwalign(...,M)`
  passes a numeric matrix. Detect in the runtime by arg type (the
  `imread('f.png')` const-char-literal → `matlab_string_from_literal`
  path); do **not** route both through the same loadObj arm.
- **Multi-return splitter**: `[score,aln,start]=nwalign(...)` and
  `[coeff,score,latent]=...`-style — use the existing multi-output
  splitter; remember `numel` of a runtime-result is 0 and
  `~`-ignore-output is unsupported (Stats trap).
- **`fprintf` of a comparison / reduction result** doesn't lower — print
  identity % via `round(100*matches/len)` not `fprintf('%d', a==b)`
  (recurring Stats/Image trap); `%d` of a double prints 0 → use `%.0f`.
- **struct-array round-trip in REPL/JIT**: a `fastaread` struct array or
  `phytree` produced in one REPL statement and consumed in the next must
  survive the `ws_get_mat` workspace round-trip (the recurring
  ReplMode-defeats-AOT-detector root cause — see
  [`repl_jit_cross_unit_gap.md`](repl_jit_cross_unit_gap.md)).
- **classdef ctor-intercept scope**: `phytree(...)`/`DataMatrix(...)`
  constructors — `loadObj` is not in scope in the constructor-intercept
  arm; use `lowerExpr` for inline-matrix args (the Robotics `loadrobot`
  trap).
- **CMake build enforces `-Werror=old-style-cast`** (harness doesn't) —
  use `static_cast` throughout `runtime_bioinfo.cpp`; add the file to the
  strict-no-C-cast list (the Image `runtime_images.cpp` precedent).
- **Templates can't live in `extern "C"`** (Stats trap) — put any
  templated DP-fill helper at file scope before the `extern "C"` block.

---

## 11. Test & example surface (gating)

- **Gating tests** (`test/Run/bioinfo_*.m`), one per tier headline:
  `bioinfo_seqstats` (T1: `fastaread`+`nt2aa`+`aacount`),
  `bioinfo_nwalign` (T2: global align score + identity, fixed threshold),
  `bioinfo_multialign` (T3: consensus of a small family),
  `bioinfo_phytree` (T4: `seqpdist`→`seqlinkage`→Newick string),
  `bioinfo_protein` (T5: `molweight`+`cleave`),
  `bioinfo_microarray` (T6: `quantilenorm`+`genevarfilter` row count).
- **Examples** (`examples/bioinfo/`) mirroring the UG: `align_globin_pair.m`
  (headline), `primate_phylotree.m`, `yeast_clustergram.m`, plus
  `orf_translate.m`, `cpg_island_scan.m`, `peptide_mass_fingerprint.m`,
  `msspec_preprocess.m`.
- **Determinism**: alignment scores + identities are exact integers/ratios
  (DP is deterministic); phylogenetic distances + tree topologies are
  deterministic given a fixed distance metric; mass-spec peak counts use
  fixed thresholds. Any RNG (`randseq`, `crossvalind`, `kmeans` in
  `clustergram`) is seeded for reproducible verdicts (the Stats precedent).
- **Bundled data**: small FASTA/PDB/expression fixtures committed under
  `examples/bioinfo/data/` (no network) — the precedent of the bundled
  iris means in `iris_classify.m`.

---

## 12. One-line status for MEMORY.md (when shipped)

> Bioinformatics Toolbox — roadmap `docs/bioinformatics_toolbox_roadmap.md`
> (R2026a UG). High-reuse/low-keystone: sequences = char lane
> (`char-vector-234`); alignment = DP + baked scoring matrices; profile
> HMM reuses Stats `hmmviterbi`/`hmmtrain`; phylo reuses `pdist` +
> hand-rolled UPGMA; microarray/massspec/learning reuse
> `kmeans`/`pca`/`fitcsvm`/`resample`/`findpeaks`. 6 tiers / 3 phases:
> A=T1+T2 seq+align (~3wk, headline `align_globin_pair.m`), B=T3+T4
> MSA+phylo (~4wk, `phytree` classdef), C=T5+T6 protein+microarray+massspec
> (~4wk, `DataMatrix` classdef). ~11wk full. Carved: apps, web-DB (stubbed),
> NGS/BioMap/BAM, microarray experiment objects, BioPerl, 3-D molviewer.
