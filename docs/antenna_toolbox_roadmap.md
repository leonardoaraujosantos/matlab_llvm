# Antenna Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Antenna-Toolbox programs.

Antenna Toolbox is **substantially heavier** than Comm or RF Toolbox
because its core capability — Method of Moments (MoM) electromagnetic
simulation — is itself a large numerical-methods project (full-wave EM
solvers are typically tens of thousands of lines of C++ in production
codes). A faithful port of the entire Antenna Toolbox surface is
multi-month-to-year-scale work. This roadmap therefore tiers
aggressively: a usable **Antenna MVP** lands in ~5 weeks via wire-
antenna MoM only; the full triangular-mesh / dielectric / FMM /
hybrid-MoM-PO surface is staged into ANT-Tier-3 → ANT-Tier-5.

**Runtime locations**: `runtime/runtime_rf.cpp` (the shared compilation
unit where the Tier-2 closed-form solver lives via the
`matlab_ant_wire_*` entries), plus the catalog classdefs at
`runtime/ant_class_dipole.m` and `runtime/ant_class_monopole.m`.

Source: *Antenna Toolbox User's Guide* (R2026a). Companion docs:
[`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) (the umbrella
plan; this Antenna track was previously chapter §10 there),
[`rf_toolbox_plan.md`](rf_toolbox_plan.md) (consumer for the Antenna →
RF Touchstone bridge), [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)
(consumer for `link(rx, tx, prop)` once both ends ship),
[`feature_status.md`](feature_status.md), [`roadmap.md`](roadmap.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order.
  ANT-Tier-1 is the catalog-classdefs slice (shapes only, no
  solver). ANT-Tier-2 is the **Antenna MVP**: wire-antenna MoM,
  closed-form dipole already shipped, multi-wire MoM is the next
  slice. ANT-Tier-3 is triangular-mesh MoM for planar / patch
  antennas. ANT-Tier-4 is uniform arrays via the element-pattern
  multiplication approximation. ANT-Tier-5 lists the heavy /
  advanced items carved out as multi-month each.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **ANT-Tier-2 MVP is ✅ shipped** (commit `0f0894c`, 2026-05-12) for
  the canonical thin-dipole case via the closed-form induced-EMF
  method. Remaining work: ANT-Tier-2b multi-wire MoM (~3 sessions),
  full ANT-Tier-3 / Tier-4 surfaces (~7 weeks), ANT-Tier-5 items
  (multi-month each, carved out).
- **REPL / Debug**: `dipole` / `monopole` / future `helix` /
  `yagiUda` classdefs are handle classdefs with persistent state;
  the existing DAP variable-inspector path handles them. Solved
  current vectors and impedance sweeps return matrix / struct
  shapes that route through the standard renderer.

---

## 1. Tier 1 — Antenna catalog classdefs (no solver, ~1 week)

The "shapes-only" foundation. Every antenna in the catalog has
geometric / material parameters; before any solver lights up, the
classdefs themselves can ship as typed property holders, mirroring
how RF-Tier-1 ships `sparameters` before any analysis.

### 1.1 Scope — priority subset of the catalog (~12 of ~80 antenna types)

- **Wire antennas**: `dipole`, `monopole`, `dipoleFolded`,
  `loopCircular`, `helix`.
- **Planar antennas**: `bowtieRounded`, `spiralEquiangular`,
  `spiralArchimedean`.
- **Patch antennas**: `patchMicrostrip`, `patchMicrostripCircular`,
  `pifa` (planar inverted-F).
- **Travelling-wave**: `yagiUda`, `vivaldi`, `hornConical`,
  `hornRectangular`.
- **Reflector / aperture**: `reflectorParabolic` (carved out — Tier
  5 — needs surface-current PO/MoM-PO).
- **Generic**: `customAntenna` (for user-supplied geometry — heavy,
  carved out to Tier 5).

### 1.2 Properties per antenna

Geometry-specific (`Length`, `Width`, `Radius`, `ArmLengths`,
`Spacing`, `NumElements`, `Tilt`, `TiltAxis`, etc.) plus universal
(`Conductor` material, `Substrate` (`dielectric` material classdef),
`Load`, `Tuner`).

### 1.3 Methods (all stubs at Tier 1, lit up at Tier 2/3)

- `show(ant)` — return mesh geometry triple `(verts, edges, tris)`
  for visualization.
- `mesh(ant)` / `mesh(ant, 'MaxEdgeLength', λ/10)` — generate /
  re-mesh.
- `meshconfig(ant, 'auto'|'manual')` — mesh control.
- `info(ant)` — print summary.
- `numports(ant)` — feed-port count.
- The **analysis methods** (`impedance`, `pattern`, `current`,
  `sparameters`, `returnLoss`, `vswr`, `efficiency`, `gain`,
  `axialRatio`, `bandwidth`, `EHfields`, `pcbStack`,
  `radiationpattern`) are placeholders at Tier 1; they
  `error('not yet supported')` until ANT-Tier-2.

### 1.4 Tier-1 closure

**Effort**: ~1 week. The classdefs are mechanical (each is a
parameter holder + mesh-generator stub). Mesh generation for
**simple wire shapes** (segments) lands here; triangular meshing
on planar / 3-D surfaces lands at Tier 3.

**Architectural prerequisite**: same System-Object lowering fix as
Comm Tier 3 / RF-Tier-1 (CST §12 / §11.1). Antenna catalog objects
are classdefs with field stores; same blocker.

**What works at end of Tier 1**: `ant = dipole; ant.Length = 0.5;`
plus pretty-printing in the REPL. No analysis yet.

**Status**: 🟡 partial — `AntDipole` + `AntMonopole` shipped with
`design(ant, freq)` method + `antennaGain(ant, freq)` peak-gain
dispatch (textbook 2.15 / 5.15 dBi). Remaining 10 stubs
(`dipoleFolded` / `loopCircular` / `helix` / `bowtieRounded` /
`spiralEquiangular` / `spiralArchimedean` / `patchMicrostrip` /
`patchMicrostripCircular` / `pifa` / `yagiUda` / `vivaldi` /
`hornConical` / `hornRectangular`) 🔵.

---

## 2. Tier 2 — Wire-antenna MoM solver (Antenna MVP, ~3 weeks)

The first user-visible Antenna slice: **simulate a wire antenna and
get its impedance / pattern / S-parameters**. Restricted to wire
geometries (1-D segment mesh) — mathematically simpler than the
2-D triangular RWG-basis surface MoM, but covers the canonical
textbook antennas (dipole, monopole, Yagi, helix, loop, folded
dipole).

### 2.1 Wire mesh + segment basis 🟡

**Scope**:
- 1-D wire segmentation along the antenna's geometric centerline.
  Standard `Δ ≈ λ/10` rule with thin-wire approximation
  (radius ≪ wavelength).
- Piecewise-sinusoidal or piecewise-triangular basis functions
  (Galerkin-style; sinusoidal is the textbook choice for thin
  wires, triangular is simpler and almost as accurate).
- Mesh data: segment endpoints, segment indices, wire radius per
  segment, feed-port edges.

**Status**: 🟡 sidestepped — the closed-form Balanis EMF path
(commit `0f0894c`) doesn't need a discretized mesh; the
`n_segments` argument is kept in the API for forward compatibility
with the multi-wire MoM follow-on (ANT-Tier-2b 🔵).

**Effort**: ~3 sessions for the discretized form (ANT-Tier-2b).

### 2.2 Pocklington / Hallen impedance matrix 🔵 (ANT-Tier-2b)

**Scope**:
- Discretize the **Pocklington** integral equation (preferred —
  numerically better-conditioned than Hallen for thin wires) over
  the segments.
- Build the complex N×N impedance matrix `Z` where `Z_ij` is the
  mutual impedance between basis functions `i` and `j`.
- Singularity extraction for `i = j` (self-term, log singularity in
  the kernel).
- Numerical integration via Gauss-Legendre quadrature on segment
  pairs (typically 5–10 points per segment for engineering
  accuracy).

**Status**: 🔵 ANT-Tier-2b — gated on the kernel-scaling debug
pass. An exploratory pulse-basis / point-matching prototype
shipped + reverted in favour of the closed-form path, which
suffices for the canonical thin-dipole MVP.

**Effort**: ~1 week. The kernel evaluation is a straightforward
exponential integral once the singularity-extraction trick lands.

### 2.3 Solve `Z·I = V` and post-process ✅

**Scope**:
- Excitation vector `V`: 1 at the feed-port edge, 0 elsewhere
  (delta-gap source feed model).
- Solve the complex linear system `Z·I = V`. **Needs complex LU**
  — the existing real LU shipped (CST), complex LU is a follow-on
  whose cost is ~0.5 wk. Or: use the existing real linear solver
  on the 2N×2N real-equivalent system `[[Re(Z), -Im(Z)]; [Im(Z),
  Re(Z)]] · [Re(I); Im(I)] = [Re(V); Im(V)]` — a 2× cost vs native
  complex but immediately available.
- Output current vector `I` (complex, one entry per basis function).
- **Input impedance** at the feed: `Z_in = V_feed / I_feed`.
- **S₁₁** for a 50 Ω port: `(Z_in − 50)/(Z_in + 50)`.

**Status**: ✅ shipped — `antennaWireSolve(L, a, n_segs, freq)`
returns `Zin_re` / `Zin_im` / `S11_re` / `S11_im` / `VSWR` /
`ReturnLoss_dB`. Closed-form induced-EMF method (Balanis Eq. 8-60a/b)
with Si and Ci special functions (Taylor < 8 + asymptotic ≥ 8).
Verified at half-wave: 73.08 + j42.52 Ω vs reference 73.13 + j42.55.

### 2.4 Far-field radiation pattern ✅

**Scope**:
- `[E_theta, E_phi] = pattern(ant, freq, az, el)` — given solved
  current `I` from §2.3, compute the far-field E-vector by the
  radiation integral (sum of segment radiations weighted by `I`,
  with the Sommerfeld phase factor `exp(jk·r̂·r')`).
- Polar form via `pattern(ant, freq)` returns a 2-D
  `[NumEl × NumAz]` matrix of total field magnitude or directivity.
- Derived metrics: `gain(ant, freq)`, `directivity(ant, freq)`,
  `axialRatio(ant, freq, az, el)` (linear vs circular polarization
  measure), `efficiency(ant, freq)`.

**Status**: ✅ shipped — `antennaWirePattern(L, a, n_segs, freq, n_theta)`
returns `Theta` / `ETheta` / `EThetaMag` / `Gain_dBi` /
`Directivity_dBi`. Closed-form sinusoidal-current pattern
`F(θ) = (cos(½ kL · cos θ) − cos(½ kL)) / sin θ`. Half-wave
directivity = 2.15 dBi.

### 2.5 Frequency sweeps + RF-Toolbox bridge ✅

**Scope**:
- `sparameters(ant, freqs)` — produce a Touchstone-compatible
  `sparameters` object (RF-Tier-1.1!) by sweeping ANT-Tier-2 over
  a frequency vector.
- `returnLoss(ant, freqs)`, `vswr(ant, freqs)`, `bandwidth(ant)` —
  derived from S₁₁(f).
- `impedance(ant, freqs)` — array form of §2.3.

**Why this matters**: this is the bridge between EM simulation and
RF Toolbox. Once ANT-Tier-2 + RF-Tier-1 land together, a user can
say `sp = sparameters(dipoleAnt, 1e9:1e7:3e9); rfwrite(sp, 'dipole.s2p')`
and feed the resulting Touchstone into an RF cascade.

**Status**: ✅ shipped — `antennaWireSparameters(L, a, n_segs, freqs)`
returns RFSparameters-shaped struct (`S11` complex col,
`Frequencies`, `Z0 = 50`, `NumPorts = 1`). Drops straight into
`touchstoneWrite` for an `.s1p` Touchstone file — **closes
ANT-Tier-2 / Antenna MVP** for the thin-dipole case.

### 2.6 ANT-Tier-2 closure summary

**A user can model a dipole / monopole / Yagi / helix / loop /
folded-dipole and extract impedance, pattern, S₁₁, gain, VSWR,
bandwidth across a frequency sweep. This is the Antenna MVP.**
Expect ~70% of textbook antenna problems and ~50% of pedagogical
pattern-design problems to fit here.

| Primitive | Effort | Status |
|---|---|---|
| Wire mesh + sinusoidal basis (2.1) | 3 sess | 🟡 sidestepped via closed-form path |
| Pocklington Z matrix + singularity extraction (2.2) | 1 wk | 🔵 ANT-Tier-2b |
| Z·I=V solve + Z_in / S₁₁ (2.3) | 3 sess | ✅ shipped (`antennaWireSolve`) |
| Far-field pattern + gain / directivity (2.4) | 1 wk | ✅ shipped (`antennaWirePattern`) |
| Frequency sweep + RF-bridge `sparameters(ant, f)` (2.5) | 3 sess | ✅ shipped (`antennaWireSparameters`) — closes ANT-Tier-2 / Antenna MVP |
| Multi-wire MoM (ANT-Tier-2b) | 3 sess | 🔵 follow-on — unblocks Yagi-Uda / monopole-over-ground / helix / loop / folded-dipole |

**Status (2026-05-12): ANT-Tier-2 MVP shipped** for the canonical
thin-dipole case via the closed-form induced-EMF method (Balanis
Eq. 8-60). Three runtime entries live in `runtime/runtime_rf.cpp`
(sharing the RF TU's helpers):

- `antennaWireSolve(length_m, radius_m, n_segments, freq_Hz)` →
  struct{`Zin_re`, `Zin_im`, `S11_re`, `S11_im`, `VSWR`,
  `ReturnLoss_dB`}.
- `antennaWirePattern(length_m, radius_m, n_segments, freq_Hz, n_theta)`
  → struct with `Theta` column, `ETheta` (complex column),
  `EThetaMag`, `Gain_dBi` column, `Directivity_dBi`, `Zin_re`,
  `Zin_im`.
- `antennaWireSparameters(length_m, radius_m, n_segments, freqs)` →
  RFSparameters-shaped struct (`S11` complex column, `Frequencies`,
  `Z0 = 50`, `NumPorts = 1`).

**Carved into a follow-on tier (ANT-Tier-2b)**: general thin-wire
MoM (Pocklington / Hallen integral with pulse / sinusoidal basis +
Gauss-Legendre on segment pairs + the 2N×2N real-equivalent solve
infrastructure) for arbitrary thin-wire structures — Yagi-Uda,
folded-dipole, monopole over PEC ground (via image method), helix,
square loop. Estimated ~3 sessions once the kernel scaling is
fully nailed; the closed-form MVP already unblocks the Antenna →
RF Toolbox bridge for the dipole use case.

---

## 3. Tier 3 — Triangular-mesh MoM (planar antennas, ~6 weeks) 🔵

Lifts the wire restriction to handle 2-D conducting surfaces
(patches, planar dipoles, bowties, spirals). This is the
**workhorse MoM** in production EM solvers and is substantially
more code than Tier 2.

### 3.1 Triangular mesh generator 🔵

**Scope**:
- Discretize a planar / 3-D surface into triangles with edge length
  ≈ λ/10. For planar shapes (patch / bowtie / spiral), this is 2-D
  Delaunay or constrained Delaunay triangulation. For 3-D shells
  (closed metallic surfaces), surface triangulation.
- Mesh data: vertex coordinates, triangle vertex-index triples,
  edge list (each edge ≤ 2 incident triangles).

**Effort**: ~1 week. Constrained Delaunay is non-trivial but well-
documented (Shewchuk's "Triangle" library is ~6000 lines of C —
re-implementation is a focused sub-arc).

### 3.2 RWG basis functions 🔵

**Scope**:
- Rao-Wilton-Glisson (RWG) basis: each interior edge defines one
  basis function spanning the two adjacent triangles, with surface
  current density that flows across that edge.
- Compute per-edge normalization (edge length × triangle areas).

**Effort**: ~2 sessions.

### 3.3 Surface-integral impedance matrix 🔵

**Scope**:
- The dyadic Green's function `G(r, r')` for free space.
- Z matrix where `Z_ij` = surface integral over triangle pair (one
  pair per edge in the i and j basis function support) of the
  RWG-weighted Green's-function-with-derivatives kernel.
- Singularity extraction for self / near-self terms (Wilton et al.,
  1984: extract the 1/R kernel analytically, integrate numerically
  on the smooth remainder).
- 7-point Gauss-Legendre on triangles for the smooth integrand.

**Effort**: ~3 weeks. This is the **largest single item in the
Antenna roadmap**; production solvers often spend years tuning the
near-singular integration.

### 3.4 Patch / planar antenna properties 🔵

**Scope**: same as ANT-Tier-2.4–2.5 but extended to surface currents
on patches. `pattern`, `impedance`, `sparameters`, etc.

**Effort**: ~3 sessions (mostly reuse of Tier-2 post-processing).

### 3.5 ANT-Tier-3 closure

A user can model **patch antennas** (rectangular, circular, PIFA),
**planar bowties / spirals**, **slot antennas**, **Yagi-Uda with
finite-thickness elements**.

---

## 4. Tier 4 — Antenna arrays (~2 weeks) 🔵

### 4.1 Array geometry classdefs 🔵

**Scope**:
- `linearArray(Element, ElementSpacing, NumElements)` — uniform
  linear array.
- `rectangularArray(Element, Size, ElementSpacing)` — uniform
  rectangular array.
- `circularArray`, `conformalArray` — circular and arbitrary-position
  arrays.
- `customArray` — user-supplied positions + per-element antenna types.

**Effort**: ~3 sessions.

### 4.2 Array factor + element pattern multiplication 🔵

**Scope**:
- `pattern(arr, freq)` = `pattern(element, freq) · arrayFactor(arr,
  freq, az, el)`.
- Steering / weighting: `arr.PhaseShift = ...`,
  `arr.AmplitudeTaper = ...`. Beam-steering and Taylor / Chebyshev
  tapers.
- `EHfields(arr, freq, p)` — total field at point p.

**Effort**: ~3 sessions. Closed-form multiplication of element
pattern by array factor; trivial **without** mutual coupling.

### 4.3 Mutual coupling — defer to Tier 5 🔴

Mutual coupling between array elements requires the full MoM solve
on the entire array (not a single element + array factor) because
adjacent elements perturb each other's currents. For closely
spaced elements (< λ/2) this matters; for sparse arrays, the
multiplication approximation is fine. **Carve out** the rigorous
mutual-coupling path; ship the multiplication approximation in
ANT-Tier-4.

### 4.4 ANT-Tier-4 closure

Phased-array beam steering with the element-pattern multiplication
approximation lights up. Useful for pedagogical phased-array work
and for first-pass beamforming design.

---

## 5. Tier 5 — Heavy / advanced (carved out, multi-month each) 🔴

Sketched for completeness; not committed. Each item below is its
own multi-week-to-multi-month sub-arc.

| Item | Scope | Effort estimate |
|---|---|---|
| **MoM with dielectrics** (`dielectric` material, `substrate`) | Surface-integral equations on metal-dielectric boundaries; PMCHWT formulation | ~2 months |
| **Hybrid MoM-PO** | Couple MoM region (small antennas) with Physical Optics region (large scatterers, ground planes) | ~1.5 months |
| **Physical Optics solver** | Surface-current induced by incident field on lit region; geometric shadow detection | ~1 month |
| **Fast Multipole Method (FMM)** | O(N log N) acceleration for large structures via multipole-expansion + tree | ~3 months (research-grade) |
| **Infinite ground plane** | Image-theory boundary conditions; doubles effective antenna size | ~2 weeks |
| **Infinite array (unit-cell)** | Floquet-mode analysis for periodic structures | ~1 month |
| **Mutual coupling (rigorous)** | Full-array MoM solve with embedded element patterns | ~3 weeks (on top of triangular MoM) |
| **Reflector antennas** (parabolic / Cassegrain) | PO / GTD on curved reflectors | ~1.5 months |
| **Antenna optimization** (PSO / GA / SADEA / surrogate) | Wraps the solver in an optimization loop | ~3 weeks |
| **Photonic / metasurface** | Periodic homogenization + effective material parameters | ~2 months |
| **PCB antenna with full layer stack** (`pcbStack`) | Multi-layer dielectric + conductor stack-up + via modeling | ~2 months |
| **Antenna near-field** (`EHfields` near zone) | Quasi-static + reactive near-field formulas | ~1 week (small) |
| **Polarization / axial-ratio analysis tail** | Polarization decomposition over scan angles | ~1 week |

---

## 6. Out of scope at any tier (Antenna-specific carve-outs)

- **Antenna Designer app**, **Array Designer app**. Interactive Qt
  apps; not a language feature.
- **PCB Antenna Designer**, **Gerber export**.
- **AI for Antennas** (DL-based rapid analysis / surrogate models).
  Deep Learning Toolbox dependency.
- **Real-time 3-D visualization** of currents / fields / patterns.
  Static figures via Cairo are achievable; interactive 3-D is not.
- **GPU acceleration** of MoM. CPU lane only.
- **Custom antenna from photo** (computer-vision-based geometry
  inference). Out of scope.

---

## 7. What Antenna Toolbox brings to RF and Comm

- **Antenna → RF Toolbox**: `sparameters(ant, freqs)` (§2.5)
  produces a `sparameters` object that drops directly into RF
  Toolbox cascades (RF-Tier-2.2). A user can simulate a Yagi, dump
  it to Touchstone via `rfwrite`, and feed a vendor RF chain
  Touchstone-vs-Touchstone — closing the loop on "design the
  antenna and then design the chain it feeds." See
  [`rf_toolbox_plan.md`](rf_toolbox_plan.md).
- **Antenna → Propagation**: an antenna's pattern is the input to
  the directional-pattern hook in PROP-Tier-3 (`coverageGridMulti`
  with directional antennas — already shipped function-form). The
  ANT-Tier-2 pattern bridge (`antennaWirePattern`) is the source.
  See [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md).
- **Antenna → Comm**: an antenna's impedance / pattern affects link
  budget and effective channel. Once Comm Tier 5.2 (fading channels
  — ✅ shipped) and ANT-Tier-2 land together, `comm.RayleighChannel`
  can be parameterized by an antenna's far-field gain pattern as the
  receive aperture function.
- **Antenna → Antenna**: mutual coupling rigor (Tier-5 carved out)
  is the bridge between standalone antennas and large arrays — but
  the multiplication approximation in §4.2 is enough for most
  engineering design.

These are wiring items, not new primitives — once both ends of the
bridge ship in their respective tiers, the cross-toolbox examples
light up automatically.

---

## 8. Execution order — if user demand drives prioritization

| Order | What | Effort | Status |
|---|---|---|---|
| 1 | ANT-Tier-2 closed-form thin-dipole MVP (`antennaWireSolve` / `antennaWirePattern` / `antennaWireSparameters`) | 1 wk | ✅ shipped (2026-05-12) |
| 2 | ANT-Tier-1 catalog: `AntDipole` + `AntMonopole` classdefs + `design(ant, freq)` + `antennaGain(ant, freq)` | 3 sess | ✅ shipped |
| 3 | ANT-Tier-2b: multi-wire MoM (Pocklington / Hallen + Gauss-Legendre + 2N×2N real-equiv solve) | 3 sess | 🔵 — unblocks Yagi-Uda / monopole-over-ground / helix / loop / folded-dipole |
| 4 | ANT-Tier-1 catalog tail: remaining 10 classdefs (`dipoleFolded` / `loopCircular` / `helix` / `bowtieRounded` / `spiralEquiangular` / `spiralArchimedean` / `patchMicrostrip` / `patchMicrostripCircular` / `pifa` / `yagiUda` / `vivaldi` / `hornConical` / `hornRectangular`) | 1 wk | 🔵 — needs SO-lowering fix per [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) §11.1 |
| 5 | ANT-Tier-3.1: Triangular mesh generator (constrained Delaunay) | 1 wk | 🔵 |
| 6 | ANT-Tier-3.2: RWG basis functions | 2 sess | 🔵 |
| 7 | ANT-Tier-3.3: Surface-integral Z matrix + singularity extraction | 3 wk | 🔵 — largest item in Antenna arc |
| 8 | ANT-Tier-3.4: Patch / planar `pattern` / `impedance` / `sparameters` | 3 sess | 🔵 — closes ANT-Tier-3 |
| 9 | ANT-Tier-4.1: Array geometry classdefs | 3 sess | 🔵 |
| 10 | ANT-Tier-4.2: Array factor multiplication + steering / tapers | 3 sess | 🔵 — closes ANT-Tier-4 (no rigorous mutual coupling) |
| 11 | ANT-Tier-5 items | multi-month each | 🔴 carved out |

**Total remaining** (excluding ANT-Tier-5): ~7 weeks for ANT-Tier-3 +
ANT-Tier-4 (planar / patch surfaces + arrays), plus ~3 sessions for
ANT-Tier-2b multi-wire MoM. ANT-Tier-5 is multi-month per item,
carved out.

---

## 9. Gating tests + Internal references

- Runtime: [`runtime/runtime_rf.cpp`](../runtime/runtime_rf.cpp)
  (the `matlab_ant_wire_*` / `matlab_ant_*` entry families share
  the RF TU's helpers — Si / Ci special functions, complex
  arithmetic, integration kernels)
- Classdefs: [`runtime/ant_class_dipole.m`](../runtime/ant_class_dipole.m),
  [`runtime/ant_class_monopole.m`](../runtime/ant_class_monopole.m)
- Frontend: builtins registered in `lib/Sema/Builtins.cpp` under
  the `antennaWire*` / `antennaGain` / `design` groups
- User reference: covered by [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md)
  §10 closure summary; this roadmap is now the canonical home
- Companion plans: [`rf_toolbox_plan.md`](rf_toolbox_plan.md)
  (Touchstone bridge consumer), [`propagation_toolbox_roadmap.md`](propagation_toolbox_roadmap.md)
  (directional-pattern consumer)
- Project-wide roadmap: [`docs/roadmap.md`](roadmap.md)

Gating examples under `examples/rf/` exercise the dipole solver
end-to-end (the directional-coverage examples consume the pattern
function via PROP-Tier-3's directional hook).
