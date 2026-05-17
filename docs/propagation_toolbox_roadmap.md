# Propagation Models — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** RF / Comm / Antenna propagation-model
programs.

**Most of the propagation surface is function-form and has zero
classdef / System-Object dependency** — it ships in parallel with
everything else and reaches the canonical "point-to-point with terrain
+ Fresnel zone + Longley-Rice + numeric Coverage Map + Multi-Site
Directional" workflows without waiting for any of: Comm Tier 3+,
RF-Tier-1+, ANT-Tier-1+, or the CST §12 System-Object lowering fix.

**Runtime location**: [`runtime/runtime_prop.cpp`](../runtime/runtime_prop.cpp)
(~1700 LOC; all entries under the `matlab_prop_*` and `matlab_ant_wire_*`
prefixes) + classdef wrappers [`runtime/rf_class_propagationmodel.m`](../runtime/rf_class_propagationmodel.m),
[`runtime/rf_class_txsite.m`](../runtime/rf_class_txsite.m),
[`runtime/rf_class_rxsite.m`](../runtime/rf_class_rxsite.m).

Source: ITU-R P.525 / P.838 / P.676 / P.840, NTIA Report 82-100
(Hufford et al., ITM v7.0 NTIA reference), 3GPP TR 36.942 / 38.901,
plus the cellular empirical models (Hata / COST231 / Egli / ECC33 /
SUI / Ericsson). MATLAB MathWorks API at *Communications Toolbox →
Propagation Channels*. Companion docs:
[`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) (umbrella; this
roadmap was previously §3 there), [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md)
(pattern source for the directional hook), [`rf_toolbox_plan.md`](rf_toolbox_plan.md)
(link-budget composition), [`siteviewer.md`](siteviewer.md)
(carved-out 3-D rendering track), [`feature_status.md`](feature_status.md),
[`roadmap.md`](roadmap.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. The
  arc decomposes into five sub-tiers ordered by independence:
  Tier-1a (closed-form empirical models), Tier-2a (ITM / Longley-
  Rice), Tier-2b (single-TX PtP + Coverage Map), Tier-3 (directional
  + multi-site), Tier-1b (classdef wrappers).
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **PROP-Tier-1a / 2a / 2b / 3 are ✅ shipped** (~7 weeks of
  function-form work across multiple commits). The Tier-1b classdef
  wrappers also shipped (commit `f764dbd`, alongside the matrix-
  property-storage infra fix that sidestepped the originally-planned
  System-Object dependency).
- **REPL / Debug**: every propagation runtime entry returns
  `matlab_mat *` (vectors / matrices) or `matlab_struct *`
  (multi-field link-budget / coverage results); both go through
  the standard renderer. The `TxSite` / `RxSite` /
  `PropagationModel` CamelCase classdefs use the existing handle-
  class display path.

---

## 1. Sub-tier map

The arc decomposes into five sub-tiers ordered by independence:

| Sub-tier | Effort | Dependencies | What lights up |
|---|---|---|---|
| §2 PROP-Tier-1a — Function-form closed-form models | ~1.5 wk | none (`log10`/`sqrt`/`erfc`) | All empirical path-loss formulas (FSPL/Hata/COST231/Egli/ECC33/SUI/Ericsson + ITU-R rain/gas/fog/close-in), Fresnel zones, knife-edge diffraction, Haversine/Vincenty — ✅ **shipped** |
| §3 PROP-Tier-2a — ITM / Longley-Rice (function-form) | ~3 wk | PROP-Tier-1a + complex LU or 2N×2N real workaround | Terrain-aware path loss with reliability tuning — ✅ **shipped** (engineering port; v7.0 NTIA byte-identical reference port still 🔵) |
| §4 PROP-Tier-2b — Single-TX PtP + Coverage Map (function-form) | ~1 wk | PROP-Tier-2a | `los_check`, `link_budget`, `coverage_grid` numeric API. Single-TX, omnidirectional — ✅ **shipped** |
| §5 PROP-Tier-3 — Directional + multi-site coverage (function-form) | ~1.5 wk | PROP-Tier-2b | Sector / cosine / Gaussian / 3GPP analytical patterns, mount orientation, `coverage_grid_multi` with best-server / sum-power / SINR aggregation — ✅ **shipped**. Two-poles + sectors + directionals scenario lights up here. |
| §6 PROP-Tier-1b — `propagationModel` / `txsite` / `rxsite` classdef wrappers | ~3 sess | PROP-Tier-1a + System-Object fix (CST §12) | MathWorks-API-faithful `prop = propagationModel(...)` + `pathloss(prop, rx, tx)` syntax — ✅ **shipped** |

**Total**: ~7 weeks for the function-form quartet (§2 + §3 + §4 +
§5) — fully shipped. The classdef wrapper layer (§6) also shipped,
sidestepping the originally-planned SO dependency via the matrix-
property-storage infra fix in commit `f764dbd`.

---

## 2. PROP-Tier-1a — Function-form closed-form models (~1.5 weeks) ✅

### 2.1 ITU-R / NIST closed-form models ✅

**Scope** — bare functions:
- `L = fspl(d, freq)` — ITU-R P.525 Free Space
  `L = 32.45 + 20·log10(f_MHz) + 20·log10(d_km)` dB.
- `L = pathlossRain(d, freq, rainrate, polarization)` — ITU-R
  P.838 specific attenuation `γ_R = k·R^α` integrated over `d`.
  Frequency- and polarization-dependent `(k,α)` tables.
- `L = pathlossGas(d, freq, T, P, rho)` — ITU-R P.676 oxygen +
  water-vapor attenuation; layer model from §2 of standard.
- `L = pathlossFog(d, freq, M)` — ITU-R P.840 cloud / fog
  `γ_c = K_l·M`.
- `L = pathlossCloseIn(d, freq, n, sigma, d0)` — Close-In NIST /
  3GPP TR 38.901 reference-distance model.

**Why function-form first**: bare functions need no classdef, no
field stores, no monomorphization games. They lower into clean
single-function call IR on every emit lane and ship today against
the existing runtime substrate.

### 2.2 Cellular empirical extensions (non-MathWorks namespace) ✅

**Scope** — all closed-form, ~1 day each:
- `pathlossHata(f, ht, hr, d, env)` — Okumura-Hata, 150–1500 MHz.
  `env` ∈ `{'urban-large', 'urban-medium-small', 'suburban', 'open'}`.
- `pathlossCost231Hata(f, ht, hr, d, env)` — COST231 extension,
  1500–2000 MHz.
- `pathlossEgli(f, ht, hr, d)` — Egli VHF/UHF, 30–1000 MHz.
- `pathlossEcc33(f, ht, hr, d)` — ITU-R P.529, 700–3500 MHz.
- `pathlossSui(f, ht, hr, d, terrain)` — Stanford University
  Interim, 1900–11000 MHz. `terrain` ∈ `{'A','B','C'}`.
- `pathlossEricsson9999(f, ht, hr, d, env)` — 150–1900 MHz.

**Why ship these despite being MathWorks-incompatible**: every
cellular link-budget tutorial uses one of them; coverage-planning
services like cloud-RF expose exactly this list. A user porting a
script from one of those tools should not have to re-derive the
formulas.

### 2.3 Fresnel zone math ✅

- `r = fresnelZoneRadius(d1, d2, lambda, n)` — `n`-th Fresnel
  zone radius `r = sqrt(n·λ·d1·d2/(d1+d2))`.
- `clearance = fresnelClearance(profile, d1, d2, lambda, n)` —
  given a sampled terrain profile, returns the percentage Fresnel-
  zone clearance (0% = grazing, 60% = TIA-recommended minimum).

### 2.4 Knife-edge diffraction ✅

- `Ld = diffractionKnifeEdge(h, d1, d2, lambda)` — single-edge
  Fresnel-Kirchhoff loss as a function of the diffraction
  parameter `v = h·sqrt(2·(d1+d2)/(λ·d1·d2))`. Closed-form via
  Fresnel integrals `C(v)`, `S(v)`.
- `Ld = diffractionBullington(profile, lambda)` — multi-obstacle
  via Bullington's method (single equivalent edge).
- `Ld = diffractionDeygout(profile, lambda)` — Deygout's
  recursive multi-edge method (more accurate than Bullington for
  closely-spaced obstacles).

### 2.5 Geographic helpers ✅

- `[d, az] = haversine(lat1, lon1, lat2, lon2)` — great-circle
  distance + initial bearing. Earth radius = 6371 km.
- `[d, az1, az2] = vincenty(lat1, lon1, lat2, lon2, a, f)` —
  ellipsoidal distance + bearings (WGS-84 by default). Iterative,
  converges in 5–10 iterations.
- `[lat2, lon2] = greatCircleDestination(lat1, lon1, d, az)` —
  destination point given start + distance + bearing.

---

## 3. PROP-Tier-2a — ITM (Longley-Rice) function-form (~3 weeks) ✅

### 3.1 ITM v7 core port ✅

**Scope**:
- `[L, info] = itm_pathloss(profile, freq, ht, hr, polarization,
  climate, surface_refractivity, ground_conductivity,
  ground_permittivity, time_var, location_var, situation_var)` —
  bare function, no classdef.
- `profile` is a real vector of terrain heights along the great-
  circle path (provided by §4.1 `terrainProfile` or by the user
  directly).
- `polarization` ∈ `{'horizontal','vertical'}`. `climate` ∈ 7
  named values. Reliability triple `(time_var, location_var,
  situation_var)` defaults to `(50, 50, 50)` for long-term
  median; setting `(80, 99, 99)` produces TSB-10F-compliant
  microwave-link results.
- Frequency range 20 MHz – 20 GHz (per the standard).
- Returns scalar median `L` plus an `info` struct with
  area-vs-point-to-point mode, message-success quantile, etc.

**Algorithm**: engineering port of the NTIA ITM v7.0 reference C++
source (public-domain). Three internal phases: preliminary (path
geometry + average terrain slope), area-mode (when no terrain
profile), and point-to-point (when profile provided). Tracks `m_d`
(median path loss) and `Z` quantile statistics.

**Status**: ✅ shipped (engineering port). Byte-identical NTIA
v7.0 reference port stays 🔵 as a precision follow-on; the
engineering port is good to ~0.1 dB on the NTIA reference test
suite.

---

## 4. PROP-Tier-2b — PtP + Coverage Map (function-form, ~1 week) ✅

### 4.1 Terrain profile from a heightmap ✅

- `profile = terrainProfile(heightmap, latlon_grid, lat1, lon1,
  lat2, lon2, num_samples)` — given a 2-D `heightmap` matrix
  spanning a `latlon_grid`, sample elevation along the great-
  circle path.
- Bilinear interpolation between heightmap cells.
- The user supplies the heightmap and grid; **no SRTM/DTED
  auto-fetch** (carved out — see §8).

**Why this design**: keeps the runtime hermetic. Users wanting
SRTM auto-fetch can do it in their own MATLAB script (e.g., via
`websave` + a tile-server URL) and pass the resulting matrix in.

### 4.2 Line-of-sight check ✅

- `[isClear, obstructionPoint] = los_check(tx_lat, tx_lon, tx_height,
  rx_lat, rx_lon, rx_height, profile)` — geometric LOS check
  accounting for terrain elevation *and* effective Earth radius
  (4/3 factor for standard atmosphere).
- Returns boolean + index of highest obstruction along the path.

### 4.3 Point-to-point link budget ✅

- `result = link_budget(tx_lat, tx_lon, tx_height, tx_freq, tx_power,
  rx_lat, rx_lon, rx_height, prop_model_name, profile, ...)` —
  function-form PtP analysis.
- Returns a struct: `PathLoss`, `ReceivedPower`, `Snr`,
  `LinkMargin`, `FresnelClearance`, `LosClear`, `Profile`,
  `Distance`, `Azimuth`.
- `prop_model_name` selects the underlying §2 / §3 entry
  (`'fspl'`, `'hata'`, `'cost231'`, `'longley-rice'`, …).
- Combines path loss + diffraction + atmospheric attenuation per
  the chosen model.

### 4.4 Coverage map (numeric) ✅

- `[grid, lat_grid, lon_grid] = coverage_grid(tx_lat, tx_lon,
  tx_height, tx_freq, tx_power, prop_model_name, heightmap,
  latlon_grid, ...)` — grid of received signal strength (dBm) on
  a square or rectangular lat/lon mesh centered on the transmitter.
- Each cell evaluates `link_budget(...)` independently → matrix
  output.
- Default: 100×100 cells, 10 km radius (configurable via
  `'Resolution'`, `'MaxRange'`).
- **Numeric form only** — returns a matrix. Plotting via the
  shipped Cairo backend (`imagesc(grid)`) produces a static
  heatmap PNG; users add their own colorbar/legend. Interactive
  Site Viewer is **carved out** ([`siteviewer.md`](siteviewer.md)).

### 4.5 End-to-end PtP + Coverage workflow ✅

```matlab
% Example 1: Point-to-point with terrain
heightmap = load('mySrtmTile.mat').heights;   % user-supplied DEM
gridDef   = struct('LatMin', 37.4, 'LatMax', 37.7, ...
                   'LonMin', -122.4, 'LonMax', -122.0, ...
                   'NumLat', 360, 'NumLon', 480);

profile = terrainProfile(heightmap, gridDef, ...
                         37.5, -122.3, 37.6, -122.0, 200);

result = link_budget(37.5, -122.3, 30, 5.8e9, 0.1, ...
                     37.6, -122.0, 5, ...
                     'longley-rice', profile, ...
                     'TimeVariability', 80, ...
                     'SituationVariability', 99);

disp(result.PathLoss);          % dB
disp(result.FresnelClearance);  % %
disp(result.LosClear);          % bool

% Example 2: Coverage map
[grid, lats, lons] = coverage_grid(37.5, -122.3, 30, 5.8e9, 0.1, ...
                                    'longley-rice', ...
                                    heightmap, gridDef, ...
                                    'Resolution', 100, ...
                                    'MaxRange', 20e3);
```

**Both examples work with bare functions — no classdef, no
System-Object machinery, no architectural blockers.**

---

## 5. PROP-Tier-3 — Directional + multi-site coverage (function-form, ~1.5 weeks) ✅

PROP-Tier-2b §4.4 ships **single-TX, omnidirectional** `coverage_grid`.
Real WISP / cellular / point-to-multipoint planning needs multiple
sites, multiple antennas per site, and directional patterns (sector
/ pencil-beam). PROP-Tier-3 layers that on top.

### 5.1 Sector / directional antenna pattern functions ✅

Closed-form analytical patterns. **No MoM dependency** — these are
textbook gain functions that take a `(az, el)` query and return a
gain in dBi.

- `G = sectorPattern(az, el, beamwidth_az_deg, beamwidth_el_deg, peak_gain_dBi, frontBackRatio_dB)`
  — 3GPP TR 36.942 sector pattern.
- `G = sectorPattern3GPP(az, el, beamwidth_az_deg, slld_dB, peak_gain_dBi)`
  — explicit 3GPP form `Az(φ) = -min(12·(φ/φ₃dB)², slld)`.
- `G = cosinePattern(az, el, halfBW_az, halfBW_el, peak_gain_dBi, n)`
  — cosine-power pattern `cos^n(θ)`.
- `G = gaussianPattern(az, el, halfBW_az, halfBW_el, peak_gain_dBi)`
  — Gaussian `G = G_peak·exp(-2.77·(az/halfBW)²)`.
- `G = isotropicPattern(...)` — flat 0 dBi. Reference / baseline.
- `G = customPattern(az_grid, el_grid, gain_matrix, az, el)` —
  bilinear-interpolate a user-supplied gain matrix at queried
  `(az, el)`. **Bridge to ANT-Tier-2**: a Yagi simulated by the
  MoM solver produces exactly such a matrix.

### 5.2 Antenna mount + orientation ✅

A "mount" describes a physical antenna pointing direction at a
TX/RX site. Without a mount, an antenna pattern is in the antenna's
local frame; with one, it's in the world frame.

```matlab
mount = struct('Azimuth', 120, ...        % degrees from North (0–360)
               'MechanicalTilt', 0, ...   % degrees from horizontal (+ = up)
               'ElectricalTilt', 5, ...   % electrical down-tilt (cellular)
               'Roll', 0);                % polarization tilt (rare)

% Apply orientation: input pattern is in antenna's local frame,
% output gain is what an observer sees from the mount's world frame.
G_world = applyMountOrientation(patternFunc, mount, az_world, el_world);
```

Coordinate convention: world az is from North clockwise (compass
bearing), el is positive above horizontal. Antenna local az/el is
relative to boresight. Mount applies a 3-axis rotation
(yaw=Azimuth, pitch=Tilt, roll=Roll).

### 5.3 Multi-TX coverage with directional antennas ✅

The function that lights up the "two-poles + sectors + directionals"
scenario.

```matlab
% Define each site as a struct
site1 = struct( ...
  'Lat', 37.5, 'Lon', -122.3, 'Height', 30, ...
  'Power_W', 10, 'Freq_Hz', 2.4e9, ...
  'Antennas', { ...
    % Three 120° sectors
    struct('Pattern', @(az,el) sectorPattern(az, el, 120, 10, 17), ...
           'Mount', struct('Azimuth',   0, 'ElectricalTilt', 5)), ...
    struct('Pattern', @(az,el) sectorPattern(az, el, 120, 10, 17), ...
           'Mount', struct('Azimuth', 120, 'ElectricalTilt', 5)), ...
    struct('Pattern', @(az,el) sectorPattern(az, el, 120, 10, 17), ...
           'Mount', struct('Azimuth', 240, 'ElectricalTilt', 5)), ...
    % Two directional links to other poles
    struct('Pattern', @(az,el) cosinePattern(az, el, 8, 8, 22, 30), ...
           'Mount', struct('Azimuth',  60, 'MechanicalTilt', 0)), ...
    struct('Pattern', @(az,el) cosinePattern(az, el, 8, 8, 22, 30), ...
           'Mount', struct('Azimuth', 200, 'MechanicalTilt', 0)) });

site2 = struct(...);   % the second pole, same shape

% Combined coverage
[grid, info] = coverage_grid_multi({site1, site2}, ...
                                    'longley-rice', heightmap, gridDef, ...
                                    'Aggregation', 'best-server', ...
                                    'Resolution', 200, ...
                                    'MaxRange', 20e3);
```

**Aggregation modes**:

| Mode | What it returns | Use case |
|---|---|---|
| `'best-server'` (default) | For each pixel, `max(P_rx_i)` over all (site, antenna) pairs `i`; `info.ServerIndex(p)` records which one | Coverage maps, "which sector serves where" |
| `'sum-power'` | `Σ P_rx_i` (incoherent power sum) | Conservative coverage estimate when antennas overlap |
| `'sinr'` | `max(P) / (Σ_others P + N₀·B)` per pixel; `info.NoiseFloor` records `N₀·B` | Cellular-style SINR maps; needs `Bandwidth_Hz` per site |
| `'rsrp'` | RSRP-style averaging over a configured set of resource elements (cellular-only convenience) | LTE / NR planning |

**Output `info` struct** (besides the grid):
- `ServerIndex` — `[NumLat × NumLon]` matrix of `(site, antenna)`
  index of the strongest server.
- `LinkLossDB` — strongest-server path loss matrix.
- `Azimuth` / `Elevation` — `[NumLat × NumLon]` of arrival angles
  per pixel from the strongest server.
- `Polygons` — per-server coverage polygon (integer mask matrices,
  one per server).

**Cost**: ~`N_pixels · N_sites · N_antennas` per-link evaluations.
For 200×200 pixels × 2 sites × 5 antennas = 400,000 link
evaluations. Each is ~milliseconds for `longley-rice`; expect runs
of ~minutes on serial CPU. Embarrassingly parallel — `parfor`
opportunity.

### 5.4 RX-side directional antennas ✅

Symmetric: `coverage_grid_multi` accepts an optional `RxAntenna`
parameter — a function handle returning RX gain at the angle of
arrival from each TX. Defaults to isotropic 0 dBi.

- `'RxAntennaPattern'` (function or matrix) — applies to all RX
  pixels in the grid.
- `'RxMount'` — orientation applied to the RX antenna pattern.
  For coverage maps that target a roving mobile, `'RxAzimuth' =
  'face-tx'` re-orients per-pixel toward the strongest TX; for
  fixed-mount RX (microwave links), provide an explicit Azimuth.

### 5.5 Bridge to Antenna Toolbox (ANT-Tier-2) 🔵

The `Pattern` field of an antenna mount can be:

| Source | Type | Effort to integrate |
|---|---|---|
| Analytical (`sectorPattern`, `cosinePattern`, etc.) | Function handle `(az, el) → dBi` | 0 — already supported in §5.1 |
| User-supplied gain matrix | `customPattern(az_grid, el_grid, M, az, el)` | 0 — already supported |
| **Antenna Toolbox simulated pattern** (ANT-Tier-2) | Output of `pattern(yagiUda, freq)` is a 2-D matrix of dBi values over `(az, el)` — wrap with `customPattern` | 1 session glue once ANT-Tier-2 ships |
| **MathWorks-API antenna handle** | `customPattern(antennaObj, freq, az, el)` thin wrapper that internally calls `pattern(antennaObj, freq, az, el)` | 1 session, gated on ANT-Tier-1 classdefs |

So the user can prototype with analytical sectors today, then drop
in measured Yagi patterns verbatim once
[`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md)
ANT-Tier-2b ships — same `coverage_grid_multi` call, just swap the
`Pattern` field.

---

## 6. PROP-Tier-1b — MathWorks-API classdef wrappers (~3 sessions) ✅

The function-form surface in §2–§5 is wrapped in MathWorks-
faithful classdefs:

- `prop = propagationModel('freespace'/'rain'/'gas'/'fog'/'close-in'/'longley-rice')`
  — constructor returning a value classdef. Internally delegates
  to the §2 / §3 functions.
- `tx = txsite('Latitude', ..., 'Longitude', ..., 'AntennaHeight',
  ..., 'TransmitterFrequency', ..., 'TransmitterPower', ...,
  'Antenna', ...)` — value classdef holding TX site parameters.
- `rx = rxsite(...)` — value classdef for RX site.
- `L = pathloss(prop, rx, tx)` — replaces the function-form
  `link_budget(...)`. Same numbers, MathWorks-faithful syntax.
- `[grid, lats, lons] = coverage(tx, prop, ...)` — replaces
  `coverage_grid(...)`. Accepts a `tx` array (vector of `txsite`)
  and dispatches to §5.3 `coverage_grid_multi` for the multi-
  site case.
- `[isClear, ...] = los(tx, rx)` — replaces `los_check(...)`.
- `result = link(rx, tx, prop, ...)` — replaces `link_budget(...)`.

**Status**: ✅ shipped — `TxSite` / `RxSite` / `PropagationModel`
CamelCase classdefs with kwarg ctor sugar live at
[`runtime/rf_class_txsite.m`](../runtime/rf_class_txsite.m),
[`runtime/rf_class_rxsite.m`](../runtime/rf_class_rxsite.m),
[`runtime/rf_class_propagationmodel.m`](../runtime/rf_class_propagationmodel.m).
All methods dispatch through the §2–§5 function-form runtime.

The originally planned System-Object dependency was sidestepped
via the matrix-property-storage infra fix in commit `f764dbd`.

---

## 7. PROP closure summary

| Primitive | Effort | Status | SO-fix dependency |
|---|---|---|---|
| Closed-form ITU-R / NIST models (5) (§2.1) | 4 sess | ✅ shipped (`fspl`, `pathlossRain`, `pathlossGas`, `pathlossFog`, `pathlossCloseIn`) | none |
| Cellular empirical models (6) (§2.2) | 3 sess | ✅ shipped (`pathlossHata`, `pathlossCost231`, `pathlossEgli`, `pathlossEcc33`, `pathlossSui`, `pathlossEricsson9999`) | none |
| Fresnel zone math (§2.3) | 3 sess | ✅ shipped (`fresnelZoneRadius`, `fresnelClearance`) | none |
| Knife-edge diffraction (single + multi-edge) (§2.4) | 1 wk | ✅ shipped (`diffractionKnifeEdge`, `diffractionBullington`, `diffractionDeygout`) | none |
| Haversine / Vincenty / great-circle (§2.5) | 2 sess | ✅ shipped (`haversine`, `bearing`, `vincenty`, `greatCircleDestLat`/`Lon`) — closes PROP-Tier-1a | none |
| ITM (Longley-Rice) v7 port (§3.1) | 3 wk | ✅ shipped (engineering port: `itmPathloss` with reliability quantile correction) — closes PROP-Tier-2a. Byte-identical NTIA v7.0 reference port stays 🔵. | none |
| Terrain profile from heightmap (§4.1) | 3 sess | ✅ shipped (`terrainProfile`) | none |
| `los_check` (§4.2) | 1 sess | ✅ shipped (`losObstruction`, `losClear`) | none |
| `link_budget` PtP (§4.3) | 3 sess | ✅ shipped (`linkBudget` → struct of TX dBm / RX dBm / FSPL / margin) | none |
| `coverage_grid` single-TX (§4.4) | 3 sess | ✅ shipped (`coverageGrid` → matrix) — closes PROP-Tier-2b | none |
| Sector / cosine / Gaussian / 3GPP / custom pattern functions (§5.1) | 3 sess | ✅ shipped (`sectorPattern`, `cosinePattern`, `gaussianPattern`, `isotropicPattern`) | none |
| `applyMountOrientation` (§5.2) | 2 sess | ✅ shipped (`applyMountAz`/`applyMountEl`/`applyMountOrientation`) | none |
| `coverage_grid_multi` with best-server / sum-power / SINR (§5.3) | 1 wk | ✅ shipped (`coverageGridMulti` with aggregation modes) — closes PROP-Tier-3 | none |
| RX-side directional (§5.4) | 2 sess | ✅ shipped | none |
| ANT-Tier-2 pattern bridge (§5.5) | 1 sess | 🔵 — gated on ANT-Tier-2b multi-wire MoM (closed-form dipole MVP shipped) | only on ANT-Tier-2b shipping |
| `propagationModel` / `txsite` / `rxsite` / `pathloss` / `coverage` / `los` / `link` classdef wrappers (§6) | 3 sess | ✅ shipped (`TxSite` / `RxSite` / `PropagationModel` CamelCase classdefs with kwarg ctor sugar) | — |

**Status (2026-05-17)**: **PROP-Tier-1a + 2a + 2b + 3 + 1b are all
shipped**. The function-form runtime lives in
`runtime/runtime_prop.cpp` (~1700 lines) and the classdef wrappers
ship with kwarg-sugar constructors. Only the ANT-Tier-2 pattern
bridge (§5.5) remains 🔵 — gated on the planned
[ANT-Tier-2b](antenna_toolbox_roadmap.md) multi-wire MoM.

---

## 8. Out of scope (Propagation-specific carve-outs)

- **Site Viewer** (3-D interactive map of buildings / terrain /
  ray traces, with Cesium / OSM / DTED-tile rendering). Hard 🔴 —
  needs Mapping Toolbox + 3-D graphics stack. See
  [`siteviewer.md`](siteviewer.md) for the carved-out 3-D
  rendering track.
- **Ray tracing through 3-D buildings** (`propagationModel('raytracing')`,
  `raytrace(tx, rx, scenario)`). Needs OSM buildings + ray-vs-
  triangle intersection + multi-bounce reflection model.
- **Auto-fetch SRTM / DTED / OpenStreetMap tiles**. We accept
  user-supplied heightmap matrices (§4.1); auto-download from
  web tile servers is out of scope. Users can fetch tiles in
  their own scripts and pass the matrix in.
- **TIREM** (`propagationModel('tirem')`) — proprietary US DoD
  propagation library; external license dependency.
- **MSI Planet file format** (interchange with commercial RF
  planning tools).
- **GPU acceleration** of ray tracing or coverage-map evaluation.
  CPU lane only; coverage-map grid evaluation is embarrassingly
  parallel and could use `parfor` later.
- **Multi-floor / building-aware indoor propagation**. Same
  scenario / 3-D geometry stack as ray tracing; defer.
- **Real-time animated coverage map** as TX moves. Static numeric
  matrices + Cairo PNG snapshots are in scope; live animation is
  not.

---

## 9. What Propagation brings to the rest of the roadmap

- **PROP → Comm**: link-budget realism. `link_budget` lets a Comm
  BER simulation be parameterized by a physical path-loss + thermal-
  noise floor instead of an abstract SNR. `awgn(x, snr)` (Comm Tier
  1.5) accepts the noise floor that `link_budget` predicts.
- **PROP → RF Toolbox**: `link_budget` and `rfbudget` (RF-Tier-2.3)
  compose — RF chain budget gives noise figure / IP3 from circuit;
  Propagation gives path loss from geometry; together they answer
  "how much margin does the link have."
- **PROP → Antenna Toolbox**: ANT-Tier-2 (`antennaWirePattern`)
  produces gain patterns; Propagation §5.3 / §5.4 consumes them at
  TX/RX endpoints. The closed-form dipole MVP ships today; the
  multi-wire MoM (ANT-Tier-2b) lights up Yagi / monopole-over-
  ground / helix patterns through the same `Pattern` field hook.
  See [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md).

These are **wiring opportunities** — no new primitives are implied;
once both sides of each bridge are shipped, the cross-toolbox
examples light up automatically.

---

## 10. Execution order — for completeness

| Order | What | Effort | Status |
|---|---|---|---|
| 1 | PROP-Tier-1a: closed-form ITU-R + cellular + Fresnel + knife-edge + geo helpers (§2) | 1.5 wk | ✅ shipped |
| 2 | PROP-Tier-2a: ITM (Longley-Rice) engineering port (§3) | 3 wk | ✅ shipped |
| 3 | PROP-Tier-2b: terrain profile + `los_check` + `link_budget` + `coverage_grid` (§4) | 1 wk | ✅ shipped — closes single-TX MVP |
| 4 | PROP-Tier-3: directional patterns + mount orientation + `coverage_grid_multi` (§5) | 1.5 wk | ✅ shipped — closes Multi-Site Directional MVP |
| 5 | PROP-Tier-1b: classdef wrappers `propagationModel` / `txsite` / `rxsite` (§6) | 3 sess | ✅ shipped |
| 6 | ANT-Tier-2b multi-wire MoM pattern bridge (§5.5) | 1 sess | 🔵 — gated on [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md) Tier-2b |
| 7 | Byte-identical NTIA ITM v7.0 reference port | ~2 wk | 🔵 — engineering port suffices for most users |

**Total remaining**: ~2.5 weeks (NTIA-grade ITM precision port +
ANT-Tier-2b pattern bridge wiring). Everything user-visible ships.

---

## 11. Gating tests + internal references

- Runtime: [`runtime/runtime_prop.cpp`](../runtime/runtime_prop.cpp)
  (~1700 LOC; all entries under the `matlab_prop_*` prefix)
- Classdefs: [`runtime/rf_class_propagationmodel.m`](../runtime/rf_class_propagationmodel.m),
  [`runtime/rf_class_txsite.m`](../runtime/rf_class_txsite.m),
  [`runtime/rf_class_rxsite.m`](../runtime/rf_class_rxsite.m)
- Frontend: builtins registered in `lib/Sema/Builtins.cpp` under
  the `pathloss*` / `coverage*` / `terrainProfile` / `los*` /
  `linkBudget` / `sectorPattern` / `cosinePattern` /
  `gaussianPattern` / `isotropicPattern` / `applyMount*` /
  `haversine` / `vincenty` / `bearing` / `greatCircleDest*` /
  `fresnelZone*` / `diffraction*` / `itmPathloss` groups
- Example workflows: `examples/rf/coverage_barbados.m` (PtP + ITM +
  coverage map on a synthetic Mount-Hillaby DEM with two 22 dBi
  cosine dishes), `examples/rf/pathloss_models.m`,
  `examples/rf/fresnel_diffraction.m`, `examples/rf/antenna_patterns.m`,
  `examples/rf/longley_rice_link.m`, `examples/rf/geo_helpers.m`,
  `examples/rf/coverage_three_sector.m`
- Companion plans: [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md)
  (umbrella; this roadmap was previously §3 there),
  [`antenna_toolbox_roadmap.md`](antenna_toolbox_roadmap.md)
  (pattern source for §5.5 bridge), [`rf_toolbox_plan.md`](rf_toolbox_plan.md)
  (link-budget composition), [`siteviewer.md`](siteviewer.md)
  (carved-out 3-D rendering)
- Project-wide roadmap: [`roadmap.md`](roadmap.md)
- Authoritative compat matrix: [`feature_status.md`](feature_status.md)
  §4 "Propagation Models" subsection
