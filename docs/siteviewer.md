# SiteViewer — Feature Status & Roadmap

3D-globe / geographic RF site-planning visualisation, MATLAB-compatible
with the [`siteviewer`](https://www.mathworks.com/help/antenna/ref/siteviewer.html)
class from the Antenna Toolbox. Companion to
[`plotting.md`](plotting.md) (the Cairo render backbone),
[`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) (where SiteViewer
was previously carved out of propagation work), and
[`rf_toolbox_plan.md`](rf_toolbox_plan.md) (RF site/antenna primitives).

The reference workflow this roadmap targets is MathWorks'
[*Planning a 5G Fixed Wireless Access Link Over Terrain*](https://www.mathworks.com/help/antenna/ug/planning-a-5G-fixed-wireless-access-link-over-terrain.html):
place tx/rx on terrain, check line-of-sight, run Longley-Rice path loss,
and overlay coverage maps on a 3D globe — end-to-end, headless on the
default build, interactive in a browser on the HTML-export track.

## 0. Reading guide

- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
- **Layer**: which part of the stack the feature touches.
  - **Runtime numeric** — `runtime/runtime_prop.cpp`, `runtime/rf_class_*.m`
  - **Runtime viz** — `runtime/plot/*` and (planned) `runtime/siteviewer/*`
  - **Sema** — `lib/Sema/Resolver.cpp` builtin registry
  - **Codegen** — `lib/MLIR/Passes/LowerPlot.cpp` `matlab.call_builtin` rewriter
- **Effort tags** mirror the rest of the project: a *session* is half a
  day; a *week* is ≈5 sessions.

## 1. Architecture

```
   .m source ──▶  matlabc Lex/Parse/Sema ──▶ MLIR (matlab.call_builtin)
                        │
                        │ Resolver registers siteviewer/txsite/rxsite/
                        │ show/coverage/los/raytrace/pattern as Builtins
                        ▼
                  LowerPlot pass ──▶ llvm.call @matlab_siteviewer_new
                  (extended to                + @matlab_site_show
                   route the new                + @matlab_site_coverage
                   builtins)                    + @matlab_site_los ...
                        │
                        ▼
                  matlabc JIT
                        │
                        ▼
                  runtime/siteviewer  ──▶  Figure-with-SiteView axes
                  + runtime/runtime_prop      (existing 3D ortho painter)
                  + runtime/rf_class_*.m
                        │
                  ┌─────┴───────────────┐
                  ▼                     ▼
            PNG / SVG / PDF       Cesium / globe.gl HTML
            (existing             (self-contained bundle —
             headless path)        interactive 3D in browser)
```

Static rendering re-uses the **existing** Cairo 3D pipeline (`surf` /
`mesh` / `plot3` + `view(az, el)` ortho projection + painter's-algorithm
depth sort). The interactive track emits a self-contained HTML bundle
following the same channel as `plotting.md` §4 Tier D. No live window
in the default build; one is gated on `MATLAB_LLVM_WITH_PLOT_WINDOW`
(plot Tier E).

## 2. Already shipped

### 2.1 RF site numerics

| Item | Status | Layer | Notes |
|---|---|---|---|
| `TxSite` classdef | ✅ | runtime numeric | `runtime/rf_class_txsite.m`. Properties: Latitude/Longitude (WGS84), AntennaHeight, TransmitterFrequency, TransmitterPower, AntennaGain, AntennaAngle (boresight), SystemLoss. |
| `RxSite` classdef | ✅ | runtime numeric | `runtime/rf_class_rxsite.m`. Properties: Latitude/Longitude, AntennaHeight, AntennaAngle, AntennaGain, ReceiverSensitivity, SystemLoss. |
| `link(tx, rx)` | ✅ | runtime numeric | Haversine distance + budget. |
| `los(tx, rx)` | ✅ | runtime numeric | Geometric LOS with 4/3 Earth bulge. |
| `show(site)` | 🟡 | runtime numeric | Text-only stub — prints site metadata. The viz binding is what this roadmap delivers. |
| `coverage(tx, pm)` | 🟡 | runtime numeric | Numeric grid + stats; no map overlay yet. |

### 2.2 Propagation models

| Item | Status | Notes |
|---|---|---|
| FSPL | ✅ | Free-space. |
| Hata / COST231-Hata / ECC33 / SUI / Ericsson | ✅ | Closed-form ITU-R / NIST. |
| Rain / gas / fog attenuation | ✅ | ITU-R atmospheric. |
| Longley-Rice (ITM v7.0) | ✅ | Terrain-aware, Fresnel + knife-edge diffraction. Implemented in `runtime/runtime_prop.cpp`. |
| `coverage_grid` (single TX) | ✅ | 30 × 30 default; omnidirectional. |
| `coverage_grid_multi` | ✅ | Best-server / sum-power / SINR aggregation. |

### 2.3 Geographic primitives

| Item | Status | Notes |
|---|---|---|
| Haversine distance | ✅ | `runtime/runtime_prop.cpp`. |
| Vincenty (WGS84 ellipsoidal) | ✅ | Same file. |
| Bearing, destination lat/lon | ✅ | Same file. |
| Terrain profile sampling | ✅ | Used by Longley-Rice. |

### 2.4 Cairo 3D pipeline

| Item | Status | Notes |
|---|---|---|
| Orthographic `view(az, el)` | ✅ | `runtime/plot/cairo_render.cpp:809-865`. Default (-37.5°, 30°). |
| `surf(X, Y, Z)` / `mesh` / `plot3` | ✅ | Painter's-algorithm depth sort; per-face Lambertian shading via active colormap. |
| Colormap LUTs (parula/jet/viridis/hot/cool/gray) | ✅ | `runtime/plot/colormap.cpp`. |
| `imagesc` / `imshow` | ✅ | Heatmap on a flat axes. |

### 2.5 Codegen wiring

| Item | Status | Notes |
|---|---|---|
| `siteviewer` registered as builtin | 🟡 | `lib/Sema/Resolver.cpp:159`. Currently a 0-arg stub: `LowerTensorOps` routes to `matlab_prop_siteviewer_stub`, which prints `[siteviewer stub: no GUI; site views skipped]`. |
| `pathloss` / `los` / `link` / `sigstrength` / `coverage` / `show` builtins | ✅ | `lib/Sema/Resolver.cpp:156-159`. |

## 3. MATLAB API surface to target

The plan tracks the MathWorks `siteviewer` reference. API surface below
is the target — methods land tier-by-tier in §4.

### 3.1 Viewer construction & properties

```matlab
v = siteviewer;
v = siteviewer('Name','Boston','Basemap','satellite','Terrain','gmted2010');
```

| Name/Value | Type | Notes |
|---|---|---|
| `Name` | char | Window title. |
| `Position` | [x y w h] | Window position (window backend only). |
| `Basemap` | char | `'satellite'` · `'streets'` · `'streets-light'` · `'streets-dark'` · `'topographic'` · `'darkwater'` · `'lightwater'` · `'openstreetmap'` · `'none'` (default for offline-safe build). |
| `Terrain` | char / struct | `'gmted2010'` · `'none'` · custom DTED struct. |
| `Buildings` | char / `'none'` | Path to `.osm` file. |
| `BuildingsMaterial`, `TerrainMaterial` | char | Material name (visual only in this roadmap; RF semantics belong to a future raytracer). |
| `SceneModel` | object | Pre-built scene. |
| `Transparency` | 0..1 | Layer opacity. |
| `ShowEdges` | bool | Outline buildings. |
| `ControlVisibility` | bool | UI chrome (HTML-export only). |

### 3.2 Methods

```matlab
show(v); hide(v); close(v);
centerView(v, site);
screenshot(v, 'snap.png');   % or 'snap.html' for interactive bundle
current = siteviewer.current;
addCustomBasemap(...);  removeCustomBasemap(...);
addCustomTerrain(...);  removeCustomTerrain(...);
```

### 3.3 Functions targeting a viewer via `'Map', v`

```matlab
tx = txsite('Latitude',42.36,'Longitude',-71.06, ...);
rx = rxsite(...);

show(tx, 'Map', v);
coverage(tx, pm, 'Map', v, 'SignalStrengths', -100:-30);
sinr(txs,    'Map', v);
raytrace(tx, rx, pm, 'Map', v);
los(tx, rx,        'Map', v);
pattern(tx, freq,  'Map', v);
ss = signalStrength(tx, rx, pm);   % scalar dBm — already in numerics
```

## 4. Roadmap

### Tier 1 — Data layer (~1 week)

Foundation. Everything downstream depends on these.

| Item | Effort | Notes |
|---|---|---|
| Geodetic ↔ ECEF ↔ local ENU transforms | 1 session | Extend the Vincenty/Haversine path already in `runtime/runtime_prop.cpp`. |
| DEM heightmap struct | 1 session | User-supplied `.mat` / `.hgt` / `.tif` via existing `readmatrix`; struct holds `lat0/lon0/dlat/dlon/rows/cols/data`. |
| Lat/lon → altitude bilinear sampler | trivial | Reused by Longley-Rice today (extract + share). |
| `.osm` XML parser | 2 sessions | Lightweight — building footprints (polygons) + `height` / `building:levels` tag. No relations, no roads (out of scope). |
| Tile / image basemap loader | 1 session | Single image, user-supplied path; covers the "Custom basemap" Name/Value. Tiled providers come in Tier 7. |

### Tier 2 — SiteViewer object & static headless render (~1 week)

| Item | Effort | Notes |
|---|---|---|
| `Axes::projection = SiteView` enum | trivial | New mode flag alongside Cartesian/Polar/Geographic. |
| `siteviewer(...)` real constructor | 1 session | Returns a `Figure*` handle whose active axes is in `SiteView` mode. Replaces the existing print stub. |
| 2D equirectangular fallback | 1 session | Lat/lon → x/y; terrain heightmap painted via `imagesc`; sites as `scatter` markers; status as `text`. Works in any build. |
| 3D ortho terrain | 2 sessions | Build `(X, Y, Z)` mesh from DEM with ENU coords centred on view; render via existing `surf` painter. Elevation-shaded by default. |
| `show(tx)`, `show(rx)` route to active viewer | 1 session | If first arg's class is `TxSite`/`RxSite` and a Name/Value `'Map', v` is present (or a viewer is current), pin to viewer. |
| `screenshot(v, 'f.png')` | trivial | Direct map to existing `matlab_savefig`. |
| `centerView(v, site)` | trivial | Mutates `Axes::view_az/el` + viewer pan offset. |
| `close(v)` / `hide(v)` / `current` | trivial | Reuse figure lifecycle from `runtime/plot/figure.cpp`. |

### Tier 3 — Coverage / LoS / pattern overlay (~1 week)

| Item | Effort | Notes |
|---|---|---|
| Per-vertex / per-face colour on `surf` | 1 session | Already planned in `plotting.md` §3 Tier 2 (`surf(X,Y,Z,C)` with explicit colour matrix). Prereq for transparent coverage heatmap. |
| `coverage(tx, pm, 'Map', v)` | 2 sessions | Wraps existing `coverage_grid_multi`; output painted as semi-transparent coloured mesh draped on terrain. |
| `los(tx, rx, 'Map', v)` | 1 session | 3D polyline (`plot3`) coloured green if `los(tx,rx)` is true, red otherwise. Renders both endpoints as site markers. |
| `pattern(tx, freq, 'Map', v)` | 1 session | Antenna lobe as a small textured mesh anchored at site lat/lon. Until ANT-Tier-2 lands, falls back to isotropic spheroid sized by power. |
| `raytrace(tx, rx, pm, 'Map', v)` v1 | 2 sessions | Direct ray + first-order ground reflection only. Building reflections deferred. |
| `sinr(txs, 'Map', v)` | 1 session | Existing `coverage_grid_multi` SINR mode + same heatmap painter as `coverage`. |

### Tier 4 — Interactive HTML export (~1–2 weeks)

`screenshot(v, 'site.html')` / `saveas(v, 'site.html')` emits a
self-contained bundle. Works in any browser, including iOS Safari.

| Item | Effort | Notes |
|---|---|---|
| HTML scaffolder | 2 sessions | Embedded template (single C++ header with the HTML string), JSON serialisation of viewer state. Shares scaffolding with `plotting.md` §4 Tier D. |
| Globe library choice (Cesium vs. globe.gl) | trivial | CMake-time pick. Cesium = full-fidelity, ~3 MB. globe.gl (three.js wrapper) = ~500 KB, simpler basemap story. Default = globe.gl for size; Cesium opt-in via `MATLAB_LLVM_WITH_PLOT_CESIUM`. |
| Terrain export | 2 sessions | DEM → quantized-mesh tiles (Cesium) or as a heightmap PNG + custom shader (globe.gl). |
| Sites as entities | 1 session | Cesium billboards / globe.gl points with lat/lon/height + label. Click handler opens info panel. |
| Coverage / SINR overlay | 1 session | Coloured image overlay clamped to viewer extent. |
| LoS / raytrace polylines | 1 session | Native Cesium polyline / globe.gl arc primitives. |
| Antenna pattern as 3D mesh | 1 session | Triangulated lobe exported as glTF inline. |

### Tier 5 — Native live globe (gated on plot Tier E, ~1 week on top)

Only meaningful if `MATLAB_LLVM_WITH_PLOT_WINDOW` (plot §4 Tier E) is
already enabled.

| Item | Effort | Notes |
|---|---|---|
| Viewer in SDL window | 2 sessions | Cairo paints into a texture; mouse-drag rotates az/el; wheel zooms; click picks sites. Reuses headless painter. |
| `centerView` animation | 1 session | Smooth az/el interp over ~0.5 s for nicer UX. |
| Hover tooltips on sites | 1 session | Same nearest-point lookup as `datacursormode`. |

### Tier 6 — Buildings / OSM scene (~1 week)

| Item | Effort | Notes |
|---|---|---|
| `.osm` polygon extrusion | 2 sessions | Convert each footprint + height to a 3D prism; render via existing `surf`/`mesh`. |
| `BuildingsMaterial` styling | 1 session | Visual property — colour, opacity. No RF material semantics (those belong to a future raytracer). |
| `ShowEdges` | trivial | Already implicit in mesh wireframe; just toggle the edge draw. |
| `.stl` / `.kml` building input | 1 session each | Wrap existing geometry loaders if present, otherwise minimal parsers. |

### Tier 7 — Online basemaps & tiles (~3 sessions, optional)

| Item | Effort | Notes |
|---|---|---|
| `libcurl` integration | 1 session | New optional dep behind `MATLAB_LLVM_WITH_PLOT_TILES`. Off by default — keeps the offline / iOS story intact. |
| OpenStreetMap raster tiles | 1 session | XYZ tile fetch + on-disk cache; OSM tile-usage policy respected (per-request UA, no high-volume). |
| MapTiler / Stamen / custom XYZ | 1 session | Generic XYZ template via Name/Value or env var. Vendor keys never hard-coded. |

### Tier 8 — Explicitly out of scope

These match the existing comm/RF/propagation carve-outs and stay carved
out under this roadmap.

| Item | Reason |
|---|---|
| Auto-fetch SRTM / GMTED terrain | User supplies the heightmap. No bundled tile downloader. |
| Bing / Google satellite imagery | Vendor auth, quotas, $$. Use OSM-style tiles via Tier 7 instead. |
| Auto-fetch buildings from OSM Overpass API | Use offline `.osm` files. |
| TIREM propagation | Proprietary (DoD-licensed). |
| Production ray-tracing through buildings (higher-order reflections, diffraction over edges, transmission through walls) | Separate large effort; needs an RF-material database and acceleration structure (BVH/KD-tree). Tier 3 raytrace v1 ships direct + ground reflection only. |
| WebGL realtime fidelity matching Cesium's terrain LOD streaming | Cesium handles this when embedded; we won't re-implement it natively. |
| Live-Editor / App-Designer / `uifigure` integration | Same rationale as the plotting §3 Tier 5 carve-out. |

## 5. Touch points

### Existing files extended

- `lib/Sema/Resolver.cpp:156-159` — `kBuiltins` grows: `txsite`, `rxsite`, `siteviewer`, `show`, `hide`, `close`, `centerView`, `screenshot`, `coverage`, `sinr`, `los`, `pattern`, `raytrace`, `signalStrength`.
- `lib/MLIR/Passes/LowerPlot.cpp:122` — `plotBuiltins()` extended; new arity branches in `rewriteCallee()` for `siteviewer(...)`, `show(site, 'Map', v)`, `coverage(tx, pm, 'Map', v, …)`, etc. Removes the `matlab_prop_siteviewer_stub` route once the real impl lands.
- `runtime/runtime_prop.cpp` — ENU transforms next to the existing Vincenty/Haversine helpers; DEM sampler factored out for reuse.
- `runtime/plot/figure.h` — `Axes::projection` enum (`Cartesian` / `Polar` / `SiteView`); new `SeriesKind`: `Globe`, `BuildingExtrusion`, `CoverageOverlay`, `SiteMarker`.
- `runtime/plot/cairo_render.cpp` — geo projection helper (`lat/lon/h` → ENU → screen); per-face colour wired through `surf` painter.
- `runtime/rf_class_txsite.m` / `rxsite.m` — `show` / `hide` / `pattern` methods routed to viewer when `'Map', v` is supplied.

### New files

- `runtime/siteviewer/c_api.cpp` — `matlab_siteviewer_new`, `matlab_site_show`, `matlab_site_coverage`, `matlab_site_los`, `matlab_site_pattern`, `matlab_site_raytrace`, `matlab_siteviewer_screenshot`.
- `runtime/siteviewer/terrain.cpp` — DEM struct, lat/lon → altitude, ENU mesh builder.
- `runtime/siteviewer/osm_parser.cpp` — minimal OSM XML → footprint list.
- `runtime/siteviewer/html_export.cpp` — Cesium / globe.gl bundle emitter.
- `runtime/siteviewer/templates.h` — embedded HTML / JS templates as `static constexpr` string literals.

### Build

- `CMakeLists.txt` — new optional umbrella `MATLAB_LLVM_WITH_PLOT_SITEVIEWER` (defaults ON when `WITH_PLOT` is ON; can be turned off for minimal builds). Sub-options: `…_TILES` (libcurl), `…_CESIUM` (Cesium template vs. globe.gl default).

### Tests

- `test/Runtime/test_siteviewer_basic.cpp` — direct C ABI: build a viewer, place tx/rx, render PNG, assert non-empty.
- `test/Runtime/test_siteviewer_coverage.cpp` — coverage overlay produces non-uniform colours over terrain.
- `test/Runtime/test_siteviewer_html.cpp` — HTML bundle parses as valid HTML, contains site lat/lon, opens self-contained.
- `test/Run/rf_siteviewer_basic.m`, `rf_siteviewer_5g_fwa.m` — end-to-end through matlabc reproducing the MathWorks 5G FWA tutorial. With `.skip-emit-c`, `.skip-emit-cpp`, `.skip-emit-python`, `.skip-emit-typescript` markers matching the existing `rf_*` test pattern (SystemVerilog never receives plot ops; the others stay carved out by design — see Tier G of `plotting.md` §4).

### Examples

- `examples/rf/siteviewer_basic.m` — minimal 1-line viewer + 1 site.
- `examples/rf/siteviewer_5g_fwa.m` — the MathWorks tutorial, end-to-end.
- `examples/rf/siteviewer_coverage_multisite.m` — 3-cell coverage with SINR map.

## 6. Execution order

1. **Tier 1** (data layer) — ~1 week. Prerequisite for everything else.
2. **Tier 2** (static viewer + 2D + 3D ortho terrain) — ~1 week. First visible output. Replaces the `siteviewer_stub` print. Already covers the basic MathWorks workflow with terrain.
3. **Tier 3** (coverage / LoS / pattern overlay) — ~1 week. Now matches the *5G FWA over Terrain* tutorial end-to-end.
4. **Tier 4** (Cesium / globe.gl HTML export) — ~1–2 weeks. First fully interactive globe. Works on iOS.
5. **Tier 6** (buildings) — pick up when raytrace v2 or urban coverage demand pulls it in.
6. **Tier 5** (native live globe) — only after plot Tier E ships and a desktop-only build is desired.
7. **Tier 7** (online tile basemaps) — last; optional dep; default stays offline-safe.

After Tier 3 the project ships a credible MathWorks-compatible
`siteviewer` for terrain-aware RF planning. Tier 4 is where it becomes
delightful to actually use.

## 7. Quick-reference: future usage

### From a `.m` file via matlabc (target API once Tier 3 lands)

```matlab
% Load a user-supplied DEM. SRTM / GMTED auto-fetch is carved out;
% user supplies the heightmap.
dem = load('boston_dem.mat');           % struct with lat0/lon0/dlat/dlon/data

v = siteviewer('Name','Boston FWA', ...
               'Terrain', dem, ...
               'Basemap','none');       % offline-safe default

tx = txsite('Latitude', 42.3601, 'Longitude', -71.0589, ...
            'AntennaHeight', 30, ...
            'TransmitterFrequency', 28e9, ...   % 28 GHz mmWave
            'TransmitterPower', 1.0, ...
            'AntennaGain', 25);

rx = rxsite('Latitude', 42.3554, 'Longitude', -71.0640, ...
            'AntennaHeight', 5, ...
            'AntennaGain', 12, ...
            'ReceiverSensitivity', -90);

pm = propagationModel('longley-rice');

show(tx, 'Map', v);
show(rx, 'Map', v);
los(tx, rx, 'Map', v);
coverage(tx, pm, 'Map', v, 'SignalStrengths', -100:5:-30);

screenshot(v, 'fwa_static.png');        % Tier 2 — PNG
screenshot(v, 'fwa_interactive.html');  % Tier 4 — self-contained Cesium
```

### From C++ via the C ABI (Tier 2+)

```cpp
#include "matlab_plot.h"
#include "matlab_siteviewer.h"   // new in Tier 2

auto *v = matlab_siteviewer_new();
matlab_siteviewer_set_terrain(v, &dem);          // dem: matlab_dem*

matlab_site tx = {.lat=42.3601, .lon=-71.0589, .h=30, .pow_w=1.0, .gain_dbi=25};
matlab_site rx = {.lat=42.3554, .lon=-71.0640, .h=5,  .gain_dbi=12};

matlab_site_show(v, &tx);
matlab_site_show(v, &rx);
matlab_site_los(v, &tx, &rx);
matlab_site_coverage(v, &tx, MATLAB_PM_LONGLEY_RICE);

matlab_siteviewer_screenshot(v, "fwa_static.png", 14);
matlab_siteviewer_screenshot(v, "fwa_interactive.html", 21);
```

The C ABI follows the same patterns as `matlab_plot.h`: opaque
viewer/site handles, length-prefixed strings, malloc'd buffer returns
freed via `matlab_plot_buffer_free`.

## 8. Related docs

- [`plotting.md`](plotting.md) — Cairo render backbone (§4 animation/interactivity tiers share scaffolding with this roadmap's Tier 4 HTML export).
- [`comm_toolbox_roadmap.md`](comm_toolbox_roadmap.md) — propagation tracks; this doc un-carves SiteViewer from there.
- [`rf_toolbox_plan.md`](rf_toolbox_plan.md) — TxSite/RxSite primitives and antenna patterns.
- [`feature_status.md`](feature_status.md) — top-level project status.
