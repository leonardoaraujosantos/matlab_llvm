# examples/rf — Propagation & RF Toolbox demos

End-to-end demos of the propagation-models surface shipped under PROP-Tier 1a / 2a / 2b / 3 of `docs/comm_toolbox_roadmap.md`. Every example is a single `.m` file that runs through the standard compile-and-execute path:

```bash
bash runtime/build_and_run.sh examples/rf/<name>.m /tmp/<name>
/tmp/<name>
```

The propagation runtime is `runtime/runtime_prop.cpp`; the function-name table lives in `lib/MLIR/Passes/LowerTensorOps.cpp`.

| Example | What it demonstrates |
|---|---|
| `prop_smoke.m` | Six-call smoke test: `fspl`, `pathlossHata`, `fresnelZoneRadius`, `haversine`, `sectorPattern`, `applyMountOrientation`. |
| `pathloss_models.m` | All closed-form models side-by-side at one geometry: free-space, Close-In, Hata (urban/suburban/open), COST-231, Egli, ECC-33, SUI, Ericsson 9999, plus ITU-R rain / gas / fog. |
| `fresnel_diffraction.m` | Fresnel-zone radii + single-edge knife-edge + Bullington equivalent edge + Deygout 3-edge multi-obstacle diffraction on a synthetic two-ridge profile, plus a Fresnel-clearance percentage. |
| `geo_helpers.m` | Haversine vs Vincenty distances, initial bearing, great-circle destination point. |
| `antenna_patterns.m` | Analytical directional patterns: `sectorPattern`, `cosinePattern`, `gaussianPattern`. Mount-orientation rotation via `applyMountAz` / `applyMountEl`. |
| `longley_rice_link.m` | Stand-alone Longley-Rice (ITM) sanity sweeps: reliability triple (50,50,50) → (95,99,99), climate codes 1–7, and a frequency sweep at fixed reliability. |
| `coverage_three_sector.m` | Multi-site coverage with two sites × three 120-deg sectors each, best-server and SINR aggregation. |
| `coverage_barbados.m` | **Headline scenario** — Mount Hillaby ↔ Bridgetown PtP link on a synthetic Barbados heightmap, two 22 dBi cosine-pattern directional dishes, Longley-Rice (ITM) path loss, plus a 48×48 coverage map from Mount Hillaby. |
| `amini_barbados_ulap.m` | **Real-world site survey** — Amini ULAP three-site survey in Bridgetown: Police Command Center, Ilaro Court, Queen Elizabeth Hospital. Two directional 5.8 GHz Longley-Rice PtP links (Police↔Ilaro, Police↔Hospital) with full link budget + Fresnel-zone clearance check + minimum-mast-height suggestion under TSB-10F 80/99/99 reliability. Per-site 5G access bubble at 3.5 GHz (10 W, three 120° sectors per site) reporting coverage % above the −85 dBm 600 Mbit/s NR threshold. Heightmap is a synthetic Bridgetown-area DEM; swap in a real SRTM tile via `load('srtm.mat').heights` when available. |

## Numeric-tag conventions

To keep the runtime dispatch simple (no string-arg path), the API uses small integer tags. Common codes:

**Propagation models** (`linkBudget`, `coverageGrid`, `coverageGridMulti`):

| code | model |
|---|---|
| 0 | `fspl` (free-space) |
| 1 | `pathlossHata` (urban-large) |
| 2 | `pathlossCost231` |
| 3 | `pathlossEgli` |
| 4 | `pathlossEcc33` |
| 5 | `pathlossSui` (terrain B) |
| 6 | `pathlossEricsson9999` |
| 7 | Longley-Rice (ITM) |

**Antenna pattern codes** (`coverageGridMulti` antennas matrix):

| code | pattern |
|---|---|
| 0 | `isotropicPattern` |
| 1 | `sectorPattern` (3GPP-style) |
| 2 | `cosinePattern` (parabolic dish) |
| 3 | `gaussianPattern` |
| 4 | `sectorPattern3GPP` (synonym of 1 here) |

**Climate codes** (Longley-Rice `itmPathloss`):

| code | climate |
|---|---|
| 1 | Equatorial |
| 2 | Continental subtropical |
| 3 | Maritime subtropical |
| 4 | Desert |
| 5 | Continental temperate (default) |
| 6 | Maritime temperate over land |
| 7 | Maritime temperate over sea |

**Hata / COST-231 environment codes** (`pathlossHata`, `pathlossCost231`):

| code | env |
|---|---|
| 1 | Urban large |
| 2 | Urban medium-small |
| 3 | Suburban |
| 4 | Open / rural |

**SUI terrain codes** (`pathlossSui`):

| code | terrain |
|---|---|
| 1 | A (hilly, dense) |
| 2 | B (hilly, light) |
| 3 | C (flat) |

**Multi-site coverage aggregation** (`coverageGridMulti`):

| code | aggregation |
|---|---|
| 0 | Best-server (default) — `max(P_rx_i)` per pixel, returned in dBm. |
| 1 | Sum-power — incoherent power sum across all (site, antenna) pairs, in dBm. |
| 2 | SINR (dB) — `serving / (Σ_others + N₀·B)`; useful for cellular planning. |

## Notes

- All angle inputs and outputs are degrees unless explicitly noted.
- Frequency is Hertz, distance is metres, antenna heights are metres above local ground.
- The `heightmap` matrix is `[NumLat × NumLon]` real samples spanning a `(lat_min, lat_max) × (lon_min, lon_max)` bounding box; values are elevations above mean sea level in metres. Auto-fetching SRTM/DTED tiles is out of scope (see `docs/comm_toolbox_roadmap.md §3.7`).
- `linkBudget` returns a struct with fields `Distance`, `Azimuth`, `PathLoss`, `TxPower_dBm`, `ReceivedPower`, `NoiseFloor`, `Snr`, `LinkMargin`, `FresnelClearance`, `LosClear`, `Frequency`, `Model`, `Profile`. Access via `lb.Distance` etc.
- The Longley-Rice (ITM) port is a faithful engineering implementation of the published closed-form regime equations + reliability quantile correction. For NTIA byte-identical conformance, swap in the v7.0 reference port (carved out per roadmap §3.7).
