# RF + Propagation Toolbox — Tutorial

This tutorial covers two complementary surfaces that share the `examples/rf/` directory. The **Propagation** track (the priority track) provides closed-form and empirical path-loss models, terrain/diffraction analysis, a Longley-Rice (ITM) engineering port, and multi-site directional coverage maps — everything needed for point-to-point link budgets and cellular coverage planning. The **RF** track provides S-parameter circuit primitives: Touchstone I/O, S-parameter conversions, rational fitting, RF budget (Friis) cascades, and closed-form transmission lines. As elsewhere, runtime dispatch is numeric: propagation models, antenna patterns, climates, and environments are selected by small integer tags (see the table below and `examples/rf/README.md`).

## Supported features

### Propagation (priority track)

- **Closed-form / empirical path loss:** `fspl`, `pathlossCloseIn`, `pathlossHata`, `pathlossCost231`, `pathlossEgli`, `pathlossEcc33`, `pathlossSui`, `pathlossEricsson9999`; atmospheric `pathlossRain`, `pathlossGas`, `pathlossFog`.
- **Fresnel & diffraction:** `fresnelZoneRadius`, `fresnelClearance`, `diffractionKnifeEdge`, `diffractionBullington`, `diffractionDeygout`.
- **Geodesy:** `haversine`, `vincenty`, `bearing`, `greatCircleDestLat`/`greatCircleDestLon`.
- **Terrain & LOS:** `terrainProfile`, `losClear`, `losObstruction`.
- **Longley-Rice (ITM):** `itmPathloss` with reliability triple (time/location/situation quantiles), climate codes 1–7.
- **Antenna patterns (analytical):** `isotropicPattern`, `sectorPattern`/`sectorPattern3GPP`, `cosinePattern` (dish), `gaussianPattern`; mount rotation via `applyMountAz`/`applyMountEl`/`applyMountOrientation`.
- **Link budget & coverage:** `linkBudget` (returns a struct), `coverageGrid` (single site), `coverageGridMulti` (multi-site best-server / sum-power / SINR aggregation); `PropagationModel` / `TxSite` / `RxSite` CamelCase classdefs.

### RF (S-parameters)

- **Touchstone I/O:** `touchstoneRead` (s1p…sNp), `touchstoneWrite`, `touchstoneWriteS2p`.
- **S-parameter conversions:** `sparamS2y`/`sparamS2z`/`sparamS2h` (2-port) + N-port `sparamS2yN`/`sparamS2zN`; `gammaIn`, `gammaOut`, `vswr`.
- **Rational fitting / TDR:** `rationalfit` (Gustavsen-Semlyen vector fit), `s2tdr` (TDR step response).
- **RF budget:** `rfbudgetFriis` (gain / NF / IP3 cascade).
- **Transmission lines:** `rfckt_txline`, `rfckt_coaxial`, `rfckt_microstrip` (Hammerstad-Jensen), `rfckt_cpw`, `rfckt_parallelplate`, `rfckt_twowire`, `rfckt_lcfilter`.
- **Smith chart numerics:** `smithGrid`, `smithRCircle`, `smithUnitCircle`.

## Build & run

```bash
build/matlabc -emit-llvm examples/rf/pathloss_models.m > /tmp/pathloss_models.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/pathloss_models.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/pathloss_models
/tmp/pathloss_models
```

The propagation runtime is `runtime/runtime_prop.cpp`; the function-name table lives in `lib/MLIR/Passes/LowerTensorOps.cpp`. The convenience wrapper `runtime/build_and_run.sh examples/rf/<name>.m /tmp/<name>` runs the same path.

### Common numeric tags

| Propagation model (code) | Hata/COST env (code) | Climate (Longley-Rice) | Aggregation (`coverageGridMulti`) | Pattern (code) |
|---|---|---|---|---|
| 0 fspl · 1 Hata · 2 COST-231 · 3 Egli · 4 ECC-33 · 5 SUI · 6 Ericsson · 7 ITM | 1 urban-large · 2 urban med/small · 3 suburban · 4 open/rural | 1 equatorial … 5 cont. temperate (default) … 7 maritime-sea | 0 best-server · 1 sum-power · 2 SINR | 0 isotropic · 1 sector · 2 cosine · 3 gaussian |

All angles are degrees, frequency in Hz, distance in metres, heights in metres above local ground.

## Worked examples

### Path-loss model comparison  (`examples/rf/pathloss_models.m`)

All closed-form models side-by-side at one geometry (30 m BS, 1.5 m mobile, 5 km, 2.4 GHz), plus the atmospheric add-ons.

```matlab
f_Hz = 2.4e9; f_MHz = f_Hz * 1e-6;
ht = 30.0; hr = 1.5; d_m = 5000.0; d_km = 5.0;
L_fs   = fspl(d_m, f_Hz);
L_ci   = pathlossCloseIn(d_m, f_Hz, 3.0, 4.0, 1.0);     % n=3, sigma=4
L_hata = pathlossHata(f_MHz, ht, hr, d_km, 1);          % env 1 = urban-large
L_c231 = pathlossCost231(f_MHz, ht, hr, d_km, 1);
L_sui  = pathlossSui(f_MHz, ht, hr, d_km, 2);           % SUI terrain B
L_rain = pathlossRain(d_m, f_Hz, 25.0, 1.0);            % 25 mm/h, vertical pol
L_gas  = pathlossGas(d_m, f_Hz, 288.15, 1013.25, 10.0);
L_fog  = pathlossFog(d_m, f_Hz, 0.05);
```

The cellular-empirical models take `(f_MHz, ht, hr, d_km, env)` with frequency in MHz and distance in km; `fspl`/`pathlossCloseIn`/atmospheric models take metres and Hz.

### Fresnel zones & multi-edge diffraction  (`examples/rf/fresnel_diffraction.m`)

Fresnel-zone radii at the link midpoint, then knife-edge / Bullington / Deygout diffraction loss over a synthetic two-ridge profile, plus a Fresnel clearance percentage.

```matlab
lambda = 2.998e8 / 5.8e9;
r1 = fresnelZoneRadius(d_total/2, d_total/2, lambda, 1);   % nth zone
L_ke   = diffractionKnifeEdge(h_obs, d1, d2, lambda);
L_bull = diffractionBullington(profile, h_tx, h_rx, d_total, lambda);
L_deyg = diffractionDeygout(profile, h_tx, h_rx, d_total, lambda);
clear_pct = fresnelClearance(profile, h_tx, h_rx, d_total, lambda, 1.0);  % >60% = TIA-clean
```

The terrain `profile` is a column vector of heights sampled along the path; the diffraction routines fold it into an equivalent-edge (Bullington) or recursive multi-edge (Deygout) loss.

### Geodesy helpers  (`examples/rf/geo_helpers.m`)

Great-circle distance (Haversine vs the more accurate Vincenty), initial bearing, and a destination-point projection with a round-trip check.

```matlab
dh = haversine(lat1, lon1, lat2, lon2);   % metres
dv = vincenty (lat1, lon1, lat2, lon2);
az = bearing  (lat1, lon1, lat2, lon2);   % compass degrees
dst_lat = greatCircleDestLat(src_lat, src_lon, d_m, az_east);
dst_lon = greatCircleDestLon(src_lat, src_lon, d_m, az_east);
```

### Longley-Rice (ITM) sweeps  (`examples/rf/longley_rice_link.m`)

Stand-alone `itmPathloss` exercised three ways: reliability triple swept from (50,50,50) median to (95,99,99) microwave-design conservatism, then climate codes 1–7 over an over-the-horizon path, then a frequency sweep.

```matlab
empty_profile = zeros(0, 1);   % no terrain -> ITM area mode
L = itmPathloss(empty_profile, freq_hz, ht, hr, POL_VERTICAL, ...
                5, NS_DEFAULT, SIG_AVG, EPSR_AVG, ...
                d_total, qt, ql, qs_v);   % climate 5, (q_time, q_loc, q_sit)
```

The reliability triple is the time/location/situation quantiles; passing an empty profile puts ITM into area-prediction mode, while a non-empty `terrainProfile(...)` drives the point-to-point regime.

### Multi-site three-sector coverage  (`examples/rf/coverage_three_sector.m`)

Two sites, three 120° sectors each, with both best-server and SINR aggregation over a 48×48 grid.

```matlab
% sites:    [lat lon h_m P_W f_Hz n_ant]
sites = [13.40, -59.55, 30, 5, 2.4e9, 3;
         13.10, -59.35, 30, 5, 2.4e9, 3];
% antennas: [code gain bw_az bw_el fb_or_n mount_az mount_tilt _]  (code 1 = sector)
antennas = [1, 14, 120, 12, 25,   0, 5, 0;
            1, 14, 120, 12, 25, 120, 5, 0;
            1, 14, 120, 12, 25, 240, 5, 0;  % ... repeated for site 2
            ];
grid      = coverageGridMulti(sites, antennas, heightmap, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                              48, 48, 1.5, 0.0, 0 /*FSPL*/, 0 /*best-server*/, 5, 50, 50, 50);
grid_sinr = coverageGridMulti(sites, antennas, heightmap, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                              48, 48, 1.5, 0.0, 0,        2 /*SINR*/,        5, 50, 50, 50);
P_max = max(grid(:));   % grid is [NLAT x NLON] dBm; reduce via max/min/median
```

The antenna matrix has one row per (site, antenna) pair, in site order; `n_ant` in each site row says how many of those rows belong to that site.

### Headline: Barbados PtP + coverage map  (`examples/rf/coverage_barbados.m`)

A Mount-Hillaby ↔ Bridgetown 5.8 GHz link over a synthetic Barbados DEM, with two 22 dBi cosine-pattern dishes aimed across the island via `applyMountAz`/`applyMountEl`, a Longley-Rice link budget at 80/99/99 reliability, and a 48×48 best-server coverage map from the Hillaby antenna.

```matlab
d_m   = haversine(SITE_A_LAT, SITE_A_LON, SITE_B_LAT, SITE_B_LON);
az_AB = bearing(SITE_A_LAT, SITE_A_LON, SITE_B_LAT, SITE_B_LON);
profile = terrainProfile(heightmap, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ...
                         SITE_A_LAT, SITE_A_LON, SITE_B_LAT, SITE_B_LON, 128);
az_local_A = applyMountAz(az_AB, 0.0, az_AB, 0.0);
Gtx_align  = cosinePattern(az_local_A, el_local_A, 8, 8, 22, 30);
lb = linkBudget(SITE_A_LAT, SITE_A_LON, SITE_A_H, FREQ, SITE_A_PW, Gtx_align, ...
                SITE_B_LAT, SITE_B_LON, SITE_B_H,                 Grx_align, ...
                7 /*ITM*/, profile, 3 /*maritime subtropical*/, 80, 99, 99);
fprintf('Path loss: %.2f dB, RX: %.2f dBm, SNR: %.2f dB\n', lb.PathLoss, lb.ReceivedPower, lb.Snr);
```

`linkBudget` returns a struct with `Distance`, `Azimuth`, `PathLoss`, `TxPower_dBm`, `ReceivedPower`, `NoiseFloor`, `Snr`, `LinkMargin`, `FresnelClearance`, `LosClear`, `Frequency`, `Model`, `Profile` — accessed as `lb.PathLoss` etc.

### Other examples (briefly)

- **`amini_barbados_ulap.m`** — a real-world three-site survey (Police / Ilaro Court / QE Hospital) with two directional 5.8 GHz PtP links (full link budget + Fresnel clearance + minimum-mast-height suggestion under TSB-10F 80/99/99) plus a 3.5 GHz three-sector 5G access bubble per site reporting coverage % above the −85 dBm threshold.
- **`antenna_patterns.m`** — sampled `sectorPattern` / `cosinePattern` / `gaussianPattern` gain values and a mount-orientation rotation demo.
- **`prop_smoke.m`** — six-call smoke test touching `fspl`, `pathlossHata`, `fresnelZoneRadius`, `haversine`, `sectorPattern`, `applyMountOrientation`.
- **RF S-parameter side:** there is no standalone RF S-parameter example in `examples/rf/`, but the antenna tutorial's `dipole_sparameters.m` produces an `sparameters` struct via `antennaWireSparameters` and writes it out with `touchstoneWrite("dipole_1ghz.s1p", sp)` — the canonical Antenna → RF bridge.

## Limitations & carve-outs

From `docs/propagation_toolbox_roadmap.md` and `docs/rf_toolbox_plan.md`:

- **Site Viewer 3-D rendering** and **ray tracing through buildings** (`propagationModel('raytracing')`) are carved out; in-scope is the closed-form / ITM path-loss surface.
- **Auto-fetching SRTM / DTED terrain tiles** is out of scope — supply your own `heightmap` matrix (or `load('srtm.mat').heights`).
- **NTIA byte-identical ITM conformance** — the shipped `itmPathloss` is a faithful engineering port of the published regime equations + reliability correction, not the v7.0 reference port.
- **RF Budget Analyzer / Smith Chart Tool apps** (Qt) are out of scope; the numeric primitives (`rfbudgetFriis`, `smithGrid`) ship.
- **Simulink RF Blockset** and the **Modelithics commercial component library** are out of scope.
- **rfckt classdef wrappers** are deferred (function-form `rfckt_*` ship); `analyze(block, freqs)` method dispatch and Harmonic-Balance multi-tone are deferred.
- **TIREM** terrain model is out of scope.

## See also

- Propagation roadmap / design: [`../propagation_toolbox_roadmap.md`](../propagation_toolbox_roadmap.md)
- RF roadmap / design: [`../rf_toolbox_plan.md`](../rf_toolbox_plan.md)
- Bundled cross-toolbox plan & execution order: [`../comm_toolbox_roadmap.md`](../comm_toolbox_roadmap.md)
- Examples directory: `examples/rf/` (see its `README.md` for the full numeric-tag reference and `linkBudget` field list).
