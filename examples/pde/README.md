# examples/pde — Partial Differential Equation Toolbox examples

Per-tier examples corresponding to
[`docs/pde_toolbox_roadmap.md`](../../docs/pde_toolbox_roadmap.md).
Each script is also a **gating test** for its tier: when the script
runs end-to-end under `matlabc -emit-llvm`, the tier is closed.

Status legend: ✅ shipped · 🟡 partial · 🔵 not started.

## Tier 1 — 2-D elliptic FEM (small loop)

| File | Tier | Status | What it gates |
|---|:-:|:-:|---|
| `poisson_disk.m` | T1 | 🔵 | `createpde` → `decsg` → `applyBoundaryCondition` → `specifyCoefficients` → `generateMesh` → `solvepde` → `interpolateSolution` → `pdeplot` |
| `poisson_lshape.m` | T1 | 🔵 | Same surface on the L-shaped domain (corner-singularity refinement) |
| `eig_lshape_membrane.m` | T1 | 🔵 | `solvepdeeig` on the 2-D Laplace operator |
| `wave_square.m` | T1 | 🔵 | 2-D hyperbolic transient via `solvepde(model, tlist)` |

## Tier 2 — 3-D unified `femodel` workflow (headline)

| File | Tier | Status | What it gates |
|---|:-:|:-:|---|
| `clamped_plate_pressure.m` | T2 | 🔵 | `multicuboid` → `femodel("structuralStatic")` → `faceBC` clamp + `faceLoad(Pressure=…)` → 3-D von Mises stress |
| `bracket_deflection.m` | T2 | 🔵 | `faceLoad(SurfaceTraction=[Tx;Ty;Tz])` vector traction |
| **`wind_stress_3d.m`** | T2 | 🔵 | **Headline demo** — 3-D model under 250 km/h wind pressure, von Mises stress map on the deformed shape |
| `wing_spar_static.m` | T2 | 🔵 | I-beam wing-spar with distributed traction; validates multi-face load surfaces |
| `tuningfork_modal.m` | T2 | 🔵 | STL import + `structuralModal` analysis + `solvepdeeig` |

## Tier 3 — Transient + thermal + electromagnetics

| File | Tier | Status | What it gates |
|---|:-:|:-:|---|
| `heat_sink_transient.m` | T3 | 🔵 | `thermalTransient` + ODE backbone |
| `jet_turbine_thermal_stress.m` | T3 | 🔵 | Chained thermal → structural with `cellLoad(Temperature=…)` |
| `electrostatic_busbar.m` | T3 | 🔵 | `electrostatic` analysis |
| `magnetostatic_2pole_motor.m` | T3 | 🔵 | 2-D `magnetostatic` |
| `tuningfork_transient.m` | T3 | 🔵 | Modal-superposition transient response |

## Tier 4 — ROM + nonlinear + adaptive

| File | Tier | Status | What it gates |
|---|:-:|:-:|---|
| `wing_spar_rom.m` | T4 | 🔵 | Craig-Bampton reduced-order model |
| `clamped_beam_nonlinear.m` | T4 | 🔵 | Geometric-nonlinear plane-stress |
| `adaptive_poisson_point_source.m` | T4 | 🔵 | `adaptmesh` adaptive refinement |

## Tier 5 — Battery / specialty (carve-out candidates)

| File | Tier | Status | What it gates |
|---|:-:|:-:|---|
| `battery_p2d_basic.m` | T5 | 🔵 | `batteryP2DModel` |

## Running

Once the tier is shipped, run any script via:

```sh
./matlabc -emit-llvm examples/pde/wind_stress_3d.m -o /tmp/wind.exe
/tmp/wind.exe
```

Or in the REPL:

```sh
./matlabc -repl
>> run('examples/pde/wind_stress_3d.m')
>> R                    % inspect StaticStructuralResults
>> R.VonMisesStress(1:10)
```

The PNG output lands in the current working directory as
`figure_001.png` (Cairo backend, headless).
