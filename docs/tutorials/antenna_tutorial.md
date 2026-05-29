# Antenna Toolbox — Tutorial

The Antenna Toolbox surface in matlab_llvm is the **ANT-Tier-2 closed-form thin-dipole MVP**: a center-fed, sinusoidal-current thin-wire dipole analysed with the induced-EMF method (Balanis Eq. 8-60). From a length, wire radius, segment count, and frequency it returns input impedance, S11 / VSWR / return loss, the far-field radiation pattern with directivity, and a frequency-swept S-parameter struct that drops straight into the RF Toolbox via Touchstone. The reference half-wave dipole at 1 GHz gives the textbook `Z_in ≈ 73.13 + j42.55 Ω` and ≈ 2.15 dBi directivity. This tutorial is grounded in the examples under `examples/antenna/`.

## Supported features

- **Impedance & matching solve:** `antennaWireSolve(L, a, n_segs, freq)` → struct with `Zin_re`, `Zin_im`, `VSWR`, `ReturnLoss_dB` (S11 referenced to 50 Ω).
- **Far-field pattern:** `antennaWirePattern(L, a, n_segs, freq, n_theta)` → struct with `Directivity_dBi`, `Zin_re`, `Zin_im` (closed-form sinusoidal-current pattern `F(θ) = (cos(½kL·cosθ) − cos(½kL)) / sinθ`).
- **Frequency sweep + RF bridge:** `antennaWireSparameters(L, a, n_segs, freqs)` → `sparameters` struct with `NumPorts`, `Z0`, and swept S11(f); hand off to `touchstoneWrite(filename, sp)` to emit a `.s1p` Touchstone file.

All three share the same thin-wire model parameters: physical length `L` (m), wire radius `a` (m), number of current segments `n_segs`, and frequency in Hz.

## Build & run

```bash
build/matlabc -emit-llvm examples/antenna/dipole_halfwave.m > /tmp/dipole_halfwave.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/dipole_halfwave.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/dipole_halfwave
/tmp/dipole_halfwave
```

The convenience wrapper `runtime/build_and_run.sh examples/antenna/<name>.m /tmp/<name>` runs the same compile-and-execute path.

## Worked examples

### Half-wave dipole impedance  (`examples/antenna/dipole_halfwave.m`)

A center-fed thin half-wave dipole at 1 GHz solved over 21 current segments, recovering the textbook input impedance, VSWR, and return loss.

```matlab
freq = 1.0e9;
c0   = 2.99792458e8;
lambda = c0 / freq;
L = 0.5 * lambda;        % 0.15 m  (half wavelength)
a = 0.001 * lambda;      % thin wire, a/lambda = 0.001

r = antennaWireSolve(L, a, 21, freq);
disp(r.Zin_re);          % ~73.08
disp(r.Zin_im);          % ~42.52
disp(r.VSWR);            % ~2.18 (referenced to 50 ohm)
disp(r.ReturnLoss_dB);   % ~8.6 dB
```

The result struct's fields are read directly with dot access. The ~73 + j42 Ω impedance matches the half-wave reference; the 2.18 VSWR against 50 Ω is the expected mismatch of an unmatched resonant dipole.

### Radiation pattern & directivity  (`examples/antenna/dipole_pattern.m`)

The far-field pattern sampled over 181 θ points, returning the broadside directivity.

```matlab
freq = 1.0e9;
lambda = 2.99792458e8 / freq;
L = 0.5 * lambda;
a = 0.001 * lambda;

p = antennaWirePattern(L, a, 21, freq, 181);
disp(p.Directivity_dBi);    % ~2.15
disp(p.Zin_re);             % ~73.08
disp(p.Zin_im);             % ~42.52
```

The pattern peaks broadside (θ = π/2) with deep nulls along the wire axis; the ≈ 2.15 dBi directivity (1.64 linear) is the half-wave dipole textbook value. The struct also carries the impedance so a single pattern call gives both.

### Frequency sweep + Touchstone bridge  (`examples/antenna/dipole_sparameters.m`)

Sweep S11(f) across seven frequencies, then write it out as an `.s1p` Touchstone file consumable by any RF tool (Spectre / ngspice / ADS) — the canonical Antenna → RF Toolbox hand-off.

```matlab
L = 0.15;            % half-wave at 1 GHz
a = 0.0003;
freqs = [7e8; 8e8; 9e8; 1.0e9; 1.1e9; 1.2e9; 1.3e9];

sp = antennaWireSparameters(L, a, 21, freqs);
disp(sp.NumPorts);   % 1
disp(sp.Z0);         % 50

touchstoneWrite("dipole_1ghz.s1p", sp);   % swept S11(f) -> .s1p
```

`antennaWireSparameters` returns an `sparameters`-shaped struct (1 port, Z0 = 50); `touchstoneWrite` serialises it in MA format. The emitted `examples/antenna/dipole_1ghz.s1p` is the round-trip artifact and can be read back by the RF Toolbox's `touchstoneRead`.

## Limitations & carve-outs

From `docs/antenna_toolbox_roadmap.md §6` and the tier table:

- **General thin-wire MoM** (arbitrary wire geometries, multi-wire structures) is carved into a follow-on tier (ANT-Tier-2b) — the shipped model is the closed-form center-fed straight dipole only.
- **`AntDipole` / `AntMonopole` classdefs** + `design(ant, freq)` / `antennaGain(ant, freq)` are a partial ANT-Tier-1 catalog item; the function-form `antennaWire*` path is the recommended surface.
- **Reflector / aperture antennas** (`reflectorParabolic`), **arrays**, and **rigorous mutual coupling** are carved into Tier 5 (multi-month each).
- **Antenna Designer / Array Designer / PCB Antenna Designer apps** and **Gerber export** are out of scope (interactive Qt apps, not language features).
- **AI for Antennas** (DL surrogate models), **GPU-accelerated MoM**, **custom antenna from photo** (CV geometry inference), and **real-time 3-D field/current/pattern visualisation** are out of scope.
- **Simulink antenna blocks** are out of scope.

## See also

- Roadmap / design: [`../antenna_toolbox_roadmap.md`](../antenna_toolbox_roadmap.md)
- RF Toolbox bridge (Touchstone, S-parameter cascades): [`../rf_toolbox_plan.md`](../rf_toolbox_plan.md) and [`rf_propagation_tutorial.md`](rf_propagation_tutorial.md)
- Propagation directional-pattern hook (uses antenna patterns in `coverageGridMulti`): [`../propagation_toolbox_roadmap.md`](../propagation_toolbox_roadmap.md)
- Examples directory: `examples/antenna/`
