# Documentation Guide

Use this page as the entry point to the repo docs.

## Start Here

- [`../README.md`](../README.md): project overview, build, CLI, and main features
- [`feature_status.md`](feature_status.md): authoritative compatibility matrix
- [`roadmap.md`](roadmap.md): forward-looking work tracker (CocoTB verification, SV→MATLAB lift, runtime / REPL / HDL improvements; the block language is shipped — see [`flowchart_frontend.md`](flowchart_frontend.md))
- [`../examples/README.md`](../examples/README.md): runnable sample programs

## Frontend And Sema

- [`sema.md`](sema.md): tour of `lib/Sema/` — bindings, scopes, name resolution, and type inference
- [`flowchart_frontend.md`](flowchart_frontend.md): `.mflow` graphical block-language frontend — design, architecture, and shipped phases (graph → AST → every existing backend; round-trips to MATLAB source for free)
- [`flowchart_schema.md`](flowchart_schema.md): `.mflow` JSON schema reference — field-by-field contract, every block kind's required data fields and port conventions, validation rules. **Read this when implementing the IDE save/load.**
- [`save_load_compat.md`](save_load_compat.md): plan to bring `save` / `load` to the real MATLAB API and `.mat` v5 file format

## Backends And Runtime

- [`emit_c_cpp.md`](emit_c_cpp.md): C and C++ emission design and guarantees
- [`emit_cpp_classdef.md`](emit_cpp_classdef.md): plan to translate MATLAB `classdef` to real C++ classes (replaces the runtime-hash wrapper, queued for a focused follow-up)
- [`emit_python.md`](emit_python.md): Python emission status, workflow, and limits
- [`complex.md`](complex.md): complex numbers, FFT, and DSP-oriented runtime support
- [`emit_systemverilog.md`](emit_systemverilog.md): direct RTL/SystemVerilog backend plan, including combinational, register, counter, and FSM inference
- [`emit_cocotb.md`](emit_cocotb.md): `-emit-cocotb` harness — open-source HDL Verifier alternative; cycle-by-cycle co-simulation of the SV DUT against the Python reference, with HDL-Verifier-style `Latency` parameter
- [`emit_fixed_point.md`](emit_fixed_point.md): Fixed-Point Designer (`fi`) support — Phase 1 scalar arithmetic shipped, Phase 3 arrays / FIR planned
- [`sym.md`](sym.md): Symbolic Math Toolbox (`sym` / `syms`) backed by [SymPP](https://github.com/leonardoaraujosantos/SymPP) — diff/int/simplify/solve/dsolve/pdsolve/transforms/assume/vpa/taylor/limit; opt-in via `-DMATLAB_LLVM_WITH_SYM=ON`
- [`ode.md`](ode.md): Initial-value ODE solvers — `ode45`, `ode23` (Dormand–Prince / Bogacki–Shampine, scalar and **vector `y`**) with adaptive FSAL, cubic-Hermite dense output, full odeset (`RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, `Stats`), 2-/3-return forms, forward and backward integration, user-time-grid `tspan`
- [`plotting.md`](plotting.md): headless **plot** runtime (Cairo backend, PNG/SVG/PDF output, macOS / Linux / iOS). Shipped: 2D family (`plot`, `scatter`, `bar`, `stem`, `stairs`, `area`, `errorbar`, `histogram`, `imshow`, `imagesc`, `pcolor`, `contour`, `contourf`, `quiver`); 3D family (`plot3`, `mesh`, `surf`); decoration (`title`, `xlabel`/`ylabel`/`zlabel`, `legend`, `text`, `colorbar`); layout (`subplot`, `figure`, `gcf`, `close`); axes options (`xlim`/`ylim`, `axis equal/tight/off/on`, `box`, `grid`, `hold`, `view`, `xline`/`yline`, `xticks`/`yticks`, `xticklabels`/`yticklabels`, `yyaxis`, `loglog`/`semilogx`/`semilogy`); style (Name/Value pairs `LineWidth`/`Color`/`LineStyle`/`Marker`/`MarkerSize`, auto colour cycle, 6 colormaps `gray`/`parula`/`jet`/`viridis`/`hot`/`cool`); output (`saveas`, `print`). Wired through Sema → MLIR (`LowerPlot.cpp`) → JIT. Roadmap: `polarplot`, `tiledlayout`, TeX/LaTeX, `boxplot`/`violinplot`/`pie`, lighting, `fplot`/`fmesh`/`fsurf` (need symbolic), animation. Build with `-DMATLAB_LLVM_WITH_PLOT=ON`.
- [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md): tiered plan for Control System Toolbox compatibility. **Tier-1 numeric stack + Tier-2 SISO design loop + Tier-3 state-space design + Tier-4 reduction + Tier-2 interconnection (matrix-arg) all closed** (~50 functions across the LLVM/C/C++/Python/TS lanes). Tier-1 numerics: `expm`, `hess`, `schur`, non-symmetric `eig` (1-return), `lyap`/`dlyap`, `care`/`dare`. Design: `lqr`/`dlqr` (1- and 3-return `[K, S, e]`), `place` (SISO Ackermann), `kalman_L`/`kalmd_L` + 2-return `[L, P] = kalman/kalmd`. Discretization: `c2d` (ZOH), `c2d_tustin` + `d2c_tustin` (matrix-arg). Analysis: `bode_ss` (SISO), `bode_tf`, `step_ss`, `lsim_ss`, `gain_margin` / `phase_margin`, `bandwidth_ss`, `getPeakGain_ss` (rough H∞), `gram_c`/`gram_o`, `ctrb`/`obsv`, `isstable` / `isstable_d`, `damp`, `hsvd`, `norm_h2` / `norm_h2_d`, `dcgain_ss`, `pole`, `stepinfo`. Reduction: `balreal_T`, `balred` (1- and 3-return). Interconnection (matrix-arg): `feedback_ss`, `series_ss`, `parallel_ss`, `append_ss` (all 3-return splitters). **Open follow-ons**: model objects `tf`/`ss`/`zpk`/`frd`/`pid` (biggest UX gap; would collapse most current matrix-arg primitives to model-object form); MIMO `bode_ss` (needs 3-D arrays) + `sigma`/`allmargin`; `zero(sys)` (needs QZ); exact H∞ norm via Boyd-Balakrishnan-Kabamba bisection; multi-input `place` (Kautsky-Nichols-Van Dooren); `logm`/`lyapchol`/generalised eig; `stabsep`, `minreal`, `ctrbf`/`obsvf` staircase forms; `connect`/`sumblk`/`lft` (need model objects); `pidtune` (needs H∞). All apps (Control System Designer, Linear System Analyzer, Control System Tuner, Model Reducer, PID Tuner, Linearizer), Simulink linearization, LPV/LTV simulation, sparse second-order models, `systune` / `looptune` / `hinfstruct` / `TuningGoal.*` / `genss`, and Robust Control / SysID / MPC / Adaptive Control toolbox bridges are explicitly carved out
- [`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md): tiered plan for Signal Processing Toolbox compatibility. **Tier-1 IIR/FIR design loop (LP + HP/BP/BS) + Tier-2 bulk + Tier-3 §4.1/§4.2/§4.3/§4.4 are closed** (~95 functions across the C/C++/Python/TS lanes). Tier-1: windows (17); polynomial helpers; IIR design (`butter`/`cheby1`/`cheby2` LP+HP+BP+BS via `'high'`/`'stop'` + 2-elem-Wn dispatch, `besself`, `buttord`/`cheb1ord`/`cheb2ord`); standalone analog↔digital (`bilinear` / `freqs`); form conversions (`tf2zp`/`zp2tf`/`tf2sos`/`sos2tf`); FIR (`fir1`/`sgolay`/`sgolayfilt`); filter response (`freqz`/`impz`/`stepz`/`grpdelay`); filter implementation (`filter`/`filtfilt` with steady-state ICs/`sosfilt`). Tier-2: transforms, spectral, LP + parametric PSD, time-frequency. Tier-3: real multirate (`upfirdn`/`decimate`/`interp`/`resample`), waveform generators, alignment helpers, pulse measurements (full §4.3 surface — including `statelevels`/`slewrate`/`pulseperiod`/`pulsewidth`/`overshoot`/`undershoot`/`settlingtime`). **Open follow-ons**: `ellip`/`ellipord` (§2.1, Jacobi elliptic), the analog prototype builtins as standalone 3-return entries, `tf2ss`/`ss2tf`/`zp2sos` (§2.1 state-space tail); `fir2`/`firls`/`firpm`/`firrcos`/`kaiserord` (§2.2); strict 1996 Gustafsson `filtfilt` (scipy's `method='gust'`) + `phasez`/`zerophase` (§2.5); `dpss`/`pmtm`/`czt`/`stft`/`istft`/subspace methods (Tier-2 follow-on); chirp non-linear methods + `pulstran`/`diric`/`gmonopuls`/`vco` (§4.2); `findpeaks` name-value options (§4.3); `alignsignals`/`gccphat`/xcorr scaling-options (§4.4); polyphase decomposition (§4.1); vibration analysis (§4.6); `digitalFilter` system object (§5.1); wavelets (§5.4). Apps, GUIs, deep-learning, MATLAB Coder, Python coexecution explicitly carved out

## Interactive Tooling

- [`repl.md`](repl.md): JIT-backed REPL and workspace behavior
- [`debug.md`](debug.md): DAP mode, `dbg`, and runtime debugging aids
- [`lsp.md`](lsp.md): `matlab-lsp` capabilities and editor setup

## How To Read The Status Docs

- Treat [`feature_status.md`](feature_status.md) as the source of truth for
  what is implemented, partial, or missing.
- Treat backend docs as design and behavior notes for specific codegen paths.
- Treat the examples and tests as the best concrete reference for supported
  source patterns.
