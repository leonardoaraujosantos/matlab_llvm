# Toolbox Tutorials

Hands-on, example-driven tutorials for every toolbox surface shipped in
matlab_llvm. Each tutorial is grounded in the runnable programs under
`examples/<toolbox>/` — the code excerpts are faithful to real files that
compile and run end-to-end through the normal pipeline.

For the canonical compile-and-run flow see [`../build_and_run.md`](../build_and_run.md).
The high-level inventory of what the compiler supports lives in
[`../feature_status.md`](../feature_status.md); the per-toolbox design notes
and carve-outs live in the `*_roadmap.md` docs one level up.

Most tutorials use the default LLVM lane:

```bash
build/matlabc -emit-llvm examples/<dir>/<name>.m > /tmp/<name>.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/<name>.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/<name>
/tmp/<name>
```

The HDL / Stateflow / Embedded-Coder tutorials use the SystemVerilog,
cocotb, and `.mflow` emit modes instead — see those pages for the exact
invocations.

## Start here

- [The MATLAB Language](matlab_language_tutorial.md) — the core `.m` subset: matrices, control flow, functions/handles, `classdef`, plotting, emit backends
- [`.mflow` Flowchart Frontend](mflow_tutorial.md) — the graphical block-language family (flowchart / signal_flow / state_chart dialects)

## Signal, communications & DSP

- [Signal Processing](signal_tutorial.md) — filter design, spectral estimation, peaks/alignment
- [DSP System](dsp_tutorial.md) — `dsp.*` streaming System Objects, adaptive/multirate, HDL lane
- [Communications](comm_tutorial.md) — modulation, channel coding, OFDM/Alamouti, BER chains
- [RF & Propagation](rf_propagation_tutorial.md) — S-parameters, path-loss/ITM, coverage maps
- [Antenna](antenna_tutorial.md) — thin-wire dipole solve, patterns, Touchstone export
- [Wavelet](wavelet_tutorial.md) — `dwt`/`wavedec`, CWT scalograms, MODWT, denoising

## Control & estimation

- [Control System](control_tutorial.md) — `tf`/`ss`, LQR/LQG, `bode`/`margin`, model reduction
- [Model Predictive Control](mpc_tutorial.md) — linear `mpc` and `nlmpc`
- [System Identification](ident_tutorial.md) — `arx`/`armax`/`n4sid`/`ssest`, recursive estimation
- [Sensor Fusion & Tracking](sensor_fusion_tutorial.md) — KF/EKF/UKF, `trackerGNN`, IMU/GPS fusion
- [Navigation](navigation_tutorial.md) — occupancy maps, RRT planning, MCL, GNSS, Frenet
- [Robotics System](robotics_tutorial.md) — `rigidBodyTree`, inverse kinematics, mobile kinematics

## Optimization & math

- [Optimization](optim_tutorial.md) — `fmincon`/`linprog`/`quadprog`, problem-based API
- [Global Optimization](global_optim_tutorial.md) — `ga`/`particleswarm`/`surrogateopt`, Pareto
- [Curve Fitting](curve_fitting_tutorial.md) — polynomial/nonlinear library fits, splines
- [Symbolic Math](symbolic_tutorial.md) — SymPP-backed `syms`/`diff`/`int`/`solve`
- [Partial Differential Equation](pde_tutorial.md) — 2-D elliptic, 3-D structural FEM

## Data, stats & finance

- [Statistics & Machine Learning](stats_ml_tutorial.md) — distributions, PCA/kmeans, classifiers
- [Econometrics](econ_tutorial.md) — ARIMA/GARCH/VAR/state-space, Bayesian regression
- [Financial](finance_tutorial.md) — efficient frontier, Black-Litterman, option pricing, backtest
- [Image Processing](image_tutorial.md) — PNG/JPEG I/O, filtering, morphology, `regionprops`

## Deep learning & acceleration

- [Deep Learning](deep_learning_tutorial.md) — `dlarray`/`dlgradient` autodiff, CNN/LSTM/Transformer
- [GPU Coder / gpuArray](gpu_tutorial.md) — `gpuArray`, `arrayfun` kernels, multi-GPU `parfor`
- [Fixed-Point Designer](fixed_point_tutorial.md) — `fi`/`numerictype`, quantization → HDL

## Hardware & model-based design

- [HDL Coder](hdl_coder_tutorial.md) — MATLAB → SystemVerilog + cocotb bit-accuracy verification
- [Stateflow / mStateflow](stateflow_tutorial.md) — hierarchical `.mflow` state charts
- [Embedded Coder / mflowLink](embedded_coder_tutorial.md) — block-diagram codegen + cocotb SIL
