# examples/control/

Self-contained programs that exercise Control System Toolbox
functionality shipped (and being shipped) by `matlab_llvm`. Each
example synthesises its plant inline and prints diagnostic output, so
they double as reading-order tours of the CST surface.

Unlike `examples/signal/`, this directory carries **forward-looking**
examples: many of them depend on tiers from
[`../../docs/control_toolbox_roadmap.md`](../../docs/control_toolbox_roadmap.md)
that have not all landed yet. Each file's header comment names the
**Tier** it requires; examples become live as their tier ships.

Run an example whose tier has shipped with:

```sh
runtime/build_and_run.sh examples/control/<name>.m /tmp/<name>
/tmp/<name>
```

| File | Tier | Demonstrates |
|---|---|---|
| `expm_basic.m` | 1.3 ✅ | `expm(A)` matrix exponential — rotation matrix, free-response state evolution `x(t) = expm(A·t)·x0`, and the augmented-matrix trick used by `c2d` ZOH. The first Tier-1 primitive shipped. |
| `eig_poles_demo.m` | 1.1 ✅ | `eig(A)` for non-symmetric state matrices — mass-spring-damper open-loop poles (lightly-damped complex pair), inverted pendulum (real ±√(g/L) pair), discrete-time stability check via `expm(A·Ts)` and `\|eig(Ad)\|<1`. Demonstrates the polymorphic real/complex return: real plants with damping return complex `matlab_mat_c` carrying both halves of each pole pair. |
| `schur_modal_split.m` | 1.2 ✅ | `schur(A)` and `[U, T] = schur(A)` — real Schur decomposition via Hessenberg + Francis QR with the orthogonal accumulator. Demonstrates trace preservation, U' U = I orthogonality, A = U T U' reconstruction, and the connection to `eig(A)`. Schur form is the launch pad for Bartels-Stewart Lyapunov (Tier 1.4) and ordered-Schur Riccati (Tier 1.5). |
| `lyap_gramian.m` | 1.4 ✅ | `lyap(A, Q)` and `dlyap(A, Q)` for the canonical control use — controllability/observability gramians of a stable plant. Mass-spring-damper plant: builds `Wc = lyap(A, B*B')` and `Wo = lyap(A', C'*C)`, verifies the Lyapunov residual is machine-zero, then repeats for the discretised plant via `expm` + `dlyap`. Gates `gram` (Tier 3.4), `norm(sys, 2)` (Tier 3.5), and balanced realisation (Tier 4.1). |
| `lqr_via_care.m` | 1.5 ✅ | `care(A, B, Q, R)` then `K = R⁻¹ B' X` then `Acl = A - B K` — the full continuous-time LQR pipeline using primitives from Tier 1.3 / 1.5. Double-integrator example matches closed-form `X = [√3, 1; 1, √3]` and `K = [1, √3]`; closed-loop poles land at `−√3/2 ± j/2`. Riccati residual on a second 2×2 plant is ~10⁻¹⁵. |
| `dare_dlqr_demo.m` | 2 ✅ | Discrete-time LQR pipeline via `dare(Ad, Bd, Q, R)` and `dlqr(Ad, Bd, Q, R)` — Newton-Kleinman iteration seeded from `X₀ = dlyap(Ad', Q)`. Three plants: a diagonal Schur-stable 2×2 (per-axis closed form), a c2d-discretised mass-spring-damper, and a 2×2 plant for residual self-consistency (`A' X A − X − A' X B (R + B' X B)⁻¹ B' X A + Q ≈ 0` to round-off). Documents the limitation that Newton-Kleinman seeded from `X₀ = dlyap(Ad', Q)` requires Ad already Schur-stable. |
| `h2norm_demo.m` | 3 ✅ | H₂ system norm via `norm_h2(A, B, C)`. Closed-form sanity (`‖1/(s+1)‖_2 = 1/√2`), monotone-decreasing damping sweep, LQR closed-loop reduces the H₂ norm versus open-loop, unstable plant returns `+Inf`. Sits cleanly on Tier-1.4 lyap and the `gram_c` / `gram_o` companion. |
| `balred_demo.m` | 4 ✅ | Balanced model reduction. 4-state plant where two HSVs collapse to ~1e-7; `balred_A`/`balred_B`/`balred_C` truncate to 2 states. Verifies stability preservation, exact match of dominant HSVs, and damping-ratio preservation of the kept mode. Computes the H∞ error bound `2·sum(HSV[3:4]) ≈ 1.37e-6`. |
| `balreal_demo.m` | 4 ✅ | Balanced realization workflow. `balreal_T(A, B, C)` returns the similarity transform that makes the controllability and observability gramians equal and diagonal (with diagonal = HSVs descending). Verifies the balanced invariant `Wcb = Wob = diag(HSV)` to round-off on a lightly-damped mass-spring-damper. The structural foundation for balanced model reduction. |
| `charac_triad.m` | 3 ✅ | Model characterization triad: `isstable(A)` / `damp(A)` / `hsvd(A, B, C)`. Three plants — open-loop mass-spring-damper (lightly damped, ζ = 0.05), the same plant after LQR (stiffer, larger ζ), and a redundant 4-state plant where two of the four Hankel singular values collapse to round-off (the diagnostic that flags reducible states). Demonstrates the Tier-1.4-gramian + Tier-1.1-eig stack as the substrate for system characterization. |
| `place_pole_assignment.m` | 3 ✅ | Pole-placement workflow — the user-facing alternative to LQR. Uses `ctrb(A, B)` for the controllability test, `place(A, B, P)` to design the gain (SISO Ackermann), `gram_c(Acl, B)` as the energy-based companion, and `obsv(A, C)` for the observability check. Linearised inverted pendulum: open-loop has one positive real eigenvalue at `+√(g/L) ≈ +3.13`; closed-loop poles assigned to `{-2, -2}`. Demonstrates the structural-rank vs. energy-gramian duality. |
| `lqr_discrete_workflow.m` | 2 ✅ | Full continuous→digital LQR pipeline using `lqr(A, B, Q, R)` and `[Ad, Bd] = c2d(A, B, Ts)` (Tier 2.4 + Tier 2.2). Designs continuous LQR for the double integrator, discretises both plant and closed-loop matrix at `Ts = 0.05` s via Van Loan ZOH, simulates the discrete recurrence `x[k+1] = Ad_cl x[k]` from `x₀ = [1; 0]`. Position decays monotonically under LQR. |
| `h2norm_via_gramian.m` | 3.4 ✅ | `gram_c(A, B)` and `gram_o(A, C)` for the mass-spring-damper, with the H₂ norm computed two ways (`√(B' Wo B)` and `√(C Wc C')`) — they must match. Plus `step_ss(A, B, C, D, dt, N)` to verify the impulse response converges to the DC gain `1/k`. Demonstrates Tier 3 analysis primitives sitting cleanly on Tier 1.4 (lyap) and Tier 2.2 (c2d). |
| `bode_demo.m` | 2.4 ✅ | `[mag, phase] = bode_ss(A, B, C, D, w)` for the lightly-damped mass-spring-damper. Spot-checks at four frequencies, DC gain matches `−C A⁻¹ B = 1/k`, and the resonant peak at `wp = wn·√(1−2ζ²) ≈ 2.97 rad/s` matches the closed-form prediction `1/(k·2ζ·√(1−ζ²)) = 0.558` to 6 decimals. |
| `loop_shaping_workflow.m` | 2.4 ✅ | Full SISO loop-shaping pipeline on the type-1 servo `L(s) = 4/(s(s+2))` — frequency-response checkpoints, `gain_margin` / `phase_margin` (Pm = 51.83° matches closed form), `c2d` ZOH discretisation at `Ts = 0.1 s`, and `lsim_ss` open-loop step response. End-to-end CST workflow demonstrated without model objects. |
| `bode_tf_filter.m` | 2.4+ ✅ | `bode_tf(b, a, w)` for the analog first-order lowpass and a biquad notch filter (the notch correctly drops to zero at the resonance). Verifies `bode_tf` and `bode_ss` agree exactly for the same plant in alternate representations. Bridges SPT-designed filter coefficients (`butter`/`cheby1`/`cheby2` return `(b, a)`) to CST-style frequency-response analysis. |
| `tf_basic.m` | 2.1 🔵 | `tf(num, den)` constructor, `disp(G)` canonical s-domain rendering, `G + H` (parallel) / `G * H` (series), `s = tf('s')` variable-builder. |
| `step_response_siso.m` | 2.3 🔵 | `step(G)` and `step(G, t)` for SISO `tf` and `ss`. First-order, second-order underdamped, double-integrator — closed-form against analytic answers. |
| `bode_first_order.m` | 2.4 🔵 | `[mag, phase, w] = bode(G)` for `H(s) = 1/(τs+1)`; checks the −3 dB corner and the high-frequency −20 dB/dec roll-off; `margin(G)` for an open-loop type-1 system. |
| `c2d_zoh_demo.m` | 2.2 🔵 | `c2d(G, Ts, 'zoh')` and `c2d(G, Ts, 'tustin')`. Discretises a continuous PI controller, compares the two methods' `step` responses against the original at the sampling instants. |
| `lqr_double_integrator.m` | 3.1 🔵 | `[K, S, e] = lqr(A, B, Q, R)` on the canonical double integrator. Verifies the closed-loop poles match the symmetric root-locus prediction; simulates the regulated response from a non-zero initial condition. |
| `kalman_tracker.m` | 3.2 🔵 | `[kest, L, P] = kalman(sys, Q, R)` for a constant-velocity tracker. Estimator converges from a noisy position-only measurement; plots-via-`disp` the steady-state Kalman gain. |

For the full CST surface, the tier ordering, and what's deliberately
carved out (apps, Simulink linearization, LPV/LTV simulation, sparse
second-order, `systune`/`looptune`/`hinfstruct`, Robust/SysID/MPC
toolbox bridges), see
[`../../docs/control_toolbox_roadmap.md`](../../docs/control_toolbox_roadmap.md).

## Notes

- These are demonstration programs, not regression tests. The
  authoritative CST regression corpus lives at
  [`../../test/Run/`](../../test/Run/) under `linalg_*.m` (Tier-1
  primitives) and `ctrl_*.m` (Tier-2+ user-visible surface).
- Examples whose tier has not shipped will fail with a "name not
  recognized" error from Sema today — that is by design; the file
  documents the intended API now and becomes live as the primitive
  lands.
- A few examples use `disp(scalar)` instead of `fprintf('%.3f',
  scalar)` to keep stdout byte-stable across the C / C++ / Python / TS
  emit lanes (Python prints numpy arrays with `[[ ]]` brackets where
  C prints a bare scalar; the numpy override path is the same one SPT
  uses).
