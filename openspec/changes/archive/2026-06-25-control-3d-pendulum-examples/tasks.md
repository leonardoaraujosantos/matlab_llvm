## 1. Scaffolding & shared patterns

- [x] 1.1 Create `examples/control/3d/` directory
- [x] 1.2 Confirm shipped CST builtin signatures against existing examples (`lqr(A,B,Q,R)`, `place(A,B,P)`, `ctrb(A,B)`, `obsv(A,C)`, `ss(A,B,C,D)`) and the sim3d scene-graph parenting pattern from `quadrotor_pid_mpc_3d.m`
- [x] 1.3 Settle the cart-pole nonlinear EOM + linearized `(A,B)` and the hinge geometry — used a pivot-hub (tiny sphere) + thin **box** pole (sim3d cylinders ignore Size/Scale for height); reused across the three cart-pole files

## 2. Cart-pole examples

- [x] 2.1 `cartpole_lqr_3d.m` — linearize, `K = lqr(A,B,Q,R)`, RK4 nonlinear plant, parented sim3d scene, export `cartpole_lqr_3d.html`; header documents model + design
- [x] 2.2 `cartpole_place_3d.m` — `ctrb` rank check, `K = place(A,B,P)`, `obsv` dual diagnostic, nonlinear sim, export `cartpole_place_3d.html`
- [x] 2.3 `cartpole_pid_3d.m` — classical PID on pole angle + slow cart-centering term, nonlinear sim, export `cartpole_pid_3d.html`
- [x] 2.4 Smoke-run all three interpreted (`matlabc -repl`); pole converges to upright (0.017 / -0.001 / 0.107 deg) and HTML is written

## 3. Double inverted pendulum examples

- [x] 3.1 Derive/encode the 6-state nonlinear two-link EOM (manipulator form) + linearized `(A,B)` about upright (absolute angles; A assembled by index-assignment)
- [x] 3.2 `double_pendulum_lqr_3d.m` — `K = lqr(A,B,Q,R)`, nonlinear sim, parented 3-level link chain, export `double_pendulum_lqr_3d.html`
- [x] 3.3 `double_pendulum_place_3d.m` — `ctrb` check, `K = place(A,B,P)` (6 distinct poles), nonlinear sim, export `double_pendulum_place_3d.html`
- [x] 3.4 `double_pendulum_pid_3d.m` — documented best-effort full-state PD; header states the single-input underactuation limit; export `double_pendulum_pid_3d.html`
- [x] 3.5 Smoke-run all three interpreted; both links stabilize from a small perturbation (0.11 / 0.01 / 0.25 deg) and HTML is written

## 4. Documentation

- [x] 4.1 Write `examples/control/3d/README.md` — plant equations, six-program controller matrix, run command + emitted HTML names, nonlinear-plant/linearized-design note, double-PID limitation, interpreter gotchas
- [x] 4.2 Add cross-link in `examples/sim3d/README.md` to the applied control-3d suite
- [x] 4.3 Add a pointer section in `examples/control/README.md` to the `3d/` subdir

## 5. Validation & wrap-up

- [x] 5.1 Verify only files under `examples/` (plus this OpenSpec change) were added/edited — no runtime/compiler/viewer changes
- [x] 5.2 Confirm the parent chain + animated hub rotations in an exported scene (cart→hub0→hub1→link2; rotations converge to upright)
- [x] 5.3 `openspec validate control-3d-pendulum-examples`; then archive the change
