# Reinforcement Learning Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Reinforcement-Learning-Toolbox programs.

Source: *Reinforcement Learning Toolbox™ User's Guide* (R2026a, 1448 pp,
9 chapters: Getting Started · Create Environments · Create Agents ·
Actors/Critics/Policy Objects · Train and Simulate Agents · Deploy Trained
Policies · Benchmark Examples · plus the function/object reference).

This toolbox sits **directly on top of the just-shipped Deep Learning
Toolbox** ([`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md)).
That dependency is the single most important fact for scoping, because of
what Deep Learning shipped along the way.

---

## 0. Status — Tiers 1–4 SHIPPED; deep agents reuse the dlnet tape (2026-05-29 → 05-30)

**T2 (control-env infra) + T3 (DQN) + T4 (REINFORCE) shipped on top of T1.**
The keystone decision: deep agents do **not** re-implement any autodiff — the
RL runtime builds each actor/critic forward pass as `dlarray` shells
(`matlab_obj_new`) and drives the **shipped Deep Learning Toolbox tape**
(`matlab_dlnet_mtimes/plus/relu/softmax/log/sum/mse` + `matlab_dlnet_grad`)
directly from C++. Only the Adam moment update + replay/episode orchestration
live RL-side. Zero dlnet changes, zero forward/backward duplication.

- **T2** — `rlPredefinedEnv("CartPole-Discrete")` continuous-state env
  (Barto cart-pole dynamics in the runtime); `getObservationInfo` returns a
  continuous (dimension-carrying) spec for `Kind!=0` envs; greedy `sim`
  rollout; **policy-use accessors `getAction(agent,obs)` / `getMaxQValue` and
  `getGreedyPolicy(agent)` → `rlMaxQPolicy`** (net + tabular dispatch).
  Example [`examples/rl/use_trained_policy.m`](../examples/rl/use_trained_policy.m),
  test `rl_getaction.m`.
  **T2 carve-downs:**
  - **`rlFunctionEnv(obs,act,@step,@reset)` — BLOCKED** on two confirmed
    frontend gaps (filed): multi-return through a function handle returns the
    wrong outputs (**bug #80**) and calling a handle stored in a struct/object
    field is unsupported (**#81**). The faithful `[NextObs,Reward,IsDone,Info]
    = step(env,a)` form (a property-stored handle called with 4 returns) needs
    both; revisit once #80/#81 land.
  - Additional predefined envs (pendulum/double-integrator) + the
    exploration-noise policy objects (`rlEpsilonGreedyPolicy` /
    `rlAdditiveNoisePolicy` OU) are deferred — the noise policies pair
    naturally with the T5 continuous-control agents.
- **T3** — `rlDQNAgent(obsInfo, actInfo)` auto-builds a 24-unit MLP critic;
  experience replay + target network + epsilon-greedy + TD-error loss
  (`mse`) → `dlgradient` → Adam. Headline
  [`examples/rl/cartpole_dqn.m`](../examples/rl/cartpole_dqn.m): the greedy
  policy balances the pole **269 steps** (random ≈10–20). Test `rl_dqn.m`.
- **T4** — `rlPGAgent(obsInfo, actInfo)` softmax-policy actor; REINFORCE with
  discounted normalized returns, `−Σ logπ·Ĝ` loss on the tape
  (`softmax`→`log`→masked `sum`). Headline
  [`examples/rl/cartpole_reinforce.m`](../examples/rl/cartpole_reinforce.m):
  **391 steps**. Test `rl_reinforce.m`.

**Deep-RL traps (carry into T5):**
- The dlnet tape **leaks per grad-call** — forward values on `reset`, orphaned
  adjoints across multi-param grad calls, and backward contribution
  temporaries (the full picture is **issue #82**). Fine for DQN/PG (few grad
  calls), but it caps training scale; surgical frees cause use-after-free in
  the conv/CNN backward paths, so a real fix needs a tape ownership/arena
  model. Adam moments + target-net copies are stored as dynamic matrix fields
  on the agent object.
- **`matlab_obj_get_mat` returns a non-null EMPTY (0×0) matrix for an unset
  field** (not `nullptr`) — so "is this network present?" must test
  `m && m->rows > 0`, never the pointer. (This footgun made `getAction`'s DDPG
  branch wrongly fire for DQN agents → a 0×1 empty result that printed as a
  literal `%.0f`.)

**T5 (continuous control) — DDPG IMPLEMENTED + wired, but BLOCKED from shipping
a demo.** `rlPredefinedEnv("Pendulum-Continuous")` (swing-up dynamics, Kind=3,
continuous-action `getActionInfo`), `rlDDPGAgent(obsInfo,actInfo)` (auto-built
deterministic actor `obs→H→1 tanh·limit` + Q(s,a) critic `[obs;act]→H→1`,
target copies + soft `tau` updates, OU exploration), the critic TD-MSE step
and the **deterministic-policy-gradient actor step (actor gradient flows
through the critic via tape `vertcat([s; actor(s)])`, grad w.r.t. actor params
only)** are all built and run end-to-end on the reused tape. `getAction`
extended to return the continuous action.

**Two blockers stop a shippable demo:**
1. **Autodiff-tape training-scale memory (issue #82).** The tape leaks per
   grad-call (forward vals on reset, orphaned adjoints across multi-param grad
   calls, backward contribution temporaries). Fine for one-shot DL / DQN+PG
   (few grad calls), but DDPG (100 ep × 200 steps × ~8 grad calls) peaks at
   ~18 GB. Surgical frees cause use-after-free in the conv/CNN backward paths
   (6 `dl_cnn_*` crashes) — a correct fix needs a tape ownership/arena model,
   tracked in #82.
2. **Convergence tuning.** Even bounded, DDPG pendulum swing-up needs more
   episodes + tuning than the memory ceiling currently allows (the bounded run
   stays near the untrained return).

So the DDPG implementation lives in `runtime/toolbox/rl/` (builds, wired) but
ships **no example/test** until #82 lands. TD3/SAC/PPO (twin critics / entropy
/ clipped surrogate) layer on top once DDPG demos cleanly.

---

## 0b. Status — Tier 1 SHIPPED (2026-05-29)

**Tier 1 (tabular) is implemented, wired, and regression-green.** The runtime
lives in `runtime/toolbox/rl/` (`runtime_rl.cpp` + `rl_classdefs.m`):

- **Environments**: `rlPredefinedEnv("BasicGridWorld")` (canonical 5×5 grid,
  built in the runtime) + `rlMDPEnv(nextState, reward)` direct deterministic-MDP
  builder (terminals self-loop; auto-detected).
- **Specs / approximators**: `rlFiniteSetSpec`, `rlNumericSpec`,
  `getObservationInfo`/`getActionInfo`, `rlTable`, `rlQValueFunction`.
- **Agents**: `rlQAgent` (Q-learning), `rlSARSAAgent` (SARSA) — the full
  epsilon-greedy TD training loop runs in the runtime over the env's
  next-state/reward tables (no NN, no autodiff). `rlQAgentOptions`/
  `rlSARSAAgentOptions`/`rlOptimizerOptions`/`rlTrainingOptions`/
  `rlSimulationOptions` carriers.
- **Drivers**: `train(agent, env, opts)` (mutates the agent's Q in place,
  returns per-episode rewards), `sim(agent, env)` (greedy rollout → cumulative
  reward), `getCritic`/`getLearnableParameters`.

**Examples** (`examples/rl/`): `gridworld_qlearning.m` (Q-learning + SARSA both
reach the optimal **11.0**, matching the MathWorks `StopTrainingValue`) and
`mdp_qlearning.m` (8-state MDP, optimal **13.0**). **Gating tests**
`test/Run/rl_gridworld.m` + `rl_mdp.m`. Full Run-test suite green
(658 → 660), zero regressions.

**Wiring touchpoints** (for the later tiers): `CMakeLists.txt` (runtime TU ×2
spots) · `test/Run/run_tests.sh` `RUNTIME_SRCS` · `lib/Sema/Resolver.cpp`
(builtin names + factory→class pins in `pinnedOfRhs`: `rlPredefinedEnv`→
`rlMDPEnv`, `getObservationInfo`/`getActionInfo`→`rlFiniteSetSpec`,
`getCritic`→`rlQValueFunction`) · `lib/MLIR/Lowering.cpp` (free-fn intercept
for `rlPredefinedEnv`; constructor intercepts for `rlMDPEnv`/`rlTable`/
`rlQValueFunction`/`rlQAgent`/`rlSARSAAgent`; method dispatch for
getObservationInfo/getActionInfo/train/sim/getCritic/getLearnableParameters) ·
`lib/MLIR/Passes/LowerTensorOps.cpp` (`pde_table` `matlab_rl_*` rows) ·
`tools/matlabc/main.cpp` (**two** prelude paths — the REPL `buildReplPrelude`
Want-table **and** the AOT `userMentionsExtClasses` `Names[]` + `extClassLeaf`
+ `kToolboxDirs`; both must list `rl`).

**Tier-1 traps discovered** (carry into later tiers):
1. **No char-string classdef properties** — `obj.Name = 'spec'` lowers to
   `matlab_obj_set_mat` with an `i8` char tensor → "unsupported call shape".
   Tier-1 specs/options drop all string fields. (Numeric/struct/matrix props
   are fine.) General frontend gap → **issue #79**.
2. **No-paren constructor** `x = rlTrainingOptions;` does not lower — use
   `rlTrainingOptions()`. General frontend gap (the prelude scanner only
   detects `Name(` / `Name =`, not a bare RHS classname) → **issue #79**.
3. **Bare call discarding an object-returning result** — `train(a,e,o);` as a
   statement left an unconverted value op; assign it (`s = train(...)`, which
   matches the MathWorks `qTrainingStats = train(...)` form anyway). Verified
   **not** a general pre-existing gap (bare `getOccupancy(m,xy);` lowers fine),
   so RL-specific and simply worked around — not filed.
4. **Builtin factory results need a `pinnedOfRhs` entry** to pin their class
   (classdef-ctor calls auto-pin; plain builtins like `rlPredefinedEnv` don't).
5. **3-D indexed property/field store** `obj.T(i,j,k)=v` is unsupported
   (`__subscript_store: 5 arguments`) — the reason the verbatim
   `createMDP`+`mdp.T(...)=` MDP form is carved and the deterministic
   `rlMDPEnv(NS,RW)` table builder is used instead. General frontend gap →
   **issue #78**.

**Remaining T1 increment (carved):** the `rlFunctionEnv(obs,act,@step,@reset)`
custom-environment example (MathWorks "Create MATLAB Environments Using Custom
Functions") needs the function-handle env callback (`reset`/`step` invoking the
user's stored handles) + multi-output `[NextObs,Reward,IsDone,Info]=step(...)`
— a distinct piece of work from the tabular core. Plus `createMDP` + 3-D
indexed-property assignment for the verbatim MDP form.

---

## 1. The one architectural fact that shapes everything: the autodiff keystone is **already cleared**

When the Deep Learning roadmap was drafted, its thesis was that *training*
rested on **reverse-mode automatic differentiation** (`dlarray`/`dlgradient`)
which the project **lacked** — the keystone gate for everything that learns.

**That gate is now closed.** `runtime/toolbox/dlnet/runtime_dlnet.cpp` ships
a working **reverse-mode tape**: ops record onto a thread-local tape, and
`dlgradient(loss, var)` sweeps it backward to produce gradients, with
conv/LSTM/gemm backward legs (Metal-accelerated GEMM) all wired. The
stochastic solvers (Adam/SGDM/RMSProp via `rlOptimizerOptions`-style updates)
and the dense layer forward/backward library ride on top.

This matters enormously for RL, because **every deep RL agent's update is
"build a scalar loss from a network forward pass, call `dlgradient`, step the
optimizer."** DQN minimises a TD error; PPO maximises a clipped surrogate;
DDPG follows the deterministic-policy gradient; SAC adds an entropy bonus.
All of them are *just specific loss functions over the shipped autodiff
engine.* The hard, cross-cutting infrastructure RL would otherwise have had
to invent **does not need to be built — it is reused wholesale.**

So unlike Deep Learning (where T2 autodiff was the project's largest single
new piece of machinery), **Reinforcement Learning adds almost no new
numerical kernel.** What it adds is *scaffolding*:

**The net-new surface** (everything RL adds *beyond* the shipped DL +
matrix + ODE + RNG + classdef base) is:

1. **The environment abstraction** — a uniform `(observation, reward,
   isdone) = step(env, action)` / `obs = reset(env)` contract, with
   observation/action **spec** objects (`rlNumericSpec`, `rlFiniteSetSpec`),
   realised by (a) tabular **MDP / grid-world** environments
   (`createGridWorld`/`GridWorld`, `rlMDPEnv`), (b) **predefined
   control-system** environments (cart-pole, pendulum, double-integrator —
   each a small ODE the project can already integrate with `ode45`), and
   (c) **custom function** environments (`rlFunctionEnv` wrapping user
   step/reset handles).
2. **The agent objects** — value-based (`rlQAgent`, `rlSARSAAgent`,
   `rlLSPIAgent`, `rlDQNAgent`), policy-gradient (`rlPGAgent`, `rlACAgent`),
   advanced PG (`rlPPOAgent`, `rlTRPOAgent`), continuous control
   (`rlDDPGAgent`, `rlTD3Agent`, `rlSACAgent`), and model-based
   (`rlMBPOAgent`). Each is a classdef carrying actor/critic approximators,
   target copies, exploration state, and an update rule.
3. **The actor/critic approximator wrappers** — `rlValueFunction`,
   `rlQValueFunction`, `rlVectorQValueFunction`,
   `rlContinuousDeterministicActor`, `rlDiscreteCategoricalActor`,
   `rlContinuousGaussianActor` — thin adapters over either a **table**
   (`rlTable`), a **custom basis function** (a function handle), or a
   **`dlnetwork`** (the shipped DL object).
4. **The training & simulation drivers** — `train`, `sim`,
   `rlTrainingOptions`, `rlSimulationOptions`, the experience buffer
   (`rlReplayMemory`), exploration schedules, and the policy objects
   (`rlMaxQPolicy`, `rlEpsilonGreedyPolicy`, `rlDeterministicActorPolicy`,
   `rlStochasticActorPolicy`, `rlAdditiveNoisePolicy`).
5. **The RL-specific losses** — TD-error, n-step/GAE advantage, the clipped
   PPO surrogate, the deterministic-policy gradient, and the SAC
   max-entropy objective — each ~10–30 lines of MATLAB-or-runtime code that
   builds a scalar and hands it to `dlgradient`.

**One architectural split mirrors the Deep Learning roadmap:** the tabular
and least-squares agents (Q-learning, SARSA, LSPI, custom-basis) need **no
autodiff at all** — they are closed-form table/linear updates over the
matrix kernel. The deep agents (DQN onward) ride the autodiff tape. So,
exactly as DL split inference-vs-training, RL splits **tabular (feasible on
the bare kernel) vs. deep (rides the now-shipped autodiff engine)**, and the
roadmap is ordered tabular-first so a self-contained, demoable slice ships
before any network is trained.

**What the project already ships that RL composes on:**

- **`dlnetwork` + the autodiff tape + Adam/SGDM/RMSProp** (Deep Learning) —
  the actor/critic networks and *every* deep-agent gradient step.
- **`ode45` + the seeded PRNG** — the predefined control-system environment
  dynamics and all stochastic exploration / experience sampling.
- **The function-handle ABI** — `rlFunctionEnv` step/reset handles, custom
  basis functions, and custom-loss handles in custom training loops.
- **The classdef + persistent-state machinery** — every agent, environment,
  approximator, and options object is a classdef carrying mutable state
  (replay buffers, target-network copies, exploration counters), the exact
  pattern proven by the stateful System-Object toolboxes (DSP/Fusion) and
  the `extendedKalmanFilter`/`trackerGNN` precedent.
- **Optim** (`fmincon`/`lsqlin`/conjugate-gradient) — the TRPO trust-region
  line search and the LSPI least-squares projection.
- **Cairo plotting** — the Reinforcement Learning Training Monitor's
  episode-reward / average-reward / Q-estimate curves.
- **`matlab_horzcat`/`vertcat`, cell arrays, structs** — experience tuples
  `(S, A, R, S', isdone)` and multi-output `[obs, rwd, done, info]` returns.

**No external dependency** — no Gym/Gymnasium, no Stable-Baselines, no
RLlib. Every environment, agent, replay buffer, and update rule is
hand-coded over the shipped kernel, the same self-contained posture as every
other shipped toolbox.

The headline tracer-bullets:

- **Tabular (closes T1)** — `examples/rl/gridworld_qlearning.m`: *the
  canonical "Train RL Agent in Basic Grid World" demo — `createGridWorld` →
  `rlMDPEnv` → `rlQAgent` with an `rlTable` Q-critic → `train` for N episodes
  → the learned greedy policy reaches the terminal state with the optimal
  return.* Exercises spec → environment → tabular agent → training loop
  end-to-end with **zero neural network and zero autodiff.**
- **Deep value-based (closes T3)** — `examples/rl/cartpole_dqn.m`: *the
  flagship "Train DQN Agent to Balance Cart-Pole" demo — `rlPredefinedEnv`
  cart-pole → a `dlnetwork` `rlVectorQValueFunction` critic → `rlDQNAgent`
  with `rlReplayMemory` + target network + epsilon-greedy → `train` →
  `sim` the trained agent balances the pole.* This is the demo that proves
  the autodiff tape carries a full deep-RL update.
- **Continuous control (closes T5)** — `examples/rl/pendulum_ddpg.m`:
  *"Train DDPG Agent to Swing Up and Balance Pendulum" — continuous
  `rlContinuousDeterministicActor` + `rlQValueFunction` critic + OU action
  noise (`rlAdditiveNoisePolicy`) → deterministic-policy-gradient update.*

---

## 2. Tiered plan

Ordered **tabular → predefined-environments/simulate → deep value-based →
policy-gradient/actor-critic → continuous-control/advanced → training-infra
& deploy**. Each tier is independently shippable and closes a demoable slice.

### Tier 1 — Spec + environment scaffolding + tabular agents (no autodiff)

The foundation: the observation/action **spec** types, the **MDP/grid-world**
environment, the **table** approximator, and the two **tabular** agents whose
updates are closed-form (no network, no gradient).

- **Specs**: `rlNumericSpec` (continuous, with `Dimension`/`LowerLimit`/
  `UpperLimit`), `rlFiniteSetSpec` (discrete, with `Elements`);
  `getObservationInfo`/`getActionInfo`.
- **Environment**: `createGridWorld` → `GridWorld` (states/actions/`T`
  transition tensor/`R` reward tensor/`TerminalStates`/`ObstacleStates`),
  `rlMDPEnv`, `rlPredefinedEnv("BasicGridWorld")` and the two Waterfall
  variants. The `reset`/`step` ABI returning `[obs, reward, isdone, info]`.
- **Approximators**: `rlTable`, `rlQValueFunction`/`rlVectorQValueFunction`
  *backed by a table*, `rlValueFunction` backed by a table.
- **Agents**: `rlQAgent` (Q-learning, off-policy TD), `rlSARSAAgent`
  (on-policy TD) + their `*AgentOptions`.
- **Training**: a `train` driver for tabular agents (episode loop, epsilon
  decay, `rlTrainingOptions` core fields `MaxEpisodes`/`MaxStepsPerEpisode`/
  `StopTrainingCriteria`), returning a training-result struct.
- **Headline**: `gridworld_qlearning.m` (Basic Grid World, Q-learning).

### Tier 2 — Predefined control environments + simulate + policy objects + custom function envs

Makes environments *rich* (continuous ODE dynamics) and adds the
simulation/rollout path and the exploration-policy objects that the deep
agents will reuse.

- **Predefined control envs**: `rlPredefinedEnv` for
  `"CartPole-Discrete"`/`"CartPole-Continuous"`,
  `"SimplePendulum*"`, `"DoubleIntegrator-*"` — each a small ODE integrated
  with the shipped `ode45`/fixed-step Euler; reward + termination logic.
- **Custom function env**: `rlFunctionEnv(obsInfo, actInfo, @stepFcn,
  @resetFcn)` — wrap user-supplied step/reset handles via the function-handle
  ABI.
- **Simulate**: `sim(agent, env, rlSimulationOptions)` returning an
  experience struct (`Observation`/`Action`/`Reward`/`NextObservation`/
  `IsDone`); `rlSimulationOptions(MaxSteps, NumSimulations)`.
- **Policy objects**: `rlMaxQPolicy`, `rlEpsilonGreedyPolicy`,
  `rlDeterministicActorPolicy`, `rlStochasticActorPolicy`,
  `rlAdditiveNoisePolicy` (OU + Gaussian noise models); `getAction`,
  `getGreedyPolicy`/`getExplorationPolicy`.
- **Accessors**: `getActor`/`setActor`, `getCritic`/`setCritic`,
  `getLearnableParameters`/`setLearnableParameters`, `getValue`,
  `getMaxQValue`.

### Tier 3 — Deep value-based: DQN (+ LSPI) — first autodiff-backed agent

The first agent that trains a **`dlnetwork`** critic through the autodiff
tape, plus the least-squares LSPI agent (autodiff-free, basis-function).

- **Deep critics**: `rlQValueFunction`/`rlVectorQValueFunction` backed by a
  `dlnetwork`; deep `rlValueFunction`.
- **`rlDQNAgent`** + `rlDQNAgentOptions`: experience replay
  (`rlReplayMemory` — circular buffer, mini-batch sampling), **target
  network** (periodic/soft update), **epsilon-greedy** exploration schedule,
  the TD-error loss `(R + γ·maxₐ' Qtarget(S',a') − Q(S,A))²` → `dlgradient` →
  Adam step. Double-DQN + dueling options.
- **`rlOptimizerOptions`** (learn rate, gradient threshold, L2) — plus the
  `rlActorOptimizerOptions`/`rlCriticOptimizerOptions` variants and the
  `rlOptimizer` object — feeding the shipped Adam/SGDM.
- **`rlAgentInitializationOptions`** — the "default-agent" path that
  auto-sizes actor/critic `dlnetwork`s from the obs/action specs (hidden-unit
  count, normalization), the basis for `rlDQNAgent(obsInfo, actInfo)` without
  hand-built networks.
- **`rlLSPIAgent`** + options: least-squares policy iteration over a custom
  basis function — closed-form `lsqlin`/normal-equations solve, **no
  autodiff** (parallels the tabular tier but with linear function approx).
- **Headline**: `cartpole_dqn.m` (DQN balances cart-pole).

### Tier 4 — Policy gradient & on-policy actor-critic

Stochastic-policy actors and the REINFORCE / advantage-actor-critic family.

- **Actors**: `rlDiscreteCategoricalActor` (softmax over discrete actions),
  `rlContinuousGaussianActor` (mean+stddev heads).
- **`rlPGAgent`** (REINFORCE, with optional value `baseline`) + options:
  Monte-Carlo return, `−Σ logπ(A|S)·(G − b(S))` loss.
- **`rlACAgent`** (advantage actor-critic / A2C) + options: bootstrapped
  advantage, combined policy + value loss + entropy regularisation.
- **Advantage estimation**: n-step returns and **GAE(λ)**.
- **Headline**: `cartpole_ac.m` (actor-critic balances cart-pole).

### Tier 5 — Continuous control & advanced policy gradient

The flagship continuous-control agents — all specific losses over the shipped
autodiff engine + the T2 noise policies.

- **`rlPPOAgent`** + options: clipped surrogate objective, multiple epochs
  over a collected trajectory batch, GAE, value clipping.
- **`rlTRPOAgent`** + options: natural-gradient / conjugate-gradient step
  with a KL trust region and backtracking line search (reuses Optim CG).
- **`rlDDPGAgent`** + options: `rlContinuousDeterministicActor` +
  `rlQValueFunction` critic, deterministic-policy gradient, OU action noise,
  target actor+critic.
- **`rlTD3Agent`** + options: twin critics + delayed-policy + target-policy
  smoothing.
- **`rlSACAgent`** + options: twin critics, squashed-Gaussian actor,
  automatic temperature (entropy target) — the max-entropy objective.
- **Headline**: `pendulum_ddpg.m` (DDPG pendulum swing-up).

### Tier 6 — Training infrastructure, model-based, custom loops & deployment

The full training surface, model-based RL, custom loops, and the
code-generation deployment path (which leans on the shipped Embedded-Coder /
MATLAB-Coder lane).

- **Full `rlTrainingOptions`**: `ScoreAveragingWindowLength`,
  `SaveAgentCriteria`/`SaveAgentValue`, `Plots`, `StopTrainingValue`,
  evaluation hooks.
- **Training Monitor** logging + Cairo reward/average curves; `rlEvaluator`
  (periodic greedy evaluation); save-candidate-agents.
- **`rlDataLogger`** (log scalars/experiences to disk during training) +
  `rlDataViewer` — the "Log Training Data to Disk" workflow.
- **`rlReplayMemory`** as a first-class object (append/sample/reset) for
  custom loops; HER (`rlHindsightReplayMemory`) optional.
- **Custom training loops**: the documented pattern — manual `getAction` /
  step / `appendExperience` / `gradient`-via-`dlgradient` / `optimize` —
  proven by the "Train Policy Using Custom Training Loop" examples.
- **`rlMBPOAgent`** (model-based policy optimization): a learned environment
  model — `rlNeuralNetworkEnvironment` composed of
  `rlContinuousDeterministicTransitionFunction` +
  `rlContinuousDeterministicRewardFunction` (`dlnetwork`s) — generating
  synthetic rollouts that feed a SAC/DDPG base agent.
- **Deploy**: `generatePolicyFunction(agent)` → a plain MATLAB function the
  existing AOT/LLVM + C/C++ codegen lane already compiles; document the
  `coder`-targeted policy export as the deployment story.
- **Headline**: `custom_train_loop.m` (REINFORCE via a hand-written loop +
  `dlgradient`).

---

## 3. Carve-outs (explicitly out of scope)

- **Reinforcement Learning Designer app** — the whole interactive GUI
  (create/import/edit agents, tune hyperparameters, train from the app). No
  app surface in this project.
- **Simulink / Simscape environments** — `rl*` blocks, the RL Agent block,
  Simulink reset functions, bus-signal observations, Simscape cart-pole.
  (MATLAB `rlFunctionEnv` + predefined MATLAB envs cover the demoable arc.)
- **Parallel & GPU training** — `UseParallel`, multi-process workers,
  multi-GPU. Training runs single-process on the shipped (optionally
  Metal-accelerated) kernel.
- **Multiagent training** — `rlMultiAgentTrainingOptions`,
  `rlMultiAgentFunctionEnv`/`rlTurnBasedFunctionEnv`, the pusher
  environments, decentralised/centralised multi-agent.
- **`rlHybridStochasticActor`** — mixed discrete+continuous action spaces (a
  niche actor; the discrete-categorical and continuous-Gaussian actors in
  T4/T5 cover the mainstream).
- **Offline / behavior-cloning RL** — `rlTrainingFromDataOptions`,
  `rlBehaviorCloningRegularizerOptions`, training an agent from a logged
  dataset rather than live environment interaction.
- **ONNX import of pretrained actor/critic networks** beyond what the DL
  toolbox's `runtime_onnx.cpp` already supports.
- **Hardware deployment targets** — Raspberry Pi, SIL/PIL, the microservice
  Docker image, the policy block for Simulink.
- **Evolutionary-strategy training**, **offline/batch RL from a dataset**,
  **Bayesian-optimization hyperparameter tuning of agents** (the Stats
  `bayesopt` exists but wiring it to a full agent-training objective is
  carved), **curriculum learning**, **transfer-learning** workflows.
- **LSTM/recurrent actors & critics** — sequence agents (`SequenceLength`,
  recurrent DQN/PPO). The BPTT kernel exists in DL, but wiring recurrent
  approximators through the agent update is a documented follow-on.

---

## 4. Effort & sequencing

Because the autodiff keystone is **already shipped**, RL is closer to the
"wire a lot of classdefs + losses over existing kernels" shape of Navigation
than to the "build a new numerical engine" shape of the original Deep
Learning plan. Rough sizing:

- **T1** (specs + grid-world + tabular agents + tabular `train`) — ~1.5 wk.
  Fully self-contained, no autodiff. First demoable slice.
- **T2** (predefined ODE envs + `sim` + policy objects + function envs) —
  ~1.5 wk.
- **T3** (DQN + replay + target nets + LSPI) — ~2 wk. First autodiff-backed
  agent; the architectural risk lives here (proving an end-to-end deep-RL
  update on the shipped tape).
- **T4** (PG + AC + stochastic actors + GAE) — ~1.5 wk.
- **T5** (PPO/TRPO/DDPG/TD3/SAC) — ~3 wk (the catalogue's bulk; each agent is
  a distinct loss + exploration + target-update recipe).
- **T6** (training infra + monitor + custom loops + MBPO + deploy) — ~2 wk.

**~11.5 wk** for all six tiers. First cuts that each close a headline:
**T1 alone** (~1.5 wk, tabular grid-world, zero risk) or **T1–T3** (~5 wk,
through the DQN cart-pole flagship — the slice that proves deep RL works on
this stack). Shipping the toolbox would take the badge **25 → 26**.

The biggest single de-risking relative to every prior plan: **the hard part
(reverse-mode autodiff) is done.** RL is the first major toolbox to *collect
the dividend* of the Deep Learning autodiff investment rather than pay into
it.

---

## 5. Wiring notes (where each piece lands)

Following the established per-toolbox pattern:

- **New runtime dir** `runtime/toolbox/rl/` — `runtime_rl.cpp`
  (`matlab_rl_*` C-ABI: grid-world transition/reward tensors, replay-buffer
  ring, TD/PG/PPO/DDPG/SAC loss assembly, OU/Gaussian noise, return/GAE
  computation) + `rl_classdefs.m` (the spec/env/agent/approximator/options
  classdefs, constructors storing matrices + forwarding method bodies to the
  C-ABI, the proven DSP/Fusion stateful-System-Object posture).
- **Approximators delegate to DL**: `rlQValueFunction`/actor `getValue`/
  `getAction` call straight into the shipped `dlnetwork` forward; the update
  rules call the shipped `dlgradient` + Adam — **no new numerics.**
- **Parser/Resolver**: `rl*` constructors intercepted in the constructor path
  (they are classdefs, like `optimoptions`/`trackingKF`); raw `matlab_rl_*`
  symbols must be registered in `LowerTensorOps` `pde_table` or they fault
  with "unsupported call shape" (the Navigation trap).
- **`train`/`sim`** are multi-arity drivers returning structs — use the
  multi-return splitter + `Rtys` pattern (Stats `anova1` precedent) and read
  options structs with `lowerExpr`, not `loadObj` (the GADS/Stats trap).
- **`rl*Options`** objects parallel `rlTrainingOptions`/`optimoptions` —
  Name=Value carrier classdefs intercepted in the constructor path with a
  separate `_opts` entry to avoid null/obj ambiguity (the GADS T6 trap).
- **Predefined-env ODE dynamics** reuse the shipped `ode45` / fixed-step
  integrator; **training-monitor curves** route through Cairo (`plotting.md`).

Companion docs:
[`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md)
(`dlnetwork` + autodiff tape + Adam/SGDM reused wholesale — the foundation),
[`optim_toolbox_roadmap.md`](optim_toolbox_roadmap.md) (CG/`lsqlin` for
TRPO/LSPI), [`embedded_coder_roadmap.md`](embedded_coder_roadmap.md)
(`generatePolicyFunction` deployment lane), [`plotting.md`](plotting.md)
(training monitor), [`feature_status.md`](feature_status.md).
