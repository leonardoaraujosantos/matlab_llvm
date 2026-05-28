# Deep Learning Toolbox (+ Deep Learning HDL) — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot + EmitSV/cocotb) needs to ship in order to faithfully **compile and
execute**, **debug/REPL**, **demo**, and (for the HDL track) **emit a
fixed-point inference datapath to SystemVerilog** for Deep-Learning-Toolbox
programs.

Sources: *Deep Learning Toolbox™ User's Guide* (R2026a — 7 chapters: Deep
Networks · Deep Network Designer · Deep Learning with Images · Deep Learning
with Time Series, Sequences, and Text · Tuning and Visualization · Manage
Experiments · Parallel and the Cloud) and *Deep Learning HDL Toolbox™ User's
Guide* (R2026a — 13 chapters: Deep Learning Processor IP core · FPGA workflow
+ APIs · LIBIIO/Ethernet · supported networks/layers · custom processor
config + codegen · quantization · IP-core user guide · LSTM support).

This is the **single largest toolbox in the catalogue** and the one with the
sharpest feasibility split, so the roadmap leads with an honest architectural
assessment rather than a flat tier list.

---

## 1. The one architectural fact that shapes everything: autodiff

Every other shipped toolbox computes a *fixed* function of its inputs. Deep
Learning is the first toolbox whose headline workflow — **training** — is an
*optimisation over a function's own gradient*. The forward pass of a network
is just matrix multiplies, convolutions, and elementwise nonlinearities — all
of which the runtime **already has** (`mtimes`, the Image/DSP `conv2`/`fft2`,
the elementwise kernel). The backward pass needs **reverse-mode automatic
differentiation** over that op graph — which the project **does not have**.

That single gap cleaves the toolbox cleanly in two:

- **Inference (forward-only)** — load a network with known weights and run
  `predict`/`classify`. Needs *no* autodiff: it is a composition of shipped
  matrix/conv/activation ops. **Feasible today on the existing kernel.**
- **Training (forward + backward)** — `trainnet` / custom loops with
  `dlgradient`. Needs the **`dlarray` traced-autodiff engine** — the keystone
  new infrastructure, on which Tiers 3–6 all rest.

So the roadmap is deliberately ordered **inference first (T1), the autodiff
engine second (T2), training third (T3)** — each independently shippable, and
each closing a self-contained, demoable slice. A reader who only wants "run a
trained net" stops after T1; the autodiff engine (T2) is the gate to
everything that *learns*.

**What the project already ships that Deep Learning composes on** (an
unusually deep base for a brand-new toolbox):

- **Dense linear algebra + the matrix kernel** (`mtimes`/`mldivide`/`svd`/…)
  — the forward pass of every fully-connected / attention / normalization
  layer is matrix arithmetic.
- **`conv2` / `conv` / `fft2`** (Image + DSP/Signal toolboxes) — the
  convolution-layer forward pass.
- **The Statistics & ML toolbox** ([`global_optim_and_stats_ml_plans.md`](global_optim_and_stats_ml_plans.md))
  — `pca`/`kmeans`/`fitcecoc`/`confusionmat`/`bayesopt`; the UG's "Choose an
  AI Model" chapter explicitly contrasts Stats-ML vs Deep-Learning training,
  and `rocmetrics`/`confusionchart` reuse the Stats classification-metrics
  surface.
- **`bayesopt`** (Stats T6) — Experiment Manager's Bayesian hyperparameter
  strategy is literally this solver.
- **`ode45`/`ode23s`** ([`ode.md`](ode.md)) — neural ODE / latent ODE forward
  integration.
- **The fixed-point `fi` type + `EmitSV` + cocotb SIL** ([`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md)
  DSP-HDL T7–T8 precedent) — the *entire* Deep Learning HDL track is "compile
  a quantized network to a fixed-point SystemVerilog datapath + verify it
  bit-accurately against the double inference in cocotb". This is the
  project's strongest fit in the whole toolbox.
- **The GPU dispatcher + `parfor` outliner** ([`gpu_coder_roadmap.md`](gpu_coder_roadmap.md))
  — single-device training acceleration (multi-GPU/cloud is carved).
- **The `classdef` carrier + function-handle ABI + `bayesopt` objective ABI**
  — `dlnetwork`/layer objects are classdef carriers; custom loops + custom
  layers ride the handle ABI.
- **`mflowLink`** ([`embedded_coder_roadmap.md`](embedded_coder_roadmap.md))
  — the block-diagram answer for the UG's Simulink Deep-Learning blocks.

**What is genuinely new** (and roughly in dependency order): the **`dlarray`
autodiff engine**, the **layer forward/backward library**, the **stochastic
solvers** (SGDM/Adam/RMSProp), the **recurrent kernels** (LSTM/GRU forward +
BPTT), and the **`dlhdl` fixed-point compiler → SV**.

**No external dependency** — no PyTorch, no TensorFlow, no cuDNN, no ONNX
runtime. Every layer, solver, and autodiff rule is hand-coded over the shipped
kernel. (ONNX/PyTorch/TF *import* is carved — see §10.)

---

## 2. Reading guide

- **Tier** = priority + dependency band, not strict order. The Deep Learning
  Toolbox proper is **Tiers 1–6**; the Deep Learning HDL Toolbox is a parallel
  **HDL track (H1–H3)** that depends only on T1 (inference) + the shipped
  `fi`/EmitSV lane, *not* on the training tiers.
- **Effort** in the existing Phase-5.6.x cadence (one focused session ≈ a
  half-day; a "week" ≈ 5 sessions). This is the **largest single estimate in
  the catalogue** — see §9. **T1 + T2 (~7 wk) is the highest-value cut**: it
  delivers inference *and* the autodiff engine, after which training tiers are
  incremental.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started. **T1 + T2 (the
  dense surface) shipped 2026-05-27** (badge 24→25): the `dlarray` value type
  over a **reverse-mode autodiff tape**, forward inference via operator
  overloading (`W*X + b`, `relu`/`sigmoid`/`tanh`/`softmax`) and reductions/
  losses (`sum`/`mean`/`crossentropy`/`mse`), plus `dlgradient` (verified
  against finite differences to **1.24e-10**) and `extractdata`.  **2-D dense
  only** — convolution + the 4-D `SSCB` tensor are carved (the runtime has no
  rank-N type; see [`any_shape_roadmap.md`](any_shape_roadmap.md) Tier C), and
  the object-array `dlnetwork`/layer-object container is carved (no classdef
  array literals) — the *functional* "custom training loop" form is the shipped
  surface.  **T3 (training) partial 🟡** — the **custom training loop** is
  proven end-to-end (`dl_mlp_train.m`: an MLP trained from scratch by SGD over
  the autodiff — forward via dlarray operators → `dlgradient` per parameter →
  manual update via `extractdata`/re-wrap — reaches 100% train accuracy, loss
  1.80→0.01); the built-in `trainnet`/`trainingOptions` driver + the functional
  solvers `adamupdate`/`sgdmupdate`/`rmspropupdate` are carved (they want the
  object-array `dlnetwork` / multi-return state, both deferred).  **T4 ✅
  complete** on the functional surface: **`lstm`** (one OP_LSTM tape node,
  BPTT through every per-timestep gate, `dl_lstm_sequence.m` → 100%, loss
  6→0); **`gru`** (same task → 100%, loss 6→0); **`bilstm`** (packed
  forward/backward weights, both directions BPTT'd in one node);
  **`lstmp`** (projected hidden, `dP` accumulated alongside `dW/dR/db`);
  **functional scaled-dot-product attention** composes from existing matmul
  + softmax + the added `transpose` — no dedicated opcode (`dl_attention.m`:
  associative-recall, softmax peaks on the matching key, loss 1→0);
  **`embed`** (gather-forward + scatter-add-backward, repeated indices
  correctly accumulate — `dl_embed_train.m` → per-element error < 0.01).
  The layer-object forms (`lstmLayer`/`gruLayer`/`bilstmLayer`/
  `lstmProjectedLayer`/`selfAttentionLayer`/`wordEmbeddingLayer`) are carved
  with the rest of `dlnetwork`.  **T5 partial 🟡** — the functional
  architectures + transfer-learning patterns ship: residual blocks compose
  from the existing overloaded `plus` (`dl_residual_train.m` trains a
  4-layer skip-connection MLP); transfer learning keeps the pretrained
  encoder as plain numeric matrices outside the autodiff and trains only
  the head as a dlarray (`dl_transfer_learn.m`: frozen 4→6 encoder + 3-class
  head → 96% accuracy).  `replaceLayer`/`freezeLayers` object-array APIs
  carved with `dlnetwork`.  **HDL Tier-H1 + H2 ✅, H3 partial 🟡**: H1 ships `dlquantize`/`dlqscale`
  (symmetric per-tensor INT8) — `dl_quantize_check.m` proves quantization
  preserves the T3 MLP's accuracy (100% in both double and INT8, max logit
  drift ≈ 0.1).  H2 ships **fi-typed SystemVerilog emission of a quantized
  MLP forward**: `examples/hdl/dlhdl_quant_mlp.m` (a Q16.8 2-2-1 net,
  hand-unrolled) lowers through the existing `EmitSV` lane to ~15 lines of
  synthesizable SV (Verilator + Yosys clean), joining the EmitSV regression
  sweep.  H3 generates the cocotb harness but currently fails its 100-vector
  compare because of the documented **SV-vs-Python fi saturation divergence**
  (the SV truncates each 16-bit op while the Python ref saturates at the
  natural-growth width) — same class of gap that previously blocked Tier-3
  cocotb cases before the per-op-wrap pass fixed them.  **T6 + HDL H3
  bit-accuracy + H4 LSTM-on-FPGA are 🔵 not started.**  The forward-pass substrate (matrix kernel),
  the fixed-point/SV/cocotb lane, `bayesopt`, `ode45`, and the classdef +
  handle ABI are all already in the runtime.
- **No external dependencies** — matching project precedent.
- **Discovered en route + since fixed**: a pre-existing plain-matrix
  copy-on-write bug (`B = A; B(i) = v` mutated `A`) surfaced while building the
  gradient-check example; fixed separately (matrix clone-on-assign) and merged
  to main.

---

## 3. Tier-1 — Inference: `dlnetwork` forward pass 🟡 (FOUNDATION — dense shipped)

*Load a network with known weights and run it. No autodiff. Pure composition
of shipped matrix/conv/activation ops.* This is the "import a trained model
and predict" lane and the foundation the HDL track also builds on.

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| 1.1 | `dlarray` (forward-only) | labelled N-D array (`'SSCB'`/`'CB'`/`'CBT'` data formats) wrapping a runtime tensor; `dims`/`finddim`/`stripdims`/`extractdata` | matlab_matN |
| 1.2 | `dlnetwork` carrier | layer array + connections + `Learnables`/`State` tables; `initialize` | classdef carrier |
| 1.3 | Core layers (forward) | `featureInputLayer`/`imageInputLayer`/`sequenceInputLayer`, `fullyConnectedLayer`, `convolution2dLayer`/`convolution1dLayer`, `batchNormalizationLayer`/`layerNormalizationLayer`, `maxPooling2dLayer`/`averagePooling2dLayer`/`globalAveragePooling2dLayer`, `dropoutLayer` (identity at inference), `softmaxLayer`, `flattenLayer`, `additionLayer`/`concatenationLayer` | `mtimes` + `conv2` |
| 1.4 | Activations | `reluLayer`/`leakyReluLayer`/`clippedReluLayer`/`tanhLayer`/`sigmoidLayer`/`geluLayer`/`eluLayer`/`swishLayer` + functional `relu`/`sigmoid`/`softmax`/`gelu` | elementwise kernel |
| 1.5 | `predict` / `classify` / `minibatchpredict` | forward pass over the layer DAG (topological order); `classify` = argmax + class labels; multi-input/output | DAG eval |
| 1.6 | Weight load | populate `Learnables` from a user matrix/`.mat` (hand-built architecture); `setLearnableValue`/`getLearnableValue` | matlab_mat |
| 1.7 | `analyzeNetwork` (headless) | layer table + activation sizes + learnable counts (the Deep Network Designer "Check Network" minus the GUI) | introspection |

**Headline-within-tier**: `dl_lenet_infer.m` — build a small LeNet-style CNN
(`imageInputLayer`→`convolution2dLayer`→`reluLayer`→`maxPooling2dLayer`→…→
`fullyConnectedLayer`→`softmaxLayer`), load known weights, `classify` a
digit image, and report the predicted class + score. Closes the
**inference lane** end-to-end on the shipped kernel.

---

## 4. Tier-2 — The `dlarray` autodiff engine 🟡 (KEYSTONE — dense shipped)

*Reverse-mode automatic differentiation over the supported op set — the single
biggest new infrastructure piece, and the gate to all training.*

| # | Surface | Notes |
|---|---------|-------|
| 2.1 | Traced `dlarray` | every supported op records onto a tape (Wengert list) when an input is a traced `dlarray`; nodes hold the op + parents + a local pullback |
| 2.2 | `dlgradient(loss, vars...)` | reverse sweep of the tape — seed the loss adjoint, accumulate `∂loss/∂var` per traced variable; multi-output + higher-order (`EnableHigherDerivatives`) stretch |
| 2.3 | `dlfeval(@fn, ...)` | evaluate a function under the AD context (open/close the tape around the call) |
| 2.4 | Differentiable op rules | `+`/`-`/`.*`/`*` (matmul)/`./`/`sum`/`mean`/`reshape`/`permute`/`exp`/`log`/`sqrt`/`tanh`/`max`/`relu`/`conv`/`batchnorm`/`crossentropy`/`softmax` — each with its analytic pullback |
| 2.5 | Loss functions | `crossentropy`/`mse`/`l1loss`/`huber`/`l2loss` as differentiable ops |
| 2.6 | Gradient check | finite-difference validation harness (`dlgradient` vs central-difference) — the gating-test backbone |

**Headline-within-tier**: `dl_autodiff_check.m` — define `y = f(x)` over a
chain of the supported ops, compute `dlgradient` and confirm it matches a
central-difference gradient to ~1e-6. Proves the engine before any training
rides on it.

---

## 5. Tier-3 — Training 🟡 (custom loop shipped; trainnet carved)

*Stochastic optimisation of a `dlnetwork` over the autodiff engine — closes
the headline "train a classifier from scratch" workflow.*

| # | Surface | Notes |
|---|---------|-------|
| 3.1 | `trainingOptions(solver, …)` | `'sgdm'`/`'adam'`/`'rmsprop'` + LearnRate/schedule/L2Regularization/MaxEpochs/MiniBatchSize/Shuffle/ValidationData/GradientThreshold |
| 3.2 | Solvers | SGD-with-momentum, Adam, RMSProp parameter-update rules over the `Learnables` table |
| 3.3 | `trainnet(data, net, lossFcn, opts)` | mini-batch loop: forward → `dlgradient` → solver step → epoch/validation logging; returns the trained `dlnetwork` |
| 3.4 | `minibatchqueue` + `arrayDatastore` | batching/shuffling over in-memory arrays (datastores for big data carved) |
| 3.5 | Custom training loop | `dlfeval`+`dlgradient`+`adamupdate`/`sgdmupdate`/`rmspropupdate` functional solvers — the full custom-loop API |
| 3.6 | `trainNetwork` (legacy) | thin shim over `trainnet` for the layer-array + `trainingOptions` classic call |
| 3.7 | Headless training monitor | per-epoch loss/accuracy table to stdout (the `trainingProgress` plot is a Cairo stretch) |

**Headline-within-tier (the roadmap headline)**: `dl_digits_train.m` — train
the LeNet CNN from §3 **from scratch** on a built-in digit dataset and report
test accuracy > 95%. This exercises T1 (forward) → T2 (autodiff) → T3
(solver) end-to-end and is the proof that *training works* in the compiler.

---

## 6. Tier-4 — Sequence / recurrent / attention ✅ (functional surface complete; layer-object forms carved with `dlnetwork`)

| # | Surface | Notes |
|---|---------|-------|
| 4.1 | **`lstm(X, H0, C0, W, R, b)` ✅** | functional form — one custom `OP_LSTM` tape node, per-timestep gate/state buffer, BPTT in the existing `dlgradient`.  `lstmLayer` object form carved with the rest of `dlnetwork`. |
| 4.2a | **`gru(X, H0, W, R, b)` ✅** | reset/update/candidate gates; BPTT handles the `r.*h_prev` path in the candidate's recurrent contribution. 3H-stacked `[r; z; h]` weights. |
| 4.2b | **`bilstm(X, H0f, C0f, H0b, C0b, W, R, b)` ✅** | bidirectional LSTM packed as `[forward; backward]` weights (8H × D / 8H × H / 8H × 1); output is `[Yf; Yb_aligned]` re-aligned to original time order; both directions BPTT'd in one tape node. |
| 4.2c | **`lstmp(X, H0, C0, W, R, P, b)` ✅** | LSTM with a projection `P` (`Hp × H`) applied to the hidden state — the recurrence + output operate at `Hp`, but the cell stays at `H`; `dlgradient` accumulates `dP` alongside `dW/dR/db`. |
| 4.3 | Sequence I/O | `sequenceInputLayer`, sequence padding/truncation, `sequenceFoldingLayer`/`sequenceUnfoldingLayer` — wait on `dlnetwork`. |
| 4.4 | **`embed(E, idx)` ✅** | wordEmbeddingLayer's functional core.  `OP_EMBED` gather-forward + scatter-add-backward (repeated indices correctly accumulate gradient).  Indices are plain integers (not dlarrays). |
| 4.5 | **functional attention ✅** | scaled-dot-product attention as `V * softmax(K' * Q)` — composes from existing matmul + softmax + the newly-added `transpose` (`OP_TRANSPOSE`), no dedicated attention opcode.  `selfAttentionLayer` object form carved with the rest of `dlnetwork`. |
| 4.6 | 1-D conv sequence path | `convolution1dLayer` — needs rank-3 tensors (`any_shape_roadmap` Tier C). |

**Headlines (shipped)**:
- `examples/dlnet/dl_lstm_sequence.m` — 4-unit LSTM trained on a first-bit-memory task → 100%, loss 6→0.
- `examples/dlnet/dl_gru_sequence.m` — same task with a GRU cell → 100%, loss 6→0 (matches the LSTM headline with fewer parameters).
- `examples/dlnet/dl_attention.m` — associative-recall task: learn `Wq`/`Wk`/`Wv` so attention focuses on the matching key (loss 1→0; softmax peaks on the correct key).
- `examples/dlnet/dl_embed_train.m` — learn a 3×5 embedding table from a token sequence with repeats (per-element error < 0.01).

---

## 7. Tier-5 — Architectures + transfer learning 🟡 (functional patterns shipped; object-array surface carved with `dlnetwork`)

| # | Surface | Notes |
|---|---------|-------|
| 5.1 | **Transfer learning ✅ (functional)** | Keep the pretrained feature extractor as plain numeric matrices (no dlarray → no gradient flow); train only the head as a dlarray with a custom training loop.  `replaceLayer`/`addLayers`/`connectLayers`/`freezeLayers` object-array APIs are carved with `dlnetwork`, but the *patterns* they encapsulate (frozen encoder + new head) work today. |
| 5.2 | **Residual / skip-connection nets ✅ (functional)** | A residual block is just `relu(W2 * relu(W1*x + b1) + b2) + x` — `plus` is already overloaded on dlarrays so the skip-add records on the tape and `dlgradient` flows gradient through both branches.  ResNet-style architectures of arbitrary depth compose from this primitive. |
| 5.3 | `imagePretrainedNetwork` | a compact baked-weight model — carved alongside real AlexNet/ResNet/BERT (the wrapper API needs `dlnetwork`; the *pattern* of "load + run pretrained weights" ships today as `dl_transfer_learn.m`'s encoder block). |
| 5.4 | GAN / WGAN-GP | generator+discriminator alternating updates — composes from the shipped autodiff + custom loop; carved as a follow-on example. |
| 5.5 | VAE / autoencoder | encoder/decoder + reparameterisation + KL — composes from existing ops + `randn` outside the autodiff; carved as a follow-on example. |
| 5.6 | Twin/Siamese network | shared-weight comparison + contrastive loss — composes from existing ops; the carved piece is two-output dlnetwork (functional form is shippable). |
| 5.7 | Neural ODE layer | `dlode45` — forward integration via the shipped `ode45` + adjoint backward (needs the adjoint-ODE solver). |

**Headlines (shipped)**:
- `examples/dlnet/dl_residual_train.m` — train a 4-layer residual MLP; the skip connections are *just* `y + x` and gradient flows through both paths automatically.
- `examples/dlnet/dl_transfer_learn.m` — adapt a frozen 4→6 pretrained encoder to a new 3-class downstream task with only a fresh classifier head trained over the autodiff — **96% accuracy** with the encoder mathematically guaranteed unchanged.

---

## 8. Tier-6 — Tuning, visualization, metrics, quantization 🔵

| # | Surface | Notes |
|---|---------|-------|
| 6.1 | Interpretability | `gradCAM`/`occlusionSensitivity`/`imageLIME` (ride the T2 gradient engine + occlusion forward sweeps) |
| 6.2 | `tsne` | t-SNE embedding for activation visualisation (reuses the Stats distance kernel) |
| 6.3 | Metrics | `rocmetrics`/`confusionchart`/`accuracy`/`precision`/`recall`/`fScore` (reuse the Stats classification-metrics surface) |
| 6.4 | Bayesian hyperparameter search | `bayesopt`-driven sweep over `trainingOptions` (reuse the shipped `bayesopt`) — the Experiment-Manager *engine* minus the app |
| 6.5 | `dlquantizer` | INT8 calibration (`calibrate`) + `validate` accuracy drop + `quantizationDetails` — the on-ramp to the HDL track |
| 6.6 | Verification (subset) | `dlnetwork` robustness/`l-inf` bound check + out-of-distribution score (a small slice of Chapter 5's verification surface) |

**Headline-within-tier**: `dl_gradcam_explain.m` — Grad-CAM heatmap over the
§1 CNN's prediction, written as a PNG via Cairo.

---

## H. Deep Learning HDL track 🟡 — H1 INT8 quantization SHIPPED; H2 (fi-SV) + H3 (cocotb) next

The DL HDL UG is, for this project, **"compile a quantized inference network
to a fixed-point SystemVerilog datapath and verify it bit-accurately in
cocotb"** — exactly the lane the project already runs for DSP HDL (T7–T8) and
Embedded Coder cocotb SIL. It depends on **T1 (inference) + T6.5
(`dlquantizer`) + the shipped `fi`/EmitSV/cocotb lane**, *not* on training.

| # | Surface | Notes | Rides |
|---|---------|-------|-------|
| H1 | **`dlquantize(W)` + `dlqscale(W)` ✅** | symmetric per-tensor INT8 quantization — `scale = max(abs(W))/127`, `Q = round(W/scale)` clipped to `[-127, 127]`, output is `Q*scale` rounded onto the int8 lattice.  Plain matrix in/out, no autodiff (post-training step).  `examples/dlnet/dl_quantize_check.m`: trains the T3 MLP, INT8-quantizes every weight, re-runs inference — both double and INT8 hit 100% accuracy, max logit drift ≈ 0.1.  The `dlhdl.ProcessorConfig`/`dlhdl.Workflow`/`estimatePerformance` object-array APIs are carved with `dlnetwork`. | T3 |
| H2 | **fi-typed SV emission ✅** | a hand-unrolled quantized MLP forward (Q16.8 weights baked as `fi` constants, `relu` as `z<0?0:z`, multi-layer linear) lowers cleanly through the existing `EmitSV` lane.  `examples/hdl/dlhdl_quant_mlp.m` + `test/EmitSV/dlhdl_quant_mlp.sv.expected`: a 2-2-1 MLP emits ~15 lines of synthesizable SV (powers-of-2 weights fold to bit-shifts; non-trivial weights to `*` with sign extension), passes Verilator lint + Yosys synth, joins the EmitSV regression sweep (80/80 green). | `fi` + EmitSV |
| H3 | **cocotb bit-accuracy** 🟡 (harness generates; bit-accuracy blocked) | `-emit-cocotb` generates the full harness (`test_<n>.py`, `<n>_ref.py`, `cocotb_fi.py`, `Makefile`).  Currently fails the 100-vector compare because of the documented **SV-vs-Python fi saturation divergence** — the SV truncates each 16-bit op while the Python reference saturates at the natural growth width (33/34/64 bits between ops); the harness drives full-range int16 stimulus so the divergence is visible immediately.  Same class of gap that previously blocked Tier-3 cocotb cases (`aes_round`, `barrel_shifter`, `crc32`, …) before the per-op-wrap pass fixed them — would need an equivalent pass for the dlhdl path. | H2 + per-op wrap |
| H4 | LSTM-on-FPGA compile 🔵 | the Chapter-13 LSTM/GRU layer compilation to the fixed-point recurrent datapath | H2 + T4 |

**Headline-within-tier (HDL tracer)**: `dlhdl_cnn_sil.mflow` /
`dlhdl_cnn_sil.m` — quantize the §1 LeNet with `dlquantizer`, `compile` it,
emit the conv/FC/relu/pool datapath to SystemVerilog, and cocotb-verify the
FPGA-bound prediction matches the double inference within the fixed-point
tolerance. This is the project's **headline DL-on-hardware demo** and reuses
the shipped HDL infrastructure wholesale.

**HDL carve-outs**: real FPGA bitstream generation, board deployment
(Arria 10 / ZCU102 / ZC706), the LIBIIO/Ethernet live connection, on-board
profiling, and `dlhdl.Target('Vivado')` synthesis — all need physical
hardware + vendor toolchains and are out of a CI/simulation scope (matching
the project's "SV + cocotb simulation surface, not silicon" precedent).

---

## 9. Status / wiring / examples / tests

### 9.1 Compile / Execute

- **Runtime**: `runtime/toolbox/dlnet/runtime_dlnet.cpp` (dlarray + autodiff
  tape, layer forward/backward, solvers, recurrent kernels, quantizer) +
  `runtime/toolbox/dlnet/dlnet_classdefs.m` (the `dlnetwork`/layer/
  `trainingOptions`/`dlhdl.*` classdefs). The autodiff tape is thread-local
  runtime state (the `dlfeval` scope), mirroring the `lsqnonlin` residual-ctx
  pattern. Add to the strict no-C-cast list.
- **Wiring**: the six-place pattern (the
  [`navigation_toolbox_roadmap.md`](navigation_toolbox_roadmap.md) §8.1 /
  Robotics §8.1 map applies verbatim — `kToolboxDirs` ×2, prelude `Cls[]` +
  AOT `Names[]` + `findToolboxClassdef`, `Resolver.cpp`, `Lowering.cpp` ctor
  intercepts + arg-0-class method dispatch, `LowerTensorOps.cpp` pde_table,
  `run_tests.sh` + `run_sweep.sh`). **Critical reuse-trap** (from Navigation):
  every raw `matlab_dlnet_*` runtime symbol emitted as a `call_builtin` callee
  MUST get a `pde_table` signature row or it fails "unsupported call shape".
  Layer constructors are classdef-ctor intercepts; `predict`/`forward`/
  `trainnet`/`dlgradient` are method/free-fn dispatch; `dlfeval` wraps a
  function-handle (LowerAnonCalls retype).
- **Backends**: LLVM JIT + native are primary. `-emit-c`/`-emit-cpp` parity is
  a per-tier stretch (inference ports cleanly; the autodiff tape is rougher).
  The **HDL track targets `-emit-systemverilog` + cocotb directly** — that is
  the whole point of H2.

### 9.2 Debug / REPL

A `dlnetwork` persists across REPL inputs and renders its layer/learnable
summary in the DAP inspector; a paused custom training loop shows the running
loss; `dlarray` renders with its data + dimension labels.

### 9.3 Examples (`examples/dlnet/`)

| Example | Closes |
|---|---|
| `dl_lenet_infer.m` | T1 — build + weight-load + `classify` |
| `dl_autodiff_check.m` | T2 — `dlgradient` vs finite difference |
| `dl_digits_train.m` | **T3 headline** — train LeNet from scratch > 95% |
| `dl_seq_classify.m` | T4 — LSTM sequence classifier |
| `dl_transfer_learn.m` | T5 — fine-tune the built-in pretrained CNN |
| `dl_gradcam_explain.m` | T6 — Grad-CAM PNG |
| `dlhdl_cnn_sil.m` / `.mflow` | **HDL headline** — quantize → compile → SV → cocotb bit-accuracy |

### 9.4 Tests (`test/Run/` + `test/EmitSV/`)

`dlnet_{dlarray,autodiff,layers,train,lstm,quantize}.m` gating tests (the
autodiff finite-difference check + a 2-layer training-convergence check are
the backbone; randomised inits seed `rng` and assert rounded/tolerance
quantities, per the Navigation precedent). The HDL track adds an `EmitSV`
golden + a cocotb bit-accuracy lane (the DSP-HDL precedent). Full regression
stays green; badge bumps to **25 toolboxes**.

### 9.5 Effort summary

| Tier | Scope | Est. | New infra |
|---|---|---|---|
| T1 | inference forward pass | ~3 wk | layer forward library + DAG eval |
| T2 | `dlarray` autodiff engine | ~4 wk | **reverse-mode AD tape (keystone)** |
| T3 | `trainnet` + solvers | ~3 wk | SGDM/Adam/RMSProp + mini-batch loop |
| T4 | sequence / recurrent / attention | ~3 wk | LSTM/GRU forward + BPTT, attention |
| T5 | architectures + transfer learning | ~3 wk | GAN/VAE/Siamese loops, `dlode45` |
| T6 | tuning / viz / metrics / quantize | ~3 wk | Grad-CAM/LIME/tsne, `dlquantizer` |
| H1–H3 | DL HDL → SV + cocotb | ~5 wk | fixed-point DL compiler → EmitSV |

**Total ~24 wk (DL) + ~5 wk (HDL)** — the catalogue's largest. **T1 + T2
(~7 wk) is the recommended first cut** (inference + the autodiff keystone);
**T1 + H1–H3 (~8 wk) is the alternative HDL-first cut** that delivers
"deep-learning-on-FPGA simulation" without ever needing the training tiers —
and plays directly to the project's SystemVerilog/cocotb/fixed-point strengths.

---

## 10. Carve-outs (explicitly out of scope)

The Deep Learning UG's ~2,000 pages lean heavily on companion products, GUIs,
external frameworks, and physical hardware. Carved:

- **Apps** — Deep Network Designer (Ch. 2) and Experiment Manager (Ch. 6) are
  GUIs; the *programmatic* network-building and the `bayesopt`-driven sweep
  *engine* are in scope (T6.4), the visual apps are not.
- **All Simulink Deep-Learning blocks** (Predict / Stateful Classify / the
  Simulink GAN/lane-detection/ECG models) — the `mflowLink` lane is the
  project's block-diagram answer.
- **External-framework import/export** — `importNetworkFromONNX` /
  `importNetworkFromPyTorch` / `importNetworkFromTensorFlow` /
  `exportNetworkToONNX` / `exportNetworkToTensorFlow`. These are large
  format-parser efforts with no numeric value to the kernel; a *minimal* ONNX
  inference-graph importer is a documented stretch, not a tier.
- **Real pretrained network weights** — AlexNet / ResNet / GoogLeNet /
  SqueezeNet / YOLO / BERT / YAMNet weight blobs (100s of MB). One *small*
  built-in pretrained CNN ships (T5.2); the named large nets are carved.
- **Multi-GPU / cluster / cloud training** (Ch. 7) and **batch-job offload** —
  single-device acceleration rides the GPU Coder dispatcher; scale-out is
  carved.
- **Big-data datastores** (`imageDatastore`/`augmentedImageDatastore` over
  disk, out-of-memory) — in-memory `arrayDatastore` + `minibatchqueue` ship;
  disk-backed pipelines are carved.
- **Reinforcement Learning / Computer Vision / Audio / Text Analytics
  toolbox** dependencies (object detectors, `bert`, `wav2vec`, the speech /
  YAMNet examples) — these belong to companion toolboxes.
- **DL HDL silicon** — real bitstream/board deployment, LIBIIO/Ethernet,
  on-board profiling, vendor synthesis (§H carve-out): simulation surface
  only.
- **`dlquantizer` GPU/CPU `'lib'` targets** (TensorRT/MKL-DNN codegen) — the
  fixed-point/FPGA calibration path ships (T6.5 → H), the vendor-library
  targets do not.

Companion docs:
[`global_optim_and_stats_ml_plans.md`](global_optim_and_stats_ml_plans.md)
(Stats-ML AI-model contrast + `bayesopt` + classification metrics),
[`gpu_coder_roadmap.md`](gpu_coder_roadmap.md) (single-device training accel),
[`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md) (the DSP-HDL T7–T8 SV +
cocotb precedent the DL HDL track follows),
[`image_toolbox_roadmap.md`](image_toolbox_roadmap.md) (image preprocessing +
`conv2`), [`ode.md`](ode.md) (neural-ODE integration),
[`embedded_coder_roadmap.md`](embedded_coder_roadmap.md) (the `mflowLink`
answer for Simulink DL blocks), [`feature_status.md`](feature_status.md).
