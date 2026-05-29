# Deep Learning Toolbox — Tutorial

The Deep Learning lane in `matlab_llvm` is built on two pillars: **forward-pass
inference** over the `dlarray` type, and **reverse-mode automatic
differentiation** (the gradient tape behind `dlgradient`). The autodiff tape is
the keystone — once a forward pass is expressed with natural operators on
`dlarray` values, `dlgradient` walks the tape backward to produce exact analytic
gradients, which is what makes training (MLP, CNN, LSTM/GRU, attention, GAN, VAE)
work end-to-end. A separate quantize + ONNX surface covers the HDL/interop
on-ramp.

## Supported features

- **Core type & tape**: `dlarray`, `extractdata`, `dlgradient` (reverse-mode
  autodiff over the recorded op tape).
- **Dense / activation ops**: matrix `*`, `+`, `.*`, `./`, `-`, broadcasting on
  the elementwise ops, `relu`, `leakyrelu`, `gelu`, `swish`, `softplus`, `elu`,
  `sigmoid`, `tanh`, `softmax`, `log`, `sqrt`, `mean(x)` / `mean(x, dim)`,
  `sum`, `transpose`.
- **Losses**: `crossentropy`, `mse`.
- **Convolution stack**: `conv2d_batch`, `maxpool2d`, `reshape` (flatten),
  on the rank-N tape (HxWxCxN convention).
- **Recurrent**: `lstm`, `gru` (BPTT through a single fused tape node).
- **Normalisation / regularisation**: LayerNorm (built from `mean`/`sqrt`/`./`),
  Dropout, batch-norm style flows.
- **Attention / Transformer**: single-head scaled dot-product self-attention,
  `embed`, full encoder block (attention → residual+LN → FFN → residual+LN).
- **dlnetwork carrier**: `dlnetwork()`, `addFC`, `addRelu`, `addSigmoid`,
  `addTanh`, `addSoftmax`, `netPredict`, `netNumLayers`, `trainnet` (Adam
  driver).
- **Quantization (HDL on-ramp)**: `dlquantize` (symmetric INT8), `dlqscale`.
- **ONNX interop**: programmatic graph build (`onnxNewModel`, `onnxAddInit`,
  `onnxBeginNode`/`onnxEndNode`, `onnxSave`, `onnxRead`, `onnxRun`) over Gemm /
  Relu / Sigmoid / Add / Mul / Softmax.

## Build & run

```bash
build/matlabc -emit-llvm examples/dlnet/dl_mlp_train.m > /tmp/dl_mlp_train.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/dl_mlp_train.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/dl_mlp_train
/tmp/dl_mlp_train
```

## Worked examples

### MLP inference — forward pass only  (`examples/dlnet/dl_mlp_infer.m`)

A two-layer classifier forward pass written with natural operators on `dlarray`.
No tape needed for inference.

```matlab
W1 = dlarray([0.10 0.20 -0.15 0.05; -0.30 0.40 0.25 0.10; 0.05 -0.20 0.30 0.45]);
b1 = dlarray([0.1; -0.1; 0.05]);
W2 = dlarray([0.3 -0.2 0.5; -0.4 0.6 0.1; 0.2 0.1 -0.3]);
b2 = dlarray([0.0; 0.0; 0.0]);

features = dlarray([5.1; 3.5; 1.4; 0.2]);   % one iris-like sample
scores   = softmax(W2*relu(W1*features + b1) + b2);
p        = extractdata(scores);
fprintf('class scores = %.4f %.4f %.4f\n', p(1), p(2), p(3));
[mx, idx] = max([p(1) p(2) p(3)]);
fprintf('predicted class = %.0f (score %.4f)\n', idx, mx);
```

`softmax`/`relu`/matrix-`*` compose directly on `dlarray`; `extractdata` returns
the underlying numeric matrix so the scores can be inspected and `max`-reduced.

### Autodiff vs finite difference  (`examples/dlnet/dl_autodiff_check.m`)

The keystone check: `dlgradient` sweeps the tape for `dL/dW` and the result is
validated against a central finite-difference estimate.

```matlab
Wd = [0.5 -0.3; 0.2 0.8];
xd = [1.0; 2.0];
W  = dlarray(Wd);
x  = dlarray(xd);
loss = sum(sigmoid(W*x));
g = dlgradient(loss, W);
fprintf('analytic dL/dW = [%.5f %.5f; %.5f %.5f]\n', g(1,1), g(1,2), g(2,1), g(2,2));

ep = 1e-6;
% ... central difference over each W(i,j), comparing to g(i,j) ...
fprintf('max |analytic - finite-diff| = %.2e\n', maxerr);
```

The recorded tape (`sum`→`sigmoid`→matmul) is differentiated exactly; the max
discrepancy against finite differences lands at ~1e-9, confirming the reverse
pass.

### Training an MLP from scratch  (`examples/dlnet/dl_mlp_train.m`)

The canonical SGD loop: forward → `crossentropy` loss → `dlgradient` per
parameter → manual update via `extractdata`.

```matlab
W1 = dlarray(0.5*randn(8,2)); b1 = dlarray(zeros(8,1));
W2 = dlarray(0.5*randn(3,8)); b2 = dlarray(zeros(3,1));
lr = 0.5;
for it = 1:300
    H = relu(W1*X + b1);
    Y = softmax(W2*H + b2);
    loss = crossentropy(Y, T);
    gW1 = dlgradient(loss, W1);  gb1 = dlgradient(loss, b1);
    gW2 = dlgradient(loss, W2);  gb2 = dlgradient(loss, b2);
    W1 = dlarray(extractdata(W1) - lr*gW1);  b1 = dlarray(extractdata(b1) - lr*gb1);
    W2 = dlarray(extractdata(W2) - lr*gW2);  b2 = dlarray(extractdata(b2) - lr*gb2);
end
```

On a 3-class, 6-sample toy set this drives the loss down and reaches 100% train
accuracy. The pattern — `g = dlgradient(loss, W); W = dlarray(extractdata(W) -
lr*g)` — is the training idiom reused by every other example here.

### End-to-end CNN classifier  (`examples/dlnet/dl_cnn_classifier.m`)

The full conv-net stack on the rank-N tape: conv (3-channel input) → ReLU →
max-pool → flatten → FC → softmax → crossentropy, trained through every layer.

```matlab
for k = 1:80
    Y_conv = conv2d_batch(Xdl, Wconv_dl);   % 4x4x4x4
    Y_relu = relu(Y_conv);
    Y_pool = maxpool2d(Y_relu, 2, 2);       % 2x2x4x4
    Y_flat = reshape(Y_pool, 16, 4);        % 16 x 4
    logits = W2_dl * Y_flat;                % 3 x 4
    yhat   = softmax(logits);
    loss   = crossentropy(yhat, Tdl);
    gWc = dlgradient(loss, Wconv_dl);
    gW2 = dlgradient(loss, W2_dl);
    Wconv_dl = dlarray(extractdata(Wconv_dl) - lr_conv * gWc);
    W2_dl    = dlarray(extractdata(W2_dl)    - lr_fc   * gW2);
end
```

Gradients flow back through `maxpool2d` and `conv2d_batch` (the conv weights are
a 3×3×3×4 tensor), letting the net learn channel-bright signatures. Reaches ≥3/4
training accuracy.

### LSTM memory task — BPTT  (`examples/dlnet/dl_lstm_sequence.m`)

A functional LSTM cell trained on a first-bit-memory task. The `lstm` op is a
single fused tape node; `dlgradient` walks it backward in time (BPTT) over every
gate and state.

```matlab
D = 1; H = 4; T = 6; N = 8;
Wx = dlarray(0.3 * randn(4*H, D));
Wr = dlarray(0.3 * randn(4*H, H));
bL = dlarray(zeros(4*H, 1));
...
Hseq   = lstm(Xn, h0, c0, Wx, Wr, bL);   % H x T
logits = Wy * Hseq + by;                 % 1 x T
p      = sigmoid(mean(logits));          % scalar
loss_n = dlarray(0) - (target * log(p) + oneT * log(oneP));
gWx_acc = gWx_acc + dlgradient(loss_n, Wx);
```

Gradients are accumulated across the 8 sequences, then a batched update is
applied. The forget gate lets the cell carry the first bit across all 6
timesteps; the task reaches 100% accuracy. The sibling
`dl_gru_sequence.m` does the same with the lighter 3-gate `gru`.

### Single-head Transformer encoder block  (`examples/dlnet/dl_transformer_block.m`)

A token classifier with the canonical encoder cell: embed → self-attention →
residual+LayerNorm → FFN → residual+LayerNorm → per-token softmax. LayerNorm is
composed from the shipped small ops (`mean(·,1)`, `sqrt`, `./`).

```matlab
z   = embed(E, toks);              % D x T
Q = Wq * z;  K = Wk * z;  V_ = Wv * z;
QK  = transpose(Q) * K;            % T x T
scl = QK ./ dsq;                   % scale by sqrt(D)
A   = softmax(scl);
ctx = V_ * A;                      % D x T
h1  = z + ctx;                     % residual
mu1 = mean(h1, 1);  diff1 = h1 - mu1;
v1  = mean(diff1 .* diff1, 1);
z2  = diff1 ./ sqrt(v1 + eps_dl);  % LayerNorm
f   = gelu(Wf1 * z2 + b1);         % FFN
```

A single Adam-flavoured step on the output projection `Wo` is shown to drop the
crossentropy loss, demonstrating the gradient flows through the whole block.

### Other examples worth reading

- **`dl_gan.m`** — least-squares GAN with alternating SGD; `mse(d_real, ones) +
  mse(d_fake, zeros)` exercises the classdef-operator-overloading dispatch.
- **`dl_vae.m`** — variational autoencoder with the reparameterisation trick.
- **`dl_dlnetwork.m`** — the `dlnetwork()` carrier + `trainnet` Adam driver
  (`addFC`/`addRelu`/`addSoftmax` → `netPredict` / `trainnet`).
- **`dl_quantize_check.m`** — train in double, `dlquantize` every weight to
  symmetric INT8, verify accuracy holds (the DL-HDL Tier-H1 software on-ramp;
  `dlqscale` reports the INT8 LSB).
- **`dl_onnx_roundtrip.m`** — build an ONNX graph programmatically (Gemm → Relu
  → Gemm → Sigmoid → Add → Mul → Softmax), `onnxSave`/`onnxRead`/`onnxRun`, and
  match a hand-computed reference to 1e-5.
- **`dl_autoencoder.m`, `dl_attention.m`, `dl_mha_train.m`, `dl_layernorm.m`,
  `dl_dropout.m`, `dl_gradcam.m`, `dl_neural_ode.m`** — further architecture and
  attribution demos on the same tape.

## Limitations & carve-outs

- Plain-matrix bias **column-broadcast** isn't wired outside the autodiff
  (inside the tape it's handled; in plain-matrix inference apply the bias per
  column explicitly, as `dl_quantize_check.m` does).
- `extractdata` of a scalar returns a 1×1 matrix, not a bare `double` — index it
  (`Lv(1)`) before a scalar comparison.
- Carved (Tier-C-dependent): multi-head attention, full object-array
  `dlnetwork` layer containers, conv with the SSCB labeled-dimension format.
- DL on GPU (cuDNN / TensorRT / MPSCNN) and real pretrained-weight import are
  out of scope for this lane.

## See also

- Roadmap / design: [`../deep_learning_toolbox_roadmap.md`](../deep_learning_toolbox_roadmap.md)
- Examples: `examples/dlnet/`
