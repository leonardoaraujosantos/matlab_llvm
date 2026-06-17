# Deep Learning Toolbox Spec

## Purpose
Documents the shipped subset of the Deep Learning Toolbox in the matlab_llvm compiler: a reverse-mode autodiff framework (`dlarray`) supporting inference, custom-loop training, recurrent/attention architectures, quantization/metrics/attribution, fixed-point SystemVerilog emission, and ONNX import/export. Tiers 1-6 plus HDL H1-H5 are marked shipped (2026-05-28). (doc: docs/deep_learning_toolbox_roadmap.md) (src: runtime/toolbox/dlnet)

## Requirements

### Requirement: dlarray autodiff engine
The system SHALL provide a tape-tracked `dlarray` type with differentiable operations and gradient computation. (src: runtime/toolbox/dlnet/dlnet_classdefs.m) (src: runtime/toolbox/dlnet/runtime_dlnet.cpp)

#### Scenario: Compute gradients of a loss
- **WHEN** a program builds an expression over `dlarray` values using differentiable ops (plus/minus/mtimes/times, relu/sigmoid/tanh/softmax/gelu/swish/elu, sum/mean, crossentropy/mse, reshape, layernorm/batchnorm/groupnorm/rmsnorm, conv2d, maxpool/avgpool, embed) and calls `dlgradient` inside `dlfeval`
- **THEN** the system SHALL return adjoints via the reverse sweep, with `extractdata` exposing the numeric value

### Requirement: Network carrier and inference
The system SHALL provide a `dlnetwork` carrier with layer builders and forward inference. (src: runtime/toolbox/dlnet/dlnet_classdefs.m) (src: runtime/toolbox/dlnet/runtime_dlnet.cpp)

#### Scenario: Build a network and predict
- **WHEN** a program builds a `dlnetwork` with `addFC`/`addRelu`/`addSigmoid`/`addTanh`/`addSoftmax`, queries `netNumLayers`, and calls `predict`/`classify`, optionally using `imageDatastore`/`splitEachLabel`/`augmentedImageDatastore`
- **THEN** the system SHALL run the forward pass and return outputs or class labels

### Requirement: Training solvers and driver
The system SHALL provide optimizer update functions and a training driver with options. (src: runtime/toolbox/dlnet/runtime_dlnet.cpp)

#### Scenario: Train with a built-in driver or custom loop
- **WHEN** a program calls `adamupdate`/`sgdmupdate`/`rmspropupdate` in a custom loop, or `trainnet` with `trainingOptions('sgdm'|'adam'|'rmsprop', ...)`
- **THEN** the system SHALL apply the optimizer step or run the mini-batch loop (forward, dlgradient, solver step) and return the trained network

### Requirement: Recurrent, sequence, and attention layers
The system SHALL provide recurrent, embedding, and attention building blocks. (src: runtime/toolbox/dlnet/runtime_dlnet.cpp)

#### Scenario: Run a recurrent or attention model
- **WHEN** a program calls `lstm`, `gru`, `bilstm`, `lstmp`, `embed`, or composes scaled-dot-product / multi-head attention from matmul + softmax
- **THEN** the system SHALL return the sequence output with BPTT gradients available through the tape

### Requirement: Quantization, metrics, and attribution
The system SHALL provide INT8 quantization, classification metrics, and attribution methods. (src: runtime/toolbox/dlnet/runtime_dlnet.cpp)

#### Scenario: Quantize and explain a model
- **WHEN** a program calls `dlquantize`/`dlqscale`/`dlqcalibrate`/`dlqclip`, metrics (`accuracy`/`precision`/`recall`/`fScore`/`rocmetrics`/`aucroc`), or attribution (`gradCAM`/`occlusionSensitivity`/`imageLIME`), pruning (`matlab_dlnet_prune_mask`), or `bayesopt` for hyperparameter search
- **THEN** the system SHALL return the quantized tensor/scale, evaluation metrics, saliency maps, prune masks, or tuned hyperparameters

### Requirement: HDL emission and ONNX interchange
The system SHALL provide fixed-point SystemVerilog emission and ONNX import/export. (doc: docs/deep_learning_toolbox_roadmap.md) (src: runtime/toolbox/dlnet/runtime_onnx.cpp)

#### Scenario: Emit hardware or interchange a model
- **WHEN** a quantized network with `fi`-typed forward is emitted to SystemVerilog, or a program round-trips a model through the ONNX importer/writer
- **THEN** the system SHALL emit synthesizable SV (cocotb bit-accurate) or parse/write ONNX across ~56 op handlers into/from the layer DAG
