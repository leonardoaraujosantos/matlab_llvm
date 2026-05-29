% dl_onnx_roundtrip — H5 gating: ONNX inference-graph importer + builder
% round-trip.
%
% Builds an ONNX model programmatically with a representative mix of ops:
%   Gemm -> Relu -> Gemm -> Sigmoid -> Add -> Mul -> Softmax
% writes it to /tmp, reads it back, runs it on a known input, and verifies
% the output matches a hand-computed reference within tight tolerance.

% Weights for a 3 -> 4 -> 2 MLP (deterministic).
W1 = [0.10 0.20 0.30 0.40;
      0.50 0.60 0.70 0.80;
      0.90 1.00 1.10 1.20];
b1 = [0.01 0.02 0.03 0.04];

W2 = [0.10 0.20;
      0.30 0.40;
      0.50 0.60;
      0.70 0.80];
b2 = [0.05 0.10];

% A small per-class bias added after Sigmoid.
bias = [0.1 0.2];
scale = [1.5 0.5];

% ---- Build the ONNX model.
onnxNewModel();
onnxAddInit('W1',    W1);
onnxAddInit('b1',    b1);
onnxAddInit('W2',    W2);
onnxAddInit('b2',    b2);
onnxAddInit('bias',  bias);
onnxAddInit('scale', scale);
onnxSetInput('X', [1 3]);
onnxSetOutput('Y');

% Node 1: T1 = Gemm(X, W1, b1)  ->  (1x3) * (3x4) + (1x4)
onnxBeginNode('Gemm');
onnxNodeInput('X'); onnxNodeInput('W1'); onnxNodeInput('b1');
onnxNodeOutput('T1');
onnxEndNode();

% Node 2: T2 = Relu(T1)
onnxBeginNode('Relu');
onnxNodeInput('T1'); onnxNodeOutput('T2');
onnxEndNode();

% Node 3: T3 = Gemm(T2, W2, b2)  ->  (1x4) * (4x2) + (1x2)
onnxBeginNode('Gemm');
onnxNodeInput('T2'); onnxNodeInput('W2'); onnxNodeInput('b2');
onnxNodeOutput('T3');
onnxEndNode();

% Node 4: T4 = Sigmoid(T3)
onnxBeginNode('Sigmoid');
onnxNodeInput('T3'); onnxNodeOutput('T4');
onnxEndNode();

% Node 5: T5 = Add(T4, bias)
onnxBeginNode('Add');
onnxNodeInput('T4'); onnxNodeInput('bias'); onnxNodeOutput('T5');
onnxEndNode();

% Node 6: T6 = Mul(T5, scale)
onnxBeginNode('Mul');
onnxNodeInput('T5'); onnxNodeInput('scale'); onnxNodeOutput('T6');
onnxEndNode();

% Node 7: Y = Softmax(T6, axis=1)  per-row normalisation
onnxBeginNode('Softmax');
onnxNodeInput('T6'); onnxNodeOutput('Y');
onnxNodeAttrInt('axis', 1);
onnxEndNode();

% Persist and reload.
onnxSave('/tmp/dl_onnx_roundtrip.onnx');
m = onnxRead('/tmp/dl_onnx_roundtrip.onnx');

% Introspection.
n_nodes = onnxNumNodes(m);
n_inits = onnxNumInits(m);
fprintf('dl_onnx_roundtrip: handle = %.0f, nodes = %.0f, inits = %.0f\n', ...
        m(1), n_nodes, n_inits);

% ---- Hand-computed reference.
X = [1.0 -0.5 2.0];   % 1x3
ref_T1 = X * W1 + b1;
ref_T2 = zeros(1, 4);
for k = 1:4
    if ref_T1(k) > 0
        ref_T2(k) = ref_T1(k);
    end
end
ref_T3 = ref_T2 * W2 + b2;
ref_T4 = zeros(1, 2);
for k = 1:2
    ref_T4(k) = 1.0 / (1.0 + exp(-ref_T3(k)));
end
ref_T5 = ref_T4 + bias;
ref_T6 = ref_T5 .* scale;
mx = ref_T6(1);
if ref_T6(2) > mx, mx = ref_T6(2); end
e1 = exp(ref_T6(1) - mx);
e2 = exp(ref_T6(2) - mx);
ref_Y_1 = e1 / (e1 + e2);
ref_Y_2 = e2 / (e1 + e2);

% ---- Run through the ONNX runtime.
Y = onnxRun(m, X);
y1 = Y(1, 1);
y2 = Y(1, 2);

fprintf('dl_onnx_roundtrip: Y = [%.6f, %.6f]\n', y1, y2);
fprintf('dl_onnx_roundtrip: ref = [%.6f, %.6f]\n', ref_Y_1, ref_Y_2);

err = abs(y1 - ref_Y_1) + abs(y2 - ref_Y_2);
fprintf('dl_onnx_roundtrip: |err| = %.8f\n', err);

if n_nodes == 7 && n_inits == 6 && err < 1e-5
    fprintf('dl_onnx_roundtrip: PASS\n');
else
    fprintf('dl_onnx_roundtrip: FAIL\n');
end
