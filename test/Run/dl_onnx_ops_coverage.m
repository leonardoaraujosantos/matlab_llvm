% dl_onnx_ops_coverage — H5 follow-on: broader op coverage on a single
% ONNX graph.  Exercises ops not covered by the roundtrip headline:
%   Conv, MaxPool, AveragePool, Tanh, LeakyRelu, Concat, Transpose,
%   ReduceSum, LayerNormalization, BatchNormalization, MatMul, LogSoftmax,
%   GlobalAveragePool.
%
% Graph (input X 4x4):
%   P1   = Conv(X, K1)               -> 2x2
%   P2   = MaxPool(P1, k=2, s=1)     -> 1x1
%   P2b  = AveragePool(P1, k=2,s=1)  -> 1x1
%   M    = MatMul(W2x2, P1)          -> 2x2
%   T    = Transpose(M)              -> 2x2
%   B    = BatchNormalization(T, g, b, mu, var)
%   L    = LayerNormalization(B, scale, bias)
%   A    = Tanh(L)
%   K    = LeakyRelu(A, alpha=0.05)
%   C    = Concat(P2, P2b, axis=1)   -> 1x2
%   R    = ReduceSum(K)              -> 1x1
%   LS   = LogSoftmax(C, axis=1)     -> 1x2 (output 1)
%   G    = GlobalAveragePool(K)      -> 1x1 (output 2)
% Final outputs are concatenated row-vector: [LS(1,:), G(1,1), R(1,1)].

% ---- Constants.
K1 = [0.25 0.50 0.25;
      0.10 0.20 0.10;
      0.05 0.10 0.05];
W2x2 = [1.0  0.5;
        0.5  1.0];
gamma_bn = [1.0 1.0]; beta_bn = [0.0 0.0];
mu_bn    = [0.0 0.0]; var_bn  = [1.0 1.0];
scale_ln = [1.0 1.0]; bias_ln = [0.0 0.0];

% ---- Build the model.
onnxNewModel();
onnxAddInit('K1',   K1);
onnxAddInit('W2',   W2x2);
onnxAddInit('g_bn', gamma_bn);
onnxAddInit('b_bn', beta_bn);
onnxAddInit('m_bn', mu_bn);
onnxAddInit('v_bn', var_bn);
onnxAddInit('s_ln', scale_ln);
onnxAddInit('b_ln', bias_ln);
onnxSetInput('X', [4 4]);
onnxSetOutput('Y');

onnxBeginNode('Conv');
onnxNodeInput('X'); onnxNodeInput('K1'); onnxNodeOutput('P1');
onnxEndNode();

onnxBeginNode('MaxPool');
onnxNodeInput('P1'); onnxNodeOutput('P2');
onnxNodeAttrInts('kernel_shape', [2 2]);
onnxNodeAttrInts('strides',      [1 1]);
onnxEndNode();

onnxBeginNode('AveragePool');
onnxNodeInput('P1'); onnxNodeOutput('P2b');
onnxNodeAttrInts('kernel_shape', [2 2]);
onnxNodeAttrInts('strides',      [1 1]);
onnxEndNode();

onnxBeginNode('MatMul');
onnxNodeInput('W2'); onnxNodeInput('P1'); onnxNodeOutput('M');
onnxEndNode();

onnxBeginNode('Transpose');
onnxNodeInput('M'); onnxNodeOutput('T');
onnxEndNode();

onnxBeginNode('BatchNormalization');
onnxNodeInput('T'); onnxNodeInput('g_bn'); onnxNodeInput('b_bn');
onnxNodeInput('m_bn'); onnxNodeInput('v_bn');
onnxNodeOutput('B');
onnxEndNode();

onnxBeginNode('LayerNormalization');
onnxNodeInput('B'); onnxNodeInput('s_ln'); onnxNodeInput('b_ln');
onnxNodeOutput('L');
onnxEndNode();

onnxBeginNode('Tanh');
onnxNodeInput('L'); onnxNodeOutput('A');
onnxEndNode();

onnxBeginNode('LeakyRelu');
onnxNodeInput('A'); onnxNodeOutput('K');
onnxNodeAttrFloat('alpha', 0.05);
onnxEndNode();

onnxBeginNode('Concat');
onnxNodeInput('P2'); onnxNodeInput('P2b'); onnxNodeOutput('C');
onnxNodeAttrInt('axis', 1);
onnxEndNode();

onnxBeginNode('ReduceSum');
onnxNodeInput('K'); onnxNodeOutput('R');
onnxEndNode();

onnxBeginNode('LogSoftmax');
onnxNodeInput('C'); onnxNodeOutput('LS');
onnxNodeAttrInt('axis', 1);
onnxEndNode();

onnxBeginNode('GlobalAveragePool');
onnxNodeInput('K'); onnxNodeOutput('G');
onnxEndNode();

% Final glue: Concat to assemble a single-row output.  We chain Concat
% axis=1 over (LS [1x2], G [1x1], R [1x1]) -> 1x4 vector.
onnxBeginNode('Concat');
onnxNodeInput('LS'); onnxNodeInput('G'); onnxNodeInput('R'); onnxNodeOutput('Y');
onnxNodeAttrInt('axis', 1);
onnxEndNode();

onnxSave('/tmp/dl_onnx_ops_coverage.onnx');
m = onnxRead('/tmp/dl_onnx_ops_coverage.onnx');

fprintf('dl_onnx_ops_coverage: handle=%.0f, nodes=%.0f, inits=%.0f\n', ...
        m(1), onnxNumNodes(m), onnxNumInits(m));

% ---- Input + run.
X = [1.0  2.0  3.0  4.0;
     0.5  1.0  1.5  2.0;
     0.25 0.5  0.75 1.0;
     0.1  0.2  0.3  0.4];

Y = onnxRun(m, X);
fprintf('dl_onnx_ops_coverage: Y = [%.4f %.4f %.4f %.4f]\n', ...
        Y(1, 1), Y(1, 2), Y(1, 3), Y(1, 4));

% Soundness checks — softmax probabilities sum to 1, no NaNs, finite values.
ps = exp(Y(1, 1)) + exp(Y(1, 2));
ok_softmax = abs(ps - 1.0) < 1e-6;
ok_gp = Y(1, 3) > -10.0 && Y(1, 3) < 10.0;
ok_rs = Y(1, 4) > -100.0 && Y(1, 4) < 100.0;

fprintf('dl_onnx_ops_coverage: softmax sum = %.6f (target 1)\n', ps);
fprintf('dl_onnx_ops_coverage: global-avg = %.4f\n', Y(1, 3));
fprintf('dl_onnx_ops_coverage: reduce-sum = %.4f\n', Y(1, 4));

if ok_softmax && ok_gp && ok_rs && onnxNumNodes(m) == 14
    fprintf('dl_onnx_ops_coverage: PASS\n');
else
    fprintf('dl_onnx_ops_coverage: FAIL\n');
end
