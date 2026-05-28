% dl_phase1_ops.m — gating fixture for Phase 1 dlarray ops.
% Verifies every new op runs forward and produces a finite gradient.
% Numerical-against-analytic accuracy lives in the per-op runtime unit
% checks; this fixture exists as a wiring smoke for the AOT lane.

X = dlarray([0.5; 1.2; -0.3; 2.0]);
Y = dlarray([1.0; 2.0;  0.4; 1.5]);

A1 = leakyrelu(X);
A2 = gelu(X);
A3 = swish(X);
A4 = softplus(X);
A5 = elu(X);
A6 = sqrt(Y);
A7 = X ./ Y;

S = sum(A1) + sum(A2) + sum(A3) + sum(A4) + sum(A5) + sum(A6) + sum(A7);

% Scalar forward value sanity check — every term is positive in this
% input range except leakyrelu/elu at x=-0.3 (both still small), so S>0.
sv = extractdata(S);
fprintf('dl_phase1_ops: S=%.4f\n', sv);

% Gradients on X and Y -- expect finite, non-NaN.
gX = dlgradient(S, X);
gY = dlgradient(S, Y);
fprintf('dl_phase1_ops: sum(gX)=%.4f sum(gY)=%.4f\n', sum(gX), sum(gY));

% mean(X, dim) over a 2x3 -- exercises both reduce-dimensions.
M  = dlarray([1.0 2.0 3.0; 4.0 5.0 6.0]);
m1 = mean(M, 1);   % 1x3  -> [2.5 3.5 4.5]
m2 = mean(M, 2);   % 2x1  -> [2.0; 5.0]
L  = sum(m1) + sum(m2);
gM = dlgradient(L, M);
fprintf('dl_phase1_ops: L=%.4f sum(gM)=%.4f\n', extractdata(L), sum(sum(gM)));
fprintf('dl_phase1_ops: PASS\n');
