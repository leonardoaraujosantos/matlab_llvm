% Regression: bracket concatenation of matrix/vector operands ([a b], [a;b]).
a = [1; 2; 3];
b = [4; 5; 6];
X = [a b];
fprintf('X12=%.0f X32=%.0f rows=%.0f cols=%.0f\n', X(1,2), X(3,2), size(X,1), size(X,2));
Y = [a b a];
fprintf('Y13=%.0f\n', Y(1,3));
V = [a; b];
fprintf('V5=%.0f len=%.0f\n', V(5), size(V,1));
