% Regression: double()/single() of a vector/matrix must work (identity cast —
% the runtime is uniformly f64). Previously only the scalar form was supported;
% a matrix operand errored 'unsupported call shape'. Covers the common
% double(logicalResult) idiom. (#202)
a = double([1 2 3]);
fprintf('dvec: %.0f %.0f %.0f\n', a(1), a(2), a(3));
x = [1 2 3 4];
b = double(x > 2);                 % logical-to-double -> [0 0 1 1]
fprintf('dcmp: %.0f %.0f %.0f %.0f\n', b(1), b(2), b(3), b(4));
c = single([1.5 2.5]);
fprintf('svec: %.1f %.1f\n', c(1), c(2));
d = double([1 2; 3 4]);            % shape preserved
fprintf('shape: %.0fx%.0f\n', size(d,1), size(d,2));
e = double(x > 2) * 2;             % usable in arithmetic -> [0 0 2 2]
fprintf('arith: %.0f %.0f %.0f %.0f\n', e(1), e(2), e(3), e(4));
fprintf('scalar: %.0f\n', double(7));   % scalar form still works
