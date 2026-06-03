% Regression: element-wise logical NOT ~ on vectors/matrices must negate per
% element (MATLAB), not collapse the operand to a scalar truth value.
% Previously `~[1 0 2 0]` returned a scalar 1. Scalar ~ is unchanged. (#200)
a = ~[1 0 2 0];            % -> [0 1 0 1]
fprintf('vec: %.0f %.0f %.0f %.0f\n', a(1), a(2), a(3), a(4));
c = ~[1; 0; 3];            % column stays column -> [0; 1; 0]
fprintf('col: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(c,1), size(c,2), c(1), c(2), c(3));
m = ~[1 0; 0 1];           % -> [0 1; 1 0]
fprintf('mat: %.0f %.0f %.0f %.0f\n', m(1,1), m(1,2), m(2,1), m(2,2));
if ~5; disp(1); else; disp(0); end         % ~5 == 0 -> else -> 0
if ~0; disp(1); else; disp(0); end         % ~0 == 1 -> 1
x = 0; if ~x; disp(42); end                % scalar condition unchanged -> 42
g = ~[1 0 1 0] & [1 1 1 1];                % combines with element-wise & -> [0 1 0 1]
fprintf('combo: %.0f %.0f %.0f %.0f\n', g(1), g(2), g(3), g(4));
