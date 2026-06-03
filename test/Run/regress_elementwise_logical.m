% Regression: element-wise & / | on vectors/matrices must operate per element
% (MATLAB), not collapse both operands to a scalar truth value. Previously
% `[1 0 1 0] & [1 1 0 0]` returned a scalar 0. Scalar & / | is unchanged. (#151)
a = [1 0 1 0];
b = [1 1 0 0];
c = a & b;                 % element-wise AND -> [1 0 0 0]
fprintf('and: %.0f %.0f %.0f %.0f\n', c(1), c(2), c(3), c(4));
d = a | b;                 % element-wise OR -> [1 1 1 0]
fprintf('or: %.0f %.0f %.0f %.0f\n', d(1), d(2), d(3), d(4));
e = [1 0 2 0] & 1;         % matrix & scalar -> [1 0 1 0]
fprintf('ms: %.0f %.0f %.0f %.0f\n', e(1), e(2), e(3), e(4));
m = [1 0; 0 3];
g = m | [0 0; 1 0];        % matrix | matrix -> [1 0; 1 1]
fprintf('mm: %.0f %.0f %.0f %.0f\n', g(1,1), g(1,2), g(2,1), g(2,2));
if (3 > 2) & (1 > 0); disp(1); else; disp(0); end   % scalar & unchanged -> 1
