% Regression: vector element deletion x(idx)=[] must remove the element(s) and
% shrink the vector (MATLAB), preserving orientation. Previously a no-op.
% Covers scalar / `end` / variable / range / index-vector / logical-mask (#188).
x = [1 2 3 4 5];
x(2) = [];                         % -> [1 3 4 5]
fprintf('mid: %.0fx%.0f vals=%.0f %.0f %.0f %.0f\n', ...
        size(x,1), size(x,2), x(1), x(2), x(3), x(4));
y = [1 2 3 4 5];
y(end) = [];                       % pop last -> [1 2 3 4]
fprintf('end: %.0fx%.0f last=%.0f\n', size(y,1), size(y,2), y(4));
z = [10 20 30 40];
k = 3;
z(k) = [];                         % variable index -> [10 20 40]
fprintf('var: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(z,1), size(z,2), z(1), z(2), z(3));
c = [1; 2; 3; 4];
c(2) = [];                         % column stays column -> [1; 3; 4]
fprintf('col: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(c,1), size(c,2), c(1), c(2), c(3));
r = [1 2 3 4 5];
r(2:3) = [];                       % range -> [1 4 5]
fprintf('rng: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(r,1), size(r,2), r(1), r(2), r(3));
v = [10 20 30 40 50];
ix = [1 4];
v(ix) = [];                        % index vector -> [20 30 50]
fprintf('vec: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(v,1), size(v,2), v(1), v(2), v(3));
w = [1 2 3 4 5];
w(w > 3) = [];                     % logical mask -> [1 2 3]
fprintf('msk: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(w,1), size(w,2), w(1), w(2), w(3));
