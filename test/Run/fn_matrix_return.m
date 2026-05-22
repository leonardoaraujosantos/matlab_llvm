% Matrix-returning user functions are now usable and indexable.
% The func-boundary tensor->ptr fix (RefineFuncSigs settles the
% tensor->ptr result type, then a follow-up LowerTensorOps sweep retypes
% the caller's slot and lowers its matrix uses) plus interprocedural
% funcReturns3D (routes 3-D results through the *3 helpers).

% 2-D matrix return: index and use in arithmetic
A = mk2d();
fprintf('mk %.0fx%.0f v=%.0f\n', size(A,1), size(A,2), A(2,2));
B = A + 1;
fprintf('add v=%.0f\n', B(2,2));
C = A * 2;
fprintf('mul v=%.0f\n', C(2,2));

% 3-D matrix return: 3-D index + size/numel/ndims + element store
V = makeVol(4);
fprintf('vol d=%.0f n=%.0f nd=%.0f\n', size(V,3), numel(V), ndims(V));
V(1,1,4) = 88;
fprintf('vol3 v=%.0f\n', V(1,1,4));

% 3-D produced by cat(3,...) inside the function
W = scaleVol();
fprintf('w d=%.0f v=%.0f\n', size(W,3), W(1,1,2));

% param-dependent 3-D return: procVol(x) is 3-D iff its argument is 3-D
% (argument-flow funcReturns3D seeds the param from the call-site arg).
P = procVol(W);
fprintf('pv d=%.0f v=%.0f\n', size(P,3), P(1,1,2));

% scalar-returning function is unaffected by the boundary change
s = addup(3, 4);
fprintf('s=%.0f\n', s);

function r = mk2d()
    r = ones(3,3) * 5;
    r(2,2) = 42;
end
function r = makeVol(n)
    r = zeros(2, 2, n);
end
function r = scaleVol()
    r = cat(3, ones(2,2)*10, ones(2,2)*20, ones(2,2)*30);
end
function r = addup(a, b)
    r = a + b;
end
function o = procVol(x)
    o = imadd(x, x);
end
