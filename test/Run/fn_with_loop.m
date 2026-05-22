% for-loops inside user functions, including VARIABLE param bounds
% (`for k = 1:n`, n a parameter), building a scalar, a 2-D matrix, and a
% 3-D array, with the result used by the caller.  Param-bound loops in
% functions were previously left unlowered (the range bound was still
% `none`-typed when seq-loop lowering ran); a re-run after param refinement
% now lowers them.
s = accScalar(5);
fprintf('scalar s=%.0f\n', s);

M = buildMat(3);
fprintf('mat %.0fx%.0f v=%.0f\n', size(M,1), size(M,2), M(3,3));

V = buildVol(3);
fprintf('vol d=%.0f v=%.0f n=%.0f\n', size(V,3), V(1,1,3), numel(V));

function r = accScalar(n)
    r = 0;
    for k = 1:n
        r = r + k;
    end
end
function r = buildMat(n)
    r = zeros(n,n);
    for k = 1:n
        r(k,k) = k;
    end
end
function r = buildVol(n)
    r = zeros(2,2,n);
    for k = 1:n
        r(:,:,k) = k;
    end
end
