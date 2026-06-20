% Regression for #330: string-mode builtin call shapes that previously failed
% lowering with "unsupported call shape".
%   sum(X,'all')      -> whole-array sum (a scalar), not the column-wise sum
%   norm(X,'fro')     -> Frobenius norm over all elements
%   zeros(sz,'like',A)-> zeros with the prototype's type (double on the CPU lane)
% fprintf keeps scalar formatting identical across the emit backends.
A = [1 2; 3 4];
fprintf('%g\n', sum(A, 'all'));     % 10
fprintf('%g\n', norm(A, 'fro'));    % sqrt(30) = 5.47723
B = zeros(2, 3, 'like', A);
fprintf('%g %g\n', size(B, 1), size(B, 2));   % 2 3
fprintf('%g\n', sum(B, 'all'));     % 0
v = [3 4];
fprintf('%g\n', norm(v, 'fro'));    % 5 (vector Frobenius == 2-norm)
